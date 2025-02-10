import math
import random
import torch
from torch import Tensor, nn
from typing import Dict, Tuple, Union
from mmdet.models.layers.ec_positional_encoding import EC_SinePositionalEncoding

from mmdet.registry import MODELS
from mmdet.models.layers.decoder_layer import EC_TransformerDecoderLayer

@MODELS.register_module()
class FusionLayer(nn.Module):
    def __init__(self,
            ffn_channel,
            dataset_name,
            embed_dims = None,
            group_factor_a = None,
            group_factor_b = None,
            num_heads = 8,
            use_shuffle: bool = False,
            gg_shuffle: bool = False,
        ):
        super().__init__()

        assert group_factor_a in [0, 2**2, 3**2, 4**2, 5**2, 6**2, 7**2, 8**2]
        assert group_factor_b in [0, 2**2, 3**2, 4**2, 5**2, 6**2, 7**2, 8**2]
        # 相等或必有一个为 0
        assert group_factor_a == group_factor_b or group_factor_a * group_factor_b == 0
        
        assert use_shuffle + gg_shuffle != 2   # do not True + True
        # TFM 不受影响，testing 不受影响
        self.use_shuffle = use_shuffle
        self.gg_shuffle = gg_shuffle

        self.group_factor_a = group_factor_a
        self.cell_size_a = int(math.sqrt(group_factor_a))
        self.group_factor_b = group_factor_b
        self.cell_size_b = int(math.sqrt(group_factor_b))

        self.positional_encoding = EC_SinePositionalEncoding(num_feats=embed_dims//2, normalize=True)
        self.decoder_layer_e = EC_TransformerDecoderLayer(embed_dims, num_heads, ffn_channel, 0.1, 'relu',)
        self.decoder_layer_c = EC_TransformerDecoderLayer(embed_dims, num_heads, ffn_channel, 0.1, 'relu',)
        self.decoder_layer_mix = EC_TransformerDecoderLayer(embed_dims, num_heads, ffn_channel, 0.1, 'relu', cross_only=True)
        self.mix_conv = nn.Conv2d(in_channels=embed_dims*2, out_channels=embed_dims, kernel_size=1, stride=1)

        self.dataset_name = dataset_name
        

    def create_index_1d(self, h, w, gf, cs, use_shuffle):

        # if cs = 3: [0, 0, 0, 1, 1, 1, 2, 2, 2]
        x_offset = [i // cs for i in range(gf)]

        # if cs = 3: [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_offset = [i % cs for i in range(gf)]

        def get_coor(cx, cy, i):
            return cx + x_offset[i], cy + y_offset[i]

        def check_coor(x, y):
            return 0 <= x < h and 0 <= y < w
        
        # H, W: 填补后的高和宽
        H = math.ceil(h / cs) * cs
        W = math.ceil(w / cs) * cs

        index_1d = torch.zeros(H * W, dtype=int)

        def save_next(rank, x, y):
            # equals to: index_1d[rank] = index_2d[x][y]
            index_1d[rank] = x * w + y
            return rank + 1

        # cell left-top
        # 对于向下 x，向右 y 的坐标系，在不能整除的边界情况下，left-top 必定存在，right-bottom 必定不存在
        rank = 0
        for cx in range(0, H, cs):
            for cy in range(0, W, cs):
                # right-bottom
                x, y = get_coor(cx, cy, gf - 1)
                # full cell
                if check_coor(x, y):
                    for i in range(gf):
                        x, y = get_coor(cx, cy, i)
                        rank = save_next(rank, x, y)
                # not full cell
                else:
                    for i in range(gf):
                        # i = 0 时为 left-top
                        x, y = get_coor(cx, cy, i)
                        if check_coor(x, y):
                            rank = save_next(rank, x, y)
                        else:
                            # 边界填充
                            # 只向右 out of cell
                            if (0 <= x < h) and not (0 <= y < w):
                                rank = save_next(rank, x, w - 1)
                            # 只向下 out of cell
                            elif not (0 <= x < h) and (0 <= y < w):
                                rank = save_next(rank, h - 1, y)
                            # 左下角边界情况，同时向右，向下 out of cell
                            else:
                                rank = save_next(rank, h - 1, w - 1)

        # already got index_1d
        if use_shuffle:
            length = index_1d.shape[0]
            shuffle_index = list(range(length))

            for i in range(0, length, gf):
                t = shuffle_index[i:i+gf]
                random.shuffle(t)
                shuffle_index[i:i+gf] = t

            index_1d = index_1d[shuffle_index]

        return index_1d.to("cuda")

    def get_pos_embed(self, pos, bs, dim):
        pos_embed = self.positional_encoding(pos)
        pos_embed = pos_embed.reshape(bs, dim, -1)
        return pos_embed

    def forward(self, e_feat, c_feat):

        # testing 不受影响
        if self.training == False:
            self.gg_shuffle = False
            self.use_shuffle = False

        bs, dim, e_h, e_w = e_feat.shape
        bs, dim, c_h, c_w = c_feat.shape

        e = e_feat.reshape(bs, dim, -1)
        c = c_feat.reshape(bs, dim, -1)

        e_pos = torch.zeros(bs, e_h, e_w).bool().cuda()
        c_pos = torch.zeros(bs, c_h, c_w).bool().cuda()
        e_pos = self.get_pos_embed(e_pos, bs, dim)
        c_pos = self.get_pos_embed(c_pos, bs, dim)

        gf_a = self.group_factor_a
        cs_a = self.cell_size_a
        gf_b = self.group_factor_b
        cs_b = self.cell_size_b


# cross fusion

        e_out = torch.zeros_like(e)
        c_out = torch.zeros_like(c)

        if gf_a:
            e_index_1d = self.create_index_1d(e_h, e_w, gf_a, cs_a, self.use_shuffle)
        if gf_b:
            c_index_1d = self.create_index_1d(c_h, c_w, gf_b, cs_b, self.use_shuffle)

        def group_attention(i, gf, j = None):

            if self.gg_shuffle == False:
                e_index_group = e_index_1d[i::gf]
                c_index_group = c_index_1d[i::gf]
            else:
                e_index_group = e_index_1d[i::gf]
                c_index_group = c_index_1d[j::gf]

            # index_select return copies the indexed fields into new memory location
            e_group = e.index_select(dim=2, index=e_index_group)
            c_group = c.index_select(dim=2, index=c_index_group)
            e_pos_group = e_pos.index_select(dim=2, index=e_index_group)
            c_pos_group = c_pos.index_select(dim=2, index=c_index_group)

            e_out_group = self.decoder_layer_e(query=e_group, key=c_group, query_pos_embed=e_pos_group, key_pos_embed=c_pos_group)
            c_out_group = self.decoder_layer_c(query=c_group, key=e_group, query_pos_embed=c_pos_group, key_pos_embed=e_pos_group)
            
            # slice return a view of a tensor
            e_out[:,:, e_index_group] = e_out_group
            c_out[:,:, c_index_group] = c_out_group


        def group_side_a(i, gf):
            e_index_group = e_index_1d[i::gf]

            # index_select return copies the indexed fields into new memory location
            e_group = e.index_select(dim=2, index=e_index_group)
            e_pos_group = e_pos.index_select(dim=2, index=e_index_group)

            e_out_group = self.decoder_layer_e(query=e_group, key=c, query_pos_embed=e_pos_group, key_pos_embed=c_pos)
            
            nonlocal c_out
            c_out = self.decoder_layer_c(query=c, key=e_out_group, query_pos_embed=c_pos, key_pos_embed=e_pos_group)
            
            # slice return a view of a tensor
            e_out[:,:, e_index_group] = e_out_group

        def group_side_b(i, gf):
            c_index_group = c_index_1d[i::gf]

            # index_select return copies the indexed fields into new memory location
            c_group = c.index_select(dim=2, index=c_index_group)
            c_pos_group = c_pos.index_select(dim=2, index=c_index_group)

            c_out_group = self.decoder_layer_c(query=c_group, key=e, query_pos_embed=c_pos_group, key_pos_embed=e_pos)
            
            nonlocal e_out
            e_out = self.decoder_layer_e(query=e, key=c_out_group, query_pos_embed=e_pos, key_pos_embed=c_pos_group)
            
            # slice return view of a tensor
            c_out[:,:, c_index_group] = c_out_group
        

# mix fusion
        # already got e_out, c_out

        mix_feat = None

        def group_mix(i, gf, j = None):

            if self.gg_shuffle == False:
                e_index_group = e_index_1d[i::gf]
                c_index_group = c_index_1d[i::gf]
            else:
                e_index_group = e_index_1d[i::gf]
                c_index_group = c_index_1d[j::gf]

            # index_select return copies the indexed fields into new memory location
            e_group = e_out.index_select(dim=2, index=e_index_group)
            c_group = c_out.index_select(dim=2, index=c_index_group)
            e_pos_group = e_pos.index_select(dim=2, index=e_index_group)
            c_pos_group = c_pos.index_select(dim=2, index=c_index_group)

            if self.dataset_name == 'DSEC':
                out_group = self.decoder_layer_mix(query=e_group, key=c_group, query_pos_embed=e_pos_group, key_pos_embed=c_pos_group)
                # slice return a view of a tensor
                mix_feat[:,:, e_index_group] = out_group
            elif self.dataset_name == 'DSEC-Soft':
                out_group = self.decoder_layer_mix(query=c_group, key=e_group, query_pos_embed=c_pos_group, key_pos_embed=e_pos_group)
                mix_feat[:,:, c_index_group] = out_group
        
        def group_mix_side_a(i, gf):
            # NOTE: this means grouping on small-flow
            e_index_group = e_index_1d[i::gf]

            # index_select return copies the indexed fields into new memory location
            e_group = e_out.index_select(dim=2, index=e_index_group)
            e_pos_group = e_pos.index_select(dim=2, index=e_index_group)

            e_out_group = self.decoder_layer_mix(query=c_out, key=e_group, query_pos_embed=c_pos, key_pos_embed=e_pos_group)
            
            # slice return a view of a tensor
            nonlocal mix_feat
            mix_feat += e_out_group
        
        def group_mix_side_b(i, gf):
            c_index_group = c_index_1d[i::gf]

            # index_select return copies the indexed fields into new memory location
            c_group = c_out.index_select(dim=2, index=c_index_group)
            c_pos_group = c_pos.index_select(dim=2, index=c_index_group)

            c_out_group = self.decoder_layer_mix(query=c_group, key=e_out, query_pos_embed=c_pos_group, key_pos_embed=e_pos)
            
            # slice return view of a tensor
            mix_feat[:,:, c_index_group] = c_out_group


# concat

        def concat(mix_feat, out, h, w):
            mix_feat = mix_feat.reshape(bs, dim, h, w)
            out = out.reshape(bs, dim, h, w)
            feat = torch.cat((mix_feat, out), dim=1)
            return feat
            

# processes control
        

        # 0 0 
        if gf_a == 0 and gf_b == 0:
            # TFM 不受 shuffle 和 gg_shuffle 影响
            e_out = self.decoder_layer_e(query=e, key=c, query_pos_embed=e_pos, key_pos_embed=c_pos)
            c_out = self.decoder_layer_c(query=c, key=e, query_pos_embed=c_pos, key_pos_embed=e_pos)

            if self.dataset_name == 'DSEC':
                mix_feat = self.decoder_layer_mix(query=e_out, key=c_out, query_pos_embed=e_pos, key_pos_embed=c_pos)
                del c_out
                feat = concat(mix_feat, e_out, e_h, e_w)
                del e_out, mix_feat
            elif self.dataset_name == 'DSEC-Soft':
                mix_feat = self.decoder_layer_mix(query=c_out, key=e_out, query_pos_embed=c_pos, key_pos_embed=e_pos)
                del e_out
                feat = concat(mix_feat, c_out, c_h, c_w)
                del c_out, mix_feat
        else:

            # NOTE final output shape is same with:
            # 1. e (default)
            # 2. the shape of the no group side.
            if gf_a * gf_b:
                if self.dataset_name == 'DSEC':
                    mix_feat = torch.zeros_like(e)
                elif self.dataset_name == 'DSEC-Soft':
                    mix_feat = torch.zeros_like(c)
            elif gf_a:
                mix_feat = torch.zeros_like(c)
            elif gf_b:
                mix_feat = torch.zeros_like(e)
            else:
                assert 0

            gf = max(gf_a, gf_b)

            if self.gg_shuffle == False:
                # i-th group
                for i in range(gf):
                    # 9 9
                    if gf_a * gf_b:
                        group_attention(i, gf)
                        group_mix(i, gf)
                    # 9 0
                    elif gf_a:
                        group_side_a(i, gf)
                        group_mix_side_a(i, gf)
                    # 0 9
                    elif gf_b:
                        group_side_b(i, gf)
                        group_mix_side_b(i, gf)
                    else:
                        assert 0
            else:
                group_a = list(range(gf))
                random.shuffle(group_a)
                group_b = list(range(gf))
                random.shuffle(group_b)

                # 9 9
                for i in range(gf):
                    group_attention(group_a[i], gf, group_b[i])
                    group_mix(group_a[i], gf, group_b[i])
            
            if gf_a * gf_b:
                if self.dataset_name == 'DSEC':
                    del c_out
                    feat = concat(mix_feat, e_out, e_h, e_w)
                    del e_out, mix_feat
                elif self.dataset_name == 'DSEC-Soft':
                    del e_out
                    feat = concat(mix_feat, c_out, c_h, c_w)
                    del c_out, mix_feat
            elif gf_a:
                del e_out
                mix_feat /= gf_a
                feat = concat(mix_feat, c_out, c_h, c_w)
                del c_out, mix_feat
            elif gf_b:
                del e_out
                feat = concat(mix_feat, c_out, c_h, c_w)
                del c_out, mix_feat
            else:
                assert 0

        feat = self.mix_conv(feat)

        return feat
    
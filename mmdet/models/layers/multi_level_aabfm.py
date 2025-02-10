import torch
import torch.nn as nn

import math
import torch.nn.functional as F
from mmdet.registry import MODELS

from cmath import sqrt

class AABFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Fusion module
        self.d_model = dim
        self.aps_linear1 = nn.Linear(dim, dim)
        self.aps_linear2 = nn.Linear(dim, dim)
        self.dvs_linear1 = nn.Linear(dim, dim)
        self.dvs_linear2 = nn.Linear(dim, dim)

    def forward(self, rgb, evt):

        '''
        param aps: (bs, H_tW_t, hd)
        param dvs: (bs, H_tW_t, hd)
        '''
        bs, d, H, W = rgb.shape

        aps = rgb.reshape(bs, d, H*W).transpose(1, 2)
        dvs = evt.reshape(bs, d, H*W).transpose(1, 2)

        ###### AABFM begin
        
        aps_query = self.aps_linear1(aps) # (bs, H_tW_t, hd)
        aps_key = self.aps_linear2(aps) # (bs, H_tW_t, hd)
        aps_weight = torch.div(torch.matmul(aps_query, torch.transpose(aps_key, dim0=1, dim1=2)), sqrt(self.d_model)).float() # (bs, H_tW_t, H_tW_t)
        aps_value = torch.sum(aps_weight, dim=-1) # (bs, H_tW_t)

        dvs_query = self.dvs_linear1(dvs)
        dvs_key = self.dvs_linear2(dvs)
        dvs_weight = torch.div(torch.matmul(dvs_query, torch.transpose(dvs_key, dim0=1, dim1=2)), sqrt(self.d_model)).float() # (bs, H_tW_t, H_tW_t)
        dvs_value = torch.sum(dvs_weight, dim=-1) # (bs, H_tW_t)

        fusion_weight = torch.stack([aps_value, dvs_value], dim=-1) # (bs, H_tW_t, 2)
        fusion_value = F.softmax(fusion_weight, -1) # (bs, H_tW_t, 2)
        value_list = torch.split(fusion_value, 1, -1) # list, element of shape (bs, H_tW_t, 1), len = 2

        fused_src = value_list[0] * aps + value_list[1] * dvs

        # averaging
        # fused_src = 0.5 * aps + 0.5 * dvs

        # concatenation
        # fused_src = torch.cat([aps, dvs], dim=-1)
        
        fused_src = fused_src.transpose(1, 2).reshape(bs, d, H, W)

        return fused_src


@MODELS.register_module()
class MultiLevelAABFM(nn.Module):
    def __init__(self, dims = [512, 1024, 2048]):
        super().__init__()
        self.aabfm_list = nn.ModuleList([
            AABFM(dims[0]),
            AABFM(dims[1]),
            AABFM(dims[2])
        ])

    def forward(self, e_feats, c_feats):

        feats = []
        for i, aabfm in enumerate(self.aabfm_list):
            feats.append(aabfm(c_feats[i], e_feats[i]))
        
        return feats
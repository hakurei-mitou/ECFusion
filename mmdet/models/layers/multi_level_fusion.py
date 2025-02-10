import math
import torch
from torch import Tensor, nn
from typing import Dict, List, Tuple, Union

from mmdet.registry import MODELS
from mmdet.models.layers.decoder_layer import EC_TransformerDecoderLayer

@MODELS.register_module()
class MultiLevelFusion(nn.Module):
    def __init__(self,
            fusion_layer,
            group_factors_a = [],
            group_factors_b = [],
            embed_dims: List = [],
            ffn_channel: List = [],
        ):
        super().__init__()

        assert len(group_factors_a) > 0 and len(group_factors_b) > 0
        assert len(group_factors_a) == len(group_factors_b)
        self.group_factors_a = group_factors_a
        self.group_factors_b = group_factors_b

        self.fusion_layers = nn.ModuleList()
        for i in range(len(group_factors_a)):
            fusion_layer.update(group_factor_a=group_factors_a[i])
            fusion_layer.update(group_factor_b=group_factors_b[i])

            fusion_layer.update(embed_dims=embed_dims[i])
            fusion_layer.update(ffn_channel=ffn_channel[i])

            self.fusion_layers.append(MODELS.build(fusion_layer))

    def forward(self, e_feats, c_feats):
        feats = []
        for fl, e, c in zip(self.fusion_layers, e_feats, c_feats):
            feats.append(fl.forward(e, c))
        feats = tuple(feats)
        return feats

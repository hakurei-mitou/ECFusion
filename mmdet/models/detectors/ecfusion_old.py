from typing import Dict, Tuple, Union
from mmdet.models.dense_heads.ec_centernet_head import EC_CenterNetHead
from mmdet.models.utils.misc import multi_apply
from mmdet.structures.det_data_sample import SampleList

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import OptSampleList
from mmdet.models.detectors.base import BaseDetector
import logging

@MODELS.register_module()
class ECFusionOld(BaseDetector):
    def __init__(self,
                 event_backbone: ConfigType,
                 color_backbone: ConfigType,
                 heatmap_head: ConfigType,
                 decoder_layer: ConfigType,
                 event_neck: ConfigType = None,
                 # 'backbone' and 'neck' just for the init_detector's requirement
                 backbone: ConfigType = None,
                 neck: ConfigType = None,
                 flow: ConfigType = dict(only_event=False, only_color=True, fusion=False),
                 freeze: ConfigType = dict(freeze_event=False),
                 data_preprocessor: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.only_event = flow['only_event']
        self.only_color = flow['only_color']
        self.fusion = flow['fusion']
        self.only_heatmap = flow['only_heatmap']
        self.one_level = flow['one_level']
        assert self.only_event + self.only_color + self.fusion == 1, 'only one in this flow type can affect'

        if self.only_event:
            self.event_backbone = MODELS.build(event_backbone)
        elif self.only_color:
            self.color_backbone = MODELS.build(color_backbone)
        elif self.fusion:
            self.event_backbone = MODELS.build(event_backbone)
            self.color_backbone = MODELS.build(color_backbone)

        if event_neck is not None:
            self.event_neck = MODELS.build(event_neck)

        self.heatmap_head = MODELS.build(heatmap_head)

        if not self.only_heatmap:
            self.first_decoder_layer = MODELS.build(decoder_layer)

        self.freeze_event = freeze['freeze_event']
        self.event_checkpoint = freeze['event_checkpoint']

        self.freeze_weights()

    def freeze_single(self, module, pt):
        module.load_state_dict(pt['state_dict'], strict=False)
        for para in module.parameters():
            para.required_grad = False

    def freeze_weights(self):
        if self.freeze_event:
            pt = torch.load(self.event_checkpoint)
            self.freeze_single(self.event_backbone, pt)
            self.freeze_single(self.event_neck, pt)
            self.freeze_single(self.heatmap_head, pt)

    def extract_color_feat(self, batch_inputs: Tensor):
        x = self.color_backbone(batch_inputs)
        if self.color_neck:
            x = self.color_neck(x)
        return x

    def extract_event_feat(self, batch_inputs: Tensor):
        x = self.event_backbone(batch_inputs)
        if self.event_neck:
            x = self.event_neck(x)
        return x

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor will be shape (N, C, H ,W).
            feats is a tuple, prepare for the FPN
        """
        event_feats = None
        color_feats = None

        if self.only_event:
            event_feats = self.extract_event_feat(batch_inputs)
        elif self.only_color:
            color_feats = self.extract_color_feat(batch_inputs)
        elif self.fusion:
            event_feats = self.extract_event_feat(batch_inputs)
            color_feats = self.extract_color_feat(batch_inputs)

        return event_feats, color_feats

    def flow_select(self, event_feats, color_feats):
        ''' select:
                E or onlt heatmap or EC with doing fusion
                one level or all level
        '''
        
        if self.one_level:
            event_feats = event_feats[0] if event_feats is not None else None
            color_feats = color_feats[0] if color_feats is not None else None

        feats = None
        if self.only_event:
            feats = event_feats
        elif self.only_color:
            feats = color_feats
        elif self.fusion:
            assert 0, 'not implemented'

        if isinstance(feats, tuple):
            return feats
        else:
            return (feats,)

    def loss(self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        final_loss = {}

        event_feats, color_feats = self.extract_feat(batch_inputs)

        # NOTE feats: [BS, C, H, W]
        feats = self.flow_select(event_feats, color_feats)
        
        losses, heatmaps = self.heatmap_head.loss(feats, batch_data_samples)        

        if not self.freeze_event:
            final_loss.update({f'loss_heatmap': losses['loss_center_heatmap']})

            if isinstance(self.heatmap_head, EC_CenterNetHead):
                final_loss.update({f'loss_wh': losses['loss_wh']})
                final_loss.update({f'loss_offset': losses['loss_offset']})

        if self.only_heatmap:
            return final_loss

        losses = self.first_decoder_layer.loss(feats[0], heatmaps[0].detach().clone(), batch_data_samples)

        # NOTE _10 = first decoder layer level 0
        final_loss.update({f'loss_cls_1': losses['loss_cls'],
                            f'loss_bbox_1': losses['loss_bbox'],
                            f'loss_iou_1': losses['loss_iou']})

        return final_loss

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        event_feats, color_feats = self.extract_feat(batch_inputs)
        
        feats = self.flow_select(event_feats, color_feats)

        if self.only_heatmap:
            # just retain center_heatmap, no need for offset_heatmap and wh_heatmap 
            batch_data_samples, heatmaps = self.heatmap_head.predict(feats, batch_data_samples)
        else:
            heatmaps, _, _ =  self.heatmap_head.forward(feats)
            
        # all image for a level
        for data_sample, img_heatmap in zip(batch_data_samples, heatmaps[0]):
            data_sample.pred_heatmap = img_heatmap
        
        if self.only_heatmap:
            return batch_data_samples

        batch_data_samples_1 = self.first_decoder_layer.predict(feats[0], heatmaps[0].detach().clone(), batch_data_samples)

        batch_data_samples = batch_data_samples_1
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """

        # event_feats, color_feats = self.extract_feat(batch_inputs)
        
        # feats = self.flow_select(event_feats, color_feats)

        # heatmaps = self.heatmap_head.forward(feats)

        # results_1 = self.first_decoder_layer.forward(feats[0], heatmaps[0].detach().clone(), batch_data_samples)

        # return (heatmaps[0], results_1)
        return None
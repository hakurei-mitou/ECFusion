from typing import Dict, Tuple, Union
from mmdet.models.dense_heads.ec_centernet_head import EC_CenterNetHead
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils.misc import multi_apply, samplelist_boxtype2tensor
from mmdet.structures.det_data_sample import SampleList

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import OptSampleList
from mmdet.models.detectors.base import BaseDetector, ForwardResults
from mmdet.utils.typing_utils import InstanceList

@MODELS.register_module()
class ECFusionB(BaseDetector):
    def __init__(self,
                 event_backbone: ConfigType = None,
                 color_backbone: ConfigType = None,
                 event_neck: ConfigType = None,
                 color_neck: ConfigType = None,
                 event_head: ConfigType = None,
                 color_head: ConfigType = None,
                 multi_level_fusion: ConfigType = None,
                 center_head: ConfigType = None,
                 backbone: ConfigType = None,
                 neck: ConfigType = None,
                 flow: ConfigType = dict(only_event=False, only_color=True, fusion=False),
                 fusion_module: ConfigType = None,
                 freeze: Dict = None,
                 data_preprocessor: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.only_event = flow['only_event']
        self.only_color = flow['only_color']
        self.fusion = flow['fusion']
        assert self.only_event + self.only_color + self.fusion == 1, 'only one in this flow type can affect'
        
        if test_cfg is not None:
            if self.only_color:
                color_head.update(train_cfg=train_cfg)
                color_head.update(test_cfg=test_cfg)
            elif self.only_event:
                event_head.update(train_cfg=train_cfg)
                event_head.update(test_cfg=test_cfg)
            elif self.fusion:
                center_head.update(train_cfg=train_cfg)
                center_head.update(test_cfg=test_cfg)

        self.fusion_module = fusion_module

        self.one_neck = flow['one_neck']
        if self.fusion and self.one_neck and not fusion_module['LF']:
            assert neck is not None
            assert event_neck is None
            assert color_neck is None

        if self.only_event:
            self.event_backbone = MODELS.build(event_backbone)
            self.event_neck = MODELS.build(event_neck)
            self.event_head = MODELS.build(event_head)
        elif self.only_color:
            self.color_backbone = MODELS.build(color_backbone)
            self.color_neck = MODELS.build(color_neck)
            self.color_head = MODELS.build(color_head)
        elif self.fusion:

            # backbone
            if fusion_module['EF']:
                self.backbone = MODELS.build(backbone)
            else:
                self.event_backbone = MODELS.build(event_backbone)
                self.color_backbone = MODELS.build(color_backbone)

            # neck
            if self.one_neck and not fusion_module['LF']:
                self.neck = MODELS.build(neck)
            else:
                self.event_neck = MODELS.build(event_neck)
                self.color_neck = MODELS.build(color_neck)
            
            # fusion module
            if fusion_module['GSTFM'] or fusion_module['BDC'] or fusion_module['EICA'] or fusion_module['AABFM']:
                self.multi_level_fusion = MODELS.build(multi_level_fusion)
            elif fusion_module['CC']:
                assert multi_level_fusion is None
                assert sum(neck['in_channels']) == sum([512*2, 1024*2, 2048*2])
            elif fusion_module['EF']:
                assert event_backbone is None
                assert color_backbone is None
                assert multi_level_fusion is None
            elif fusion_module['LF']:
                assert neck is None
                assert multi_level_fusion is None
                assert event_neck['out_channels'] == 128
                assert color_neck['out_channels'] == 128

            # head
            self.center_head = MODELS.build(center_head)

        if freeze is not None:
            self.freeze_event = freeze['freeze_event']
            self.event_checkpoint = freeze['event_checkpoint']
            self.freeze_color = freeze['freeze_color']
            self.color_checkpoint = freeze['color_checkpoint']

            self.freeze_weights()

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

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
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def freeze_module(self, module, pt):
        module.load_state_dict(pt['state_dict'], strict=False)
        for para in module.parameters():
            para.required_grad = False

    def freeze_weights(self):
        if self.freeze_event:
            pt = torch.load(self.event_checkpoint)
            self.freeze_module(self.event_backbone, pt)
            self.freeze_module(self.event_neck, pt)
            self.freeze_module(self.event_head, pt)
        if self.freeze_color:
            pt = torch.load(self.color_checkpoint)
            self.freeze_module(self.color_backbone, pt)
            self.freeze_module(self.color_neck, pt)
            self.freeze_module(self.color_head, pt)

    def extract_color_feat(self, c_batch_inputs: Tensor):
        x = self.color_backbone(c_batch_inputs)
        if self.one_neck is None:
            x = self.color_neck(x)
        return x

    def extract_event_feat(self, batch_inputs: Tensor):
        x = self.event_backbone(batch_inputs)
        if self.one_neck is None:
            x = self.event_neck(x)
        return x

    def extract_feat(self, batch_inputs: Tensor, c_batch_inputs: Tensor):
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
            # NOTE: must be batch_inputs not c_batch_inputs here
            color_feats = self.extract_color_feat(batch_inputs)
        elif self.fusion:
            event_feats = self.extract_event_feat(batch_inputs)
            color_feats = self.extract_color_feat(c_batch_inputs)
        
        return event_feats, color_feats

    def loss(self,
            batch_inputs: Tensor,
            c_batch_inputs: Tensor,
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
        
        feats = self._forward(batch_inputs, c_batch_inputs, batch_data_samples)
        
        if self.only_event:
            sign = '_e'
            losses = self.event_head.loss(feats, batch_data_samples)        
        elif self.only_color:
            sign = '_c'
            losses = self.color_head.loss(feats, batch_data_samples)        
        elif self.fusion:
            sign = ''
            losses = self.center_head.loss(feats, batch_data_samples)

        # CenterNetHead
        # final_loss = ({
        #     f'loss_center_heatmap{sign}' : losses['loss_center_heatmap'],
        #     f'loss_wh{sign}' : losses['loss_wh'],
        #     f'loss_offset{sign}' : losses['loss_offset']
        # })

        # CenterNetUpdateHead
        # final_loss = ({
        #     f'loss_cls{sign}' : losses['loss_cls'],
        #     f'loss_bbox{sign}' : losses['loss_bbox']
        # })
        # return final_loss
        return losses

    def predict(self,
                batch_inputs: Tensor,
                c_batch_inputs: Tensor,
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

        feats = self._forward(batch_inputs, c_batch_inputs, batch_data_samples)

        if self.only_event:
            results_list = self.event_head.predict(feats, batch_data_samples)
        elif self.only_color:
            results_list = self.color_head.predict(feats, batch_data_samples)
        elif self.fusion:
            results_list = self.center_head.predict(feats, batch_data_samples, rescale=True)
        
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 c_batch_inputs: Tensor,
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


            # NOTE feats: [BS, C, H, W]
            

        if self.fusion_module['EF']:
            feats = self.backbone(batch_inputs)
        else:
            # NOTE: fusion and only_event and only_color
            e_feats, c_feats = self.extract_feat(batch_inputs, c_batch_inputs)
        
        if self.only_event:
            feats = self.event_neck(e_feats)

        elif self.only_color:
            feats = self.color_neck(c_feats)

        elif self.fusion:

            if self.fusion_module['GSTFM'] or self.fusion_module['BDC'] or self.fusion_module['EICA'] \
                or self.fusion_module['AABFM']:
                feats = self.multi_level_fusion.forward(e_feats, c_feats)
            elif self.fusion_module['CC']:
                feats = [torch.cat((e, c), dim=1) for e, c in zip(e_feats, c_feats)]
            elif self.fusion_module['LF']:
                e_feats = self.event_neck(e_feats)
                c_feats = self.color_neck(c_feats)
                feats = [torch.cat((e, c), dim=1) for e, c in zip(e_feats, c_feats)]

            if self.one_neck and not self.fusion_module['LF']:
                feats = self.neck.forward(feats)            

        return feats
    
    def forward(self,
                inputs: torch.Tensor = None,
                c_inputs: torch.Tensor = None,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        
        assert inputs is not None or c_inputs is not None

        if mode == 'loss':
            return self.loss(inputs, c_inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, c_inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, c_inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
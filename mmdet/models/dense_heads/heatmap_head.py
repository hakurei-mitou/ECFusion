import copy
from mmdet.models.utils.misc import unpack_gt_instances
from mmdet.structures.det_data_sample import SampleList
import torch.nn as nn
import numpy as np

from mmdet.registry import MODELS
from mmcv.cnn import ConvModule
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..utils import (gaussian_radius, gen_gaussian_target, get_local_maximum,
                     get_topk_from_heatmap, multi_apply,
                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class HeatmapHead(BaseDenseHead):
    """
    Changed from CenterNetHead:

    Args:
        in_channels (int): Number of channel in the input feature map.
        feat_channels (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (:obj:`ConfigDict` or dict): Config of center
            heatmap loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Useless in CenterNet, but we keep this variable for
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization
            config dict.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_classes: int,
                 eps: float = 1e-4,
                 loss_center_heatmap: ConfigType = dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.eps = eps
        self.heatmap_head = self._build_head(in_channels, feat_channels,
                                             num_classes)

        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channels: int, feat_channels: int,
                    out_channels: int) -> nn.Sequential:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1))
        return layer

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
        """
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        return (center_heatmap_pred, )

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> tuple:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)

        losses = self.loss_by_feat(*loss_inputs)
        
        return losses, *outs

    def loss_by_feat(
            self,
            center_heatmap_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> Tensor:
        """Compute losses of the head.

        Args:
            center_heatmap_pred (list[Tensor]): center predict heatmap, each level
                with shape (B, num_classes, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            loss_center_heatmap (Tensor): loss of center heatmap.
        """

        # center_hearmap_preds is a list of levels.

        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']

        losses = []
        for center_heatmap_pred in center_heatmap_preds:

            target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                        center_heatmap_pred.shape,
                                                        img_shape)

            center_heatmap_target = target_result['center_heatmap_target']

            center_heatmap_pred = torch.clamp(center_heatmap_pred, min=self.eps, max=1 - self.eps)

            loss_heatmap = self.loss_center_heatmap(
                center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

            losses.append(loss_heatmap)

        return losses

    def get_targets(self, gt_bboxes: List[Tensor], gt_labels: List[Tensor],
                    feat_shape: tuple, img_shape: tuple) -> Tuple[dict, int]:
        """Compute classification targets in multiple images.

                
            note: heatmap_head do not predict boxes,
                but gt_boxes are needed for computing gaussian target.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (tuple): feature map shape with value [B, _, H, W]
            img_shape (tuple): image shape.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target)
        return target_result, avg_factor

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions, *outs

    def predict_by_feat(self,
                        center_heatmap_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        """ predict heatmap_scores and heatmap_labels for all level

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmap for
                all level each with shape (B, num_classes, H, W).

        note: 
            batch_img_metas and rescale 
                just for filling args of 'predict' function in base_dense_head,
                it do nothing in function.

        Returns:

            - result_list:
                list[ list[ dict[ 'leveln': tuple(heatmap_scores, heatmap_labels) ] ] ]
        """
        result_list = []
        for img_id in range(len(batch_img_metas)):
            level_list = []
            for i, heatmap_level in enumerate(center_heatmap_preds):
                level_list.append(
                    self._predict_by_feat_single(
                        i,
                        heatmap_level[img_id:img_id + 1, ...]))
            result_list.append(level_list)
        return result_list

    def _predict_by_feat_single(self,
                                level_n : int,
                                center_heatmap_pred: Tensor) -> tuple:
        """predict heatmap_socres and heatmap_labels for a level of a image

        Args:
            center_heatmap_pred (Tensor): Center heatmap with
                shape (1, num_classes, H, W).

        Returns:
                - (scores, labels)
        """

        center_heatmap_pred.squeeze_()
        scores, labels = center_heatmap_pred.max(dim=0)

        return (scores, labels)
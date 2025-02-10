_base_ = '../_base_/default_runtime.py'

image_size = (480, 640)

model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        # bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet',
        in_channels=5,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    # bbox_head=dict(
    #     type='CenterNetUpdateHead',
    #     num_classes=3,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=128,
    #     strides=[8, 16, 32, 64, 128],
    #     loss_cls=dict(
    #         type='GaussianFocalLoss',
    #         pos_weight=0.25,
    #         neg_weight=0.75,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    # ),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=3,
        in_channels=256,
        feat_channels=128,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
    ),

    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))



flow=dict(only_event=True, only_color=False, fusion=False)
backend_args = None

train_pipeline = [
    # dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEventAndImage', flow=flow),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='RandomResize',
    #     scale=image_size,
    #     ratio_range=(0.1, 2.0),
    #     keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    # dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEventAndImage', flow=flow),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


############################################ dataset

dataset_type = 'CocoDataset'
classes = ('person', 'large_vehicle', 'car')

num_workers = 2
batch_size = 32

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='data/DSEC/train/',
        ann_file='DSEC_detection_labels/mini_train_ann.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='data/DSEC/train/',
        ann_file='DSEC_detection_labels/val_ann.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='data/DSEC/train/',
        ann_file='DSEC_detection_labels/test_ann.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/DSEC/train/DSEC_detection_labels/val_ann.json',
    metric='bbox',
    # proposal_nums=(num_queries, num_queries, num_queries),
    format_only=False,
    backend_args=backend_args)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/DSEC/train/DSEC_detection_labels/test_ann.json',
    metric='bbox',
    # proposal_nums=(num_queries, num_queries, num_queries),
    format_only=False,
    backend_args=backend_args)


############################################ training

max_epochs = 40

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01 * 4, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0.))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00025,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
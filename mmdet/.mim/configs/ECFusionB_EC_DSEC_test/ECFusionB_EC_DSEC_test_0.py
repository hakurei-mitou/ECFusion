_base_ = [
    '../_base_/default_runtime.py'
]

num_classes = 3
embed_dims = 256
num_bins = 5
level_number = 1
flow=dict(
    # only event
    # only_event=True, only_color=False, fusion=False,

    # only color
    # only_event=False, only_color=True, fusion=False,

    # fusion
    only_event=False, only_color=False, fusion=True,

    other_detector=False,
)
freeze=dict(
    freeze_event=False, event_checkpoint='',
    freeze_color=False, color_checkpoint='',
)

# num_queries = 20

same_structure_head = dict(
    type='EC_CenterNetHead',
    loss_center_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    loss_wh=dict(type='L1Loss', loss_weight=0.1),
    loss_offset=dict(type='L1Loss', loss_weight=1.0),
    in_channels=embed_dims,
    feat_channels=128,
    num_classes=num_classes,
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100)
)

model = dict(
    type='ECFusionB',
    flow=flow,
    freeze=freeze,
    data_preprocessor=dict(
        type='EC_DetDataPreprocessor',
        # event no been normalized
        mean=[123.675, 116.28, 103.53],  # 3 channels have 3 values
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=1,
        flow=flow,
    ),
    event_backbone=dict(
        type='ResNet',
        in_channels=num_bins,
        frozen_stages=-1,
        deep_stem=False,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    event_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level= 1,
        add_extra_convs='on_output',
        num_outs=3,
        upsample_cfg=dict(mode='bilinear'),
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    event_head=same_structure_head,
    color_backbone=dict(
        type='ResNet',
        in_channels=3,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',),
    color_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3,
        upsample_cfg=dict(mode='bilinear'),
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    color_head=same_structure_head,
    fusion_layer=dict(
        type='FusionLayer',
        embed_dims=embed_dims,
        num_heads=8,
        ffn_channel=1024,
        positional_encoding=dict(
            type='EC_SinePositionalEncoding',
            num_feats=embed_dims//2,
            normalize=True,
        ),
    ),
    center_head=same_structure_head,
)


############################################ data

backend_args = None
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    # LoadImageFromFile
    dict(type='LoadEventAndImage', flow=flow),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomFlip', prob=0.5),  # 两条路需要保持增强方式的一致，随机的变换只能用在单路上
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),

    # c_img_path will be automatically added, when flow['fusion'] = True
    dict(type='EC_PackDetInputs', flow=flow, meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'])
]
test_pipeline = [
    dict(type='LoadEventAndImage', flow=flow),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='EC_PackDetInputs', flow=flow, meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'])
]

############################################ dataset

dataset_type = 'CocoDataset'
classes = ('person', 'large_vehicle', 'car')

num_workers = 2
batch_size = 2
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=8)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='data/DSEC/train/',
        ann_file='DSEC_detection_labels/train_ann.json',
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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    # paramwise_cfg=dict(
    #     custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
    )

# learning policy
max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

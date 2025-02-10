_base_ = [
    '../_base_/default_runtime.py'
]

dataset_name = 'DSEC-Soft'
num_classes = 3
num_bins = 5
flow=dict(
    # only event
    # only_event=True, only_color=False, fusion=False,

    # only color
    # only_event=False, only_color=True, fusion=False,

    # fusion
    only_event=False, only_color=False, fusion=True,

    # fusion before neck
    one_neck = True,

    other_detector=False,
)

fusion_module=dict(
    # some config parameters of the model structure need to be changed by hand.
    GSTFM=False,
    CC=True,
    EF=False,
    LF=False,
    BDC=False,
    EICA=False,
    AABFM=False,
)
# there is only one True
assert fusion_module['GSTFM'] + fusion_module['CC'] + fusion_module['EF'] + fusion_module['LF'] \
    + fusion_module['BDC'] + fusion_module['EICA'] + fusion_module['AABFM'] == 1

# neck
start_level = 0
num_outs = 5

model = dict(
    type='ECFusionB',
    flow=flow,
    fusion_module=fusion_module,
    data_preprocessor=dict(
        type='EC_DetDataPreprocessor',
        flow=flow,
        # event not been normalized
        mean=[123.675, 116.28, 103.53],  # 3 channels have 3 values
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=1,
    ),
    event_backbone=dict(
        type='ResNet',
        in_channels=num_bins,
        frozen_stages=-1,
        deep_stem=False,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    color_backbone=dict(
        type='ResNet',
        in_channels=3,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',),
    # multi_level_fusion=dict(
    #     type='MultiLevelFusion',
    #     group_factors_a=[9, 4, 0],
    #     group_factors_b=[9, 4, 0],
    #     embed_dims=[512, 1024, 2048],
    #     ffn_channel=[1024, 2048, 4096],
    #     fusion_layer=dict(
    #         type='FusionLayer',
    #         use_shuffle=False,   # TFM 不受影响
    #         num_heads=8,
    #         dataset_name=dataset_name
    #     ),
    # ),
    neck=dict(
        type='FPN',
        # in_channels=[512, 1024, 2048],
        in_channels=[512*2, 1024*2, 2048*2],   # CC
        out_channels=256,
        start_level=start_level,
        add_extra_convs='on_output',
        num_outs=num_outs,
        upsample_cfg=dict(mode='bilinear'),
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True
    ),
    center_head=dict(
        type='CenterNetUpdateHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=128,
        strides=[8, 16, 32, 64, 128],   # P3 begin
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)


############################################ data

backend_args = None
event_source = 0
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

transform_name = 'LT'

train_pipeline = [
    # LoadImageFromFile
    dict(type='LoadEventAndImage', flow=flow, event_source=event_source, transform_name=transform_name, fusion_module=fusion_module,\
         dataset_name=dataset_name),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomFlip', prob=0.5),  # 两条路需要保持增强方式的一致，随机的变换只能用在单路上
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),

    # c_img_path will be automatically added, when flow['fusion'] = True
    dict(type='EC_PackDetInputs', flow=flow, meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'])
]
test_pipeline = [
    dict(type='LoadEventAndImage', flow=flow, event_source=event_source, transform_name=transform_name, fusion_module=fusion_module,\
         dataset_name=dataset_name),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='EC_PackDetInputs', flow=flow, meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'])
]

############################################ dataset

dataset_type = 'CocoDataset'
classes = ('car', 'large_vehicle', 'person')

num_workers = 2


max_epochs = 20
batch_size = 2
accumulative_counts = 8   # to count iterations

data_root='data/DSEC/'

# ann_file_train = 'train/soft_ann/soft_ann_mini_train_LN.json'
ann_file_train = 'train/soft_ann/soft_ann_train_LT.json'

ann_file_test = 'train/soft_ann/soft_ann_test_LT.json'


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
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
        data_root=data_root,
        ann_file=ann_file_test,
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
        data_root=data_root,
        ann_file=ann_file_test,
        metainfo=dict(classes=classes),
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + ann_file_test,
    metric='bbox',
    
    classwise=True,

    format_only=False,
    backend_args=backend_args)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + ann_file_test,
    metric='bbox',
    
    classwise=True,

    format_only=False,
    backend_args=backend_args)


############################################ training

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=accumulative_counts,
    # paramwise_cfg=dict(
    #     custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
    )

# learning policy
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

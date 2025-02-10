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
    only_event=True, only_color=False, fusion=False,

    # fusion before neck
    one_neck = True,

    other_detector=True,
)

fusion_module=dict(
    # some config parameters of the model structure need to be changed by hand.
    GSTFM=False,
    CC=False,
    EF=False,
    LF=False,
    BDC=False,
    EICA=False,
    AABFM=False,
)

# neck
start_level = 0
num_outs = 5
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)

model = dict(
    type='ECFusionB',
    flow=flow,
    backbone=dict(),
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
        in_channels=5,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',),


    event_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        # in_channels=[512*2, 1024*2, 2048*2],   # CC
        out_channels=256,
        start_level=start_level,
        add_extra_convs='on_output',
        num_outs=num_outs,
        upsample_cfg=dict(mode='nearest'),
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True
    ),
    # event_neck=dict(
    #     type='BiFPN',
    #     num_stages=3,
    #     # in_channels=[40, 112, 320],
    #     in_channels=[512, 1024, 2048],
    #     out_channels=256,
    #     start_level=0,
    #     norm_cfg=norm_cfg),




    # event_head=dict(
    #     type='CenterNetUpdateHead',
    #     num_classes=num_classes,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=128,
    #     strides=[8, 16, 32, 64, 128],   # P3 begin
    #     loss_cls=dict(
    #         type='GaussianFocalLoss',
    #         pos_weight=0.25,
    #         neg_weight=0.75,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    # ),
    event_head=dict(
        type='EfficientDetSepBNHead',
        num_classes=num_classes,
        num_ins=5,
        in_channels=256,
        feat_channels=256,
        stacked_convs=3,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='HuberLoss', beta=0.1, loss_weight=50)), 





    # train_cfg=None,
    # test_cfg=dict(
    #     nms_pre=1000,
    #     min_bbox_size=0,
    #     score_thr=0.05,
    #     nms=dict(type='nms', iou_threshold=0.6),
    #     max_per_img=100)
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='soft_nms',
            iou_threshold=0.3,
            sigma=0.5,
            min_score=1e-3,
            method='gaussian'),
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
batch_size = 16
accumulative_counts = 1   # to count iterations

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

_base_ = [
    '../_base_/default_runtime.py'
]

# todo: should be coco class
num_classes = 80
embed_dims = 2048

model = dict(
    type='ECFusion',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    heatmap_head=dict(
        type='HeatmapHead',
        loss_center_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        num_classes=num_classes,
        in_channels=embed_dims,
        feat_channels=256),
    decoder_layer=dict(  # DetrTransformerDecoder
        type='DecoderLayer',
        num_queries=100,
        num_classes=num_classes,
        embed_dims=embed_dims,
        nms_kernel_size=3,
        nms_padding=1,
        decoder=dict(
            num_layers=1,
            return_intermediate=False,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=embed_dims,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=embed_dims,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=embed_dims,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)))),
        bbox_head=dict(
            type='DETRHead',
            num_classes=num_classes,
            embed_dims=embed_dims,
            loss_cls=dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=1.),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        test_cfg=dict(max_per_img=100)))



############################################ data

backend_args = None
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),  # 两条路需要保持增强方式的一致，随机的变换只能用在单路上
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomChoice',
    #     transforms=[[
    #         dict(
    #             type='RandomChoiceResize',
    #             scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                     (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                     (736, 1333), (768, 1333), (800, 1333)],
    #             keep_ratio=True)
    #     ],
    #     [
    #         dict(
    #             type='RandomChoiceResize',
    #             scales=[(400, 1333), (500, 1333), (600, 1333)],
    #             keep_ratio=True),
    #         dict(
    #             type='RandomCrop',
    #             crop_type='absolute_range',
    #             crop_size=(384, 600),
    #             allow_negative_crop=True),
    #         dict(
    #             type='RandomChoiceResize',
    #             scales=[(480, 1333), (512, 1333), (544, 1333),
    #                     (576, 1333), (608, 1333), (640, 1333),
    #                     (672, 1333), (704, 1333), (736, 1333),
    #                     (768, 1333), (800, 1333)],
    #             keep_ratio=True)
    #     ]]),
    dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]

############################################ dataset

dataset_type = 'CocoDataset'
data_root = 'data/coco2017/'

num_workers = 2
batch_size = 12

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
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
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


############################################ mini dataset

# dataset_type = 'CocoDataset'
# data_root = 'data/mini_coco/'

# num_workers = 1
# batch_size = 2


# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# train_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=num_workers,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/mini_instances_train2017.json',
#         data_prefix=dict(img='train/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))
# val_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=num_workers,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/mini_instances_val2017.json',
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))
# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/mini_instances_val2017.json',
#     metric='bbox',
#     format_only=False,
#     backend_args=backend_args)
# test_evaluator = val_evaluator


############################################ training

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

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

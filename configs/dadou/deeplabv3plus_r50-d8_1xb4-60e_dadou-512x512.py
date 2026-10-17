"""DeepLabV3+ ResNet-50 teacher for three-class soybean lesion masks."""

_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(imports=['projects.dadou_teacher'], allow_failed_imports=False)

crop_size = (512, 512)
data_root = 'data/dadou_mmseg'
dataset_type = 'DadouLesionDataset'
classes = ('background', 'healthy_leaf', 'lesion')
palette = [[0, 0, 0], [0, 180, 0], [220, 30, 30]]
metainfo = dict(classes=classes, palette=palette)

norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    # Scale jitter can produce crops smaller than 512; those black padded
    # pixels are genuine image background for this task.
    seg_pad_val=0,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[0.4914, 0.6319, 1.8768],
                loss_weight=1.0,
                loss_name='loss_ce'),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                loss_weight=1.0,
                loss_name='loss_dice'),
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[0.4914, 0.6319, 1.8768],
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Full DeepLabV3+ pretraining on Pascal VOC; the incompatible 21-class output
# layers are intentionally reinitialized for this task's three classes.
load_from = (
    'checkpoints/'
    'deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.75, 1.50),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.25, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=25, pad_val=0, seg_pad_val=0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=20,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=8),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
            pipeline=train_pipeline,
            metainfo=metainfo)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=test_pipeline,
        metainfo=metainfo))

# Defined for the eventual final evaluation, but not used by tools/train.py.
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline=test_pipeline,
        metainfo=metainfo))

val_evaluator = dict(
    type='DadouIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=3, by_epoch=True),
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=3,
        end=60,
        by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop', fp16=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=5,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [
    dict(type='EpochMetricsCsvHook', filename='epoch_metrics.csv'),
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        min_delta=0.10,
        patience=12),
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    alpha=0.65)

log_processor = dict(by_epoch=True)
randomness = dict(seed=20260801, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)
work_dir = 'work_dirs/dadou_deeplabv3plus_r50_teacher'

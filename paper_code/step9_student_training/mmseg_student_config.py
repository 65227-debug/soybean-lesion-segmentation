"""MMSegmentation inference config for the step-nine student model."""

default_scope = 'mmseg'

classes = ('background', 'healthy_leaf', 'lesion')
palette = [[0, 0, 0], [0, 180, 0], [220, 30, 30]]
metainfo = dict(classes=classes, palette=palette)

norm_cfg = dict(type='BN', eps=0.001, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MobileNetV3',
        arch='large',
        out_indices=(3, 12, 16),
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://contrib/mobilenet_v3_large')),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=960,
        in_index=2,
        channels=256,
        dilations=(1, 12, 24, 36),
        c1_in_channels=24,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs'),
]


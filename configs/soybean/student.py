"""Inference configuration for the soybean student model."""

default_scope = 'mmseg'

classes = ('background', 'healthy_leaf', 'lesion')

palette = [[0, 0, 0], [0, 180, 0], [220, 30, 30]]

metainfo = {'classes': ('background', 'healthy_leaf', 'lesion'),
 'palette': [[0, 0, 0], [0, 180, 0], [220, 30, 30]]}

data_preprocessor = {'type': 'SegDataPreProcessor',
 'mean': [123.675, 116.28, 103.53],
 'std': [58.395, 57.12, 57.375],
 'bgr_to_rgb': True,
 'size': (512, 512),
 'pad_val': 0,
 'seg_pad_val': 255}

model = {'type': 'EncoderDecoder',
 'data_preprocessor': {'type': 'SegDataPreProcessor',
                       'mean': [123.675, 116.28, 103.53],
                       'std': [58.395, 57.12, 57.375],
                       'bgr_to_rgb': True,
                       'size': (512, 512),
                       'pad_val': 0,
                       'seg_pad_val': 255},
 'backbone': {'type': 'MobileNetV3',
              'arch': 'large',
              'out_indices': (3, 12, 16),
              'norm_cfg': {'type': 'BN', 'eps': 0.001, 'requires_grad': True}},
 'decode_head': {'type': 'DepthwiseSeparableASPPHead',
                 'in_channels': 960,
                 'in_index': 2,
                 'channels': 256,
                 'dilations': (1, 12, 24, 36),
                 'c1_in_channels': 24,
                 'c1_channels': 48,
                 'dropout_ratio': 0.1,
                 'num_classes': 3,
                 'norm_cfg': {'type': 'BN', 'eps': 0.001, 'requires_grad': True},
                 'align_corners': False},
 'test_cfg': {'mode': 'whole'}}

test_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': False},
 {'type': 'PackSegInputs'}]

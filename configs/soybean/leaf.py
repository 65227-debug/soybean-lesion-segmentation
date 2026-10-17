"""Inference configuration for the soybean leaf model."""

default_scope = 'mmseg'

classes = ('background', 'leaf')

palette = [[0, 0, 0], [0, 180, 0]]

metainfo = {'classes': ('background', 'leaf'), 'palette': [[0, 0, 0], [0, 180, 0]]}

data_preprocessor = {'type': 'SegDataPreProcessor',
 'mean': [123.675, 116.28, 103.53],
 'std': [58.395, 57.12, 57.375],
 'bgr_to_rgb': True,
 'pad_val': 0,
 'seg_pad_val': 0,
 'size': (512, 512)}

model = {'type': 'EncoderDecoder',
 'data_preprocessor': {'type': 'SegDataPreProcessor',
                       'mean': [123.675, 116.28, 103.53],
                       'std': [58.395, 57.12, 57.375],
                       'bgr_to_rgb': True,
                       'pad_val': 0,
                       'seg_pad_val': 0,
                       'size': (512, 512)},
 'backbone': {'type': 'UNet',
              'in_channels': 3,
              'base_channels': 64,
              'num_stages': 5,
              'strides': (1, 1, 1, 1, 1),
              'enc_num_convs': (2, 2, 2, 2, 2),
              'dec_num_convs': (2, 2, 2, 2),
              'downsamples': (True, True, True, True),
              'enc_dilations': (1, 1, 1, 1, 1),
              'dec_dilations': (1, 1, 1, 1),
              'with_cp': False,
              'conv_cfg': None,
              'norm_cfg': {'type': 'BN', 'requires_grad': True},
              'act_cfg': {'type': 'ReLU'},
              'upsample_cfg': {'type': 'InterpConv'},
              'norm_eval': False},
 'decode_head': {'type': 'FCNHead',
                 'in_channels': 64,
                 'in_index': 4,
                 'channels': 64,
                 'num_convs': 1,
                 'concat_input': False,
                 'dropout_ratio': 0.1,
                 'num_classes': 2,
                 'norm_cfg': {'type': 'BN', 'requires_grad': True},
                 'align_corners': False},
 'auxiliary_head': {'type': 'FCNHead',
                    'in_channels': 128,
                    'in_index': 3,
                    'channels': 64,
                    'num_convs': 1,
                    'concat_input': False,
                    'dropout_ratio': 0.1,
                    'num_classes': 2,
                    'norm_cfg': {'type': 'BN', 'requires_grad': True},
                    'align_corners': False},
 'test_cfg': {'mode': 'whole'}}

test_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': False},
 {'type': 'PackSegInputs'}]

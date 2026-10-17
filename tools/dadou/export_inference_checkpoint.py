#!/usr/bin/env python
"""Strip optimizer/runtime metadata from a locally trained MMEngine checkpoint."""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Trusted checkpoint produced by this run')
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--task', choices=('lesion', 'leaf'), default='lesion',
        help='Dataset metadata to retain in the portable checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = (Path(args.output) if args.output else checkpoint_path.with_name(
        f'{checkpoint_path.stem}_inference.pth'))

    # This source is explicitly a trusted, locally produced training artifact.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dataset_meta = ({
        'classes': ('background', 'healthy_leaf', 'lesion'),
        'palette': [[0, 0, 0], [0, 180, 0], [220, 30, 30]],
    } if args.task == 'lesion' else {
        'classes': ('background', 'leaf'),
        'palette': [[0, 0, 0], [0, 180, 0]],
    })
    inference_checkpoint = {
        'meta': {
            'dataset_meta': dataset_meta
        },
        'state_dict': checkpoint['state_dict'],
    }
    torch.save(inference_checkpoint, output_path)
    print(output_path.resolve())


if __name__ == '__main__':
    main()

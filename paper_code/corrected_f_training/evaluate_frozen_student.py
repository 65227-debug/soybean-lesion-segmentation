#!/usr/bin/env python
"""Evaluate one validation-frozen student on the test split exactly once."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


STEP9_DIR = Path(__file__).resolve().parents[1] / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/dadou_student')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = common.StudentSegmentationDataset(
        Path(args.data_root) / 'supervised_test_manifest.csv', training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
        worker_init_fn=common.seed_worker)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = common.build_student(args.model_config, device)
    model.load_state_dict(checkpoint['state_dict'])
    class_weights = torch.tensor(
        [0.4914, 0.6319, 1.8768], dtype=torch.float32, device=device)
    metrics, per_image = common.evaluate(
        model, loader, device, class_weights,
        save_dir=output_dir / 'predictions')
    common.write_csv(output_dir / 'test_per_image.csv', per_image)
    summary = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'test_images': len(dataset),
        'checkpoint_selected_on': 'validation split only',
        'test_evaluation_count_for_this_checkpoint': 1,
        'metrics': metrics,
    }
    (output_dir / 'test_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

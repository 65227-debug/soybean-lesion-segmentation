#!/usr/bin/env python
"""Select a single-model A/corrected-F weight soup on validation only."""

import argparse
import csv
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
    parser.add_argument('--checkpoint-a', required=True)
    parser.add_argument('--checkpoint-f', required=True)
    parser.add_argument('--supervised-root', default='data/dadou_student')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--alphas', default='0,0.025,0.05,0.075,0.10,0.15,0.20,0.30,0.40')
    return parser.parse_args()


def blend_states(state_a, state_f, alpha):
    blended = {}
    for key, value_a in state_a.items():
        value_f = state_f[key]
        if torch.is_floating_point(value_a):
            blended[key] = value_a * (1.0 - alpha) + value_f * alpha
        else:
            blended[key] = value_a.clone()
    return blended


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_val_manifest.csv',
        training=False)
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)
    model = common.build_student(args.model_config, device)
    checkpoint_a = torch.load(
        args.checkpoint_a, map_location='cpu', weights_only=True)
    checkpoint_f = torch.load(
        args.checkpoint_f, map_location='cpu', weights_only=True)
    state_a = checkpoint_a['state_dict']
    state_f = checkpoint_f['state_dict']
    class_weights = torch.tensor(
        [0.4914, 0.6319, 1.8768], dtype=torch.float32, device=device)
    rows = []
    best = None
    best_state = None
    for alpha in [float(value) for value in args.alphas.split(',')]:
        state = blend_states(state_a, state_f, alpha)
        model.load_state_dict(state)
        metrics, _ = common.evaluate(model, loader, device, class_weights)
        row = {
            'alpha_corrected_f': alpha,
            **{key: value for key, value in metrics.items()
               if key != 'confusion_matrix'},
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best is None or metrics['mIoU'] > best['metrics']['mIoU']:
            best = {'alpha_corrected_f': alpha, 'metrics': metrics}
            best_state = {key: value.cpu() for key, value in state.items()}
    with (output_dir / 'validation_grid.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({
        'state_dict': best_state,
        'epoch': -1,
        'metrics': best['metrics'],
        'architecture': 'DeepLabV3+ MobileNetV3-Large corrected F weight soup',
        'alpha_corrected_f': best['alpha_corrected_f'],
        'source_checkpoints': {
            'A': str(Path(args.checkpoint_a).resolve()),
            'corrected_F': str(Path(args.checkpoint_f).resolve()),
        },
    }, output_dir / 'best_validation_weight_soup.pth')
    summary = {
        **best,
        'selection_split': 'validation only',
        'test_split_used': False,
        'single_model_after_weight_interpolation': True,
    }
    (output_dir / 'selection_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


#!/usr/bin/env python
"""Select an A/corrected-F soft-logit ensemble on validation only."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
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
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--alphas-f', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
    parser.add_argument('--healthy-biases', default='-0.6,-0.3,0,0.3')
    parser.add_argument('--lesion-biases', default='-1.2,-0.8,-0.4,0,0.4')
    return parser.parse_args()


def parse_grid(text):
    return [float(value) for value in text.split(',')]


@torch.inference_mode()
def collect_logits(checkpoint_path, model_config, loader, device):
    model = common.build_student(model_config, device)
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logits_parts = []
    target_parts = []
    sample_ids = []
    for batch in loader:
        targets = batch['mask'].to(device, non_blocking=True)
        logits = common.forward_logits(
            model, batch['image'].to(device, non_blocking=True),
            targets.shape[-2:])
        logits_parts.append(logits.to(torch.float16))
        target_parts.append(targets.to(torch.uint8))
        sample_ids.extend(batch['sample_id'])
    del model
    return torch.cat(logits_parts), torch.cat(target_parts), sample_ids


@torch.inference_mode()
def confusion_from_scores(scores, targets, healthy_bias, lesion_bias):
    class_bias = torch.tensor(
        [0.0, healthy_bias, lesion_bias], dtype=scores.dtype,
        device=scores.device).view(1, 3, 1, 1)
    prediction = (scores + class_bias).argmax(dim=1)
    valid = targets < 3
    encoded = targets[valid].long() * 3 + prediction[valid]
    return torch.bincount(encoded, minlength=9).reshape(3, 3).cpu().numpy()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_val_manifest.csv',
        training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
        worker_init_fn=common.seed_worker)
    logits_a, targets, sample_ids = collect_logits(
        args.checkpoint_a, args.model_config, loader, device)
    logits_f, targets_f, sample_ids_f = collect_logits(
        args.checkpoint_f, args.model_config, loader, device)
    if sample_ids != sample_ids_f or not torch.equal(targets, targets_f):
        raise RuntimeError('Validation ordering changed between checkpoints.')

    rows = []
    best = None
    for alpha_f in parse_grid(args.alphas_f):
        scores = logits_a * (1.0 - alpha_f) + logits_f * alpha_f
        for healthy_bias in parse_grid(args.healthy_biases):
            for lesion_bias in parse_grid(args.lesion_biases):
                confusion = confusion_from_scores(
                    scores, targets, healthy_bias, lesion_bias)
                metrics = common.metrics_from_confusion(confusion)
                row = {
                    'alpha_corrected_f': alpha_f,
                    'healthy_logit_bias': healthy_bias,
                    'lesion_logit_bias': lesion_bias,
                    **metrics,
                }
                rows.append(row)
                if best is None or metrics['mIoU'] > best['metrics']['mIoU']:
                    best = {
                        'alpha_corrected_f': alpha_f,
                        'healthy_logit_bias': healthy_bias,
                        'lesion_logit_bias': lesion_bias,
                        'metrics': metrics,
                        'confusion_matrix': confusion.tolist(),
                    }
    rows.sort(key=lambda item: item['mIoU'], reverse=True)
    with (output_dir / 'validation_ensemble_grid.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'source_checkpoints': {
            'A': str(Path(args.checkpoint_a).resolve()),
            'corrected_F': str(Path(args.checkpoint_f).resolve()),
        },
        'validation_images': len(sample_ids),
        'selection_split': 'validation only',
        'test_split_used': False,
        'selected': best,
        'inference_models': 2,
        'intended_use': 'distillation teacher; not the final mobile student',
    }
    (output_dir / 'selection_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

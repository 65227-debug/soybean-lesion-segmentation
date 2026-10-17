#!/usr/bin/env python
"""Select and fold class-logit biases using the validation split only.

The selected biases are added to ``decode_head.conv_seg.bias`` so the output
remains one ordinary DeepLabV3+--MobileNetV3 checkpoint at inference time.
"""

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
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--supervised-root', default='data/dadou_student')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument(
        '--healthy-biases',
        default='-0.30,-0.20,-0.10,0,0.10,0.20,0.30')
    parser.add_argument(
        '--lesion-biases',
        default='-1.20,-1.00,-0.80,-0.60,-0.40,-0.20,0,0.20')
    parser.add_argument(
        '--selection-metric', default='mIoU',
        choices=('mIoU', 'IoU_lesion', 'Recall_lesion'),
        help='Validation metric used after applying the optional mIoU floor.')
    parser.add_argument(
        '--minimum-miou', type=float, default=0.0,
        help='Reject calibration candidates below this validation mIoU.')
    return parser.parse_args()


def parse_grid(text):
    return [float(value) for value in text.split(',')]


@torch.inference_mode()
def collect_validation_logits(model, loader, device):
    model.eval()
    logits_parts = []
    target_parts = []
    sample_ids = []
    for batch in loader:
        targets = batch['mask']
        logits = common.forward_logits(
            model, batch['image'].to(device, non_blocking=True),
            targets.shape[-2:])
        # Float16 is sufficient for threshold selection and limits host memory.
        logits_parts.append(logits.cpu().to(torch.float16).numpy())
        target_parts.append(targets.numpy().astype(np.uint8, copy=False))
        sample_ids.extend(batch['sample_id'])
    return (np.concatenate(logits_parts), np.concatenate(target_parts),
            sample_ids)


def confusion_for_biases(logits, targets, healthy_bias, lesion_bias):
    biased = logits.astype(np.float32, copy=True)
    biased[:, 1] += healthy_bias
    biased[:, 2] += lesion_bias
    prediction = biased.argmax(axis=1).astype(np.uint8)
    valid = targets < 3
    encoded = targets[valid].astype(np.int64) * 3 + prediction[valid]
    return np.bincount(encoded, minlength=9).reshape(3, 3)


def row_from_metrics(healthy_bias, lesion_bias, metrics):
    return {
        'healthy_logit_bias': healthy_bias,
        'lesion_logit_bias': lesion_bias,
        **metrics,
    }


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
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = common.build_student(args.model_config, device)
    model.load_state_dict(checkpoint['state_dict'])
    logits, targets, sample_ids = collect_validation_logits(
        model, loader, device)
    del model

    rows = []
    best = None
    for healthy_bias in parse_grid(args.healthy_biases):
        for lesion_bias in parse_grid(args.lesion_biases):
            confusion = confusion_for_biases(
                logits, targets, healthy_bias, lesion_bias)
            metrics = common.metrics_from_confusion(confusion)
            row = row_from_metrics(healthy_bias, lesion_bias, metrics)
            rows.append(row)
            if metrics['mIoU'] < args.minimum_miou:
                continue
            if (best is None
                    or metrics[args.selection_metric]
                    > best['metrics'][args.selection_metric]):
                best = {
                    'healthy_logit_bias': healthy_bias,
                    'lesion_logit_bias': lesion_bias,
                    'metrics': metrics,
                    'confusion_matrix': confusion.tolist(),
                }

    rows.sort(key=lambda item: item[args.selection_metric], reverse=True)
    with (output_dir / 'validation_bias_grid.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    state = checkpoint['state_dict']
    if best is None:
        raise RuntimeError(
            f'No calibration candidate reached minimum mIoU '
            f'{args.minimum_miou:.6f}.')
    bias_key = 'decode_head.conv_seg.bias'
    if bias_key not in state:
        raise KeyError(f'Missing classifier bias: {bias_key}')
    state[bias_key] = state[bias_key].clone()
    state[bias_key][1] += best['healthy_logit_bias']
    state[bias_key][2] += best['lesion_logit_bias']
    calibrated_checkpoint = dict(checkpoint)
    calibrated_checkpoint['state_dict'] = state
    calibrated_checkpoint['validation_calibration'] = {
        'healthy_logit_bias': best['healthy_logit_bias'],
        'lesion_logit_bias': best['lesion_logit_bias'],
        'metrics': best['metrics'],
        'selection_split': 'validation only',
        'selection_metric': args.selection_metric,
        'minimum_mIoU_constraint': args.minimum_miou,
    }
    torch.save(calibrated_checkpoint, output_dir / 'best_calibrated_student.pth')

    baseline = common.metrics_from_confusion(
        confusion_for_biases(logits, targets, 0.0, 0.0))
    summary = {
        'source_checkpoint': str(Path(args.checkpoint).resolve()),
        'validation_images': len(sample_ids),
        'selection_split': 'validation only',
        'selection_metric': args.selection_metric,
        'minimum_mIoU_constraint': args.minimum_miou,
        'test_split_used': False,
        'baseline_metrics': baseline,
        'selected': best,
        'single_model_checkpoint': True,
        'bias_folded_into_classifier': True,
    }
    (output_dir / 'calibration_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

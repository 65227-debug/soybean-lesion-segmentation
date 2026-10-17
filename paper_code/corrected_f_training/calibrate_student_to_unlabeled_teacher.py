#!/usr/bin/env python
"""Calibrate a student to a frozen teacher on an unlabeled target split.

Class-logit biases are selected by agreement with high-confidence teacher
pseudo labels on the target adaptation images.  A labeled *source-domain*
validation mIoU floor prevents catastrophic source-domain degradation.  The
annotated target test masks are never opened by this script.
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
    parser.add_argument('--teacher-config', required=True)
    parser.add_argument('--teacher-checkpoint', required=True)
    parser.add_argument('--target-manifest', required=True)
    parser.add_argument(
        '--source-val-manifest',
        default='data/dadou_student/supervised_val_manifest.csv')
    parser.add_argument(
        '--student-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--teacher-confidence', type=float, default=0.70)
    parser.add_argument('--source-miou-floor', type=float, default=0.79)
    parser.add_argument(
        '--healthy-biases',
        default='-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6')
    parser.add_argument(
        '--lesion-biases',
        default='-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2')
    return parser.parse_args()


def parse_grid(text):
    return [float(value) for value in text.split(',')]


def make_loader(manifest, args):
    dataset = common.StudentSegmentationDataset(Path(manifest), training=False)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=common.seed_worker)


@torch.inference_mode()
def collect_student_logits(model, loader, device):
    logits_parts = []
    targets_parts = []
    sample_ids = []
    model.eval()
    for batch in loader:
        target = batch['mask']
        logits = common.forward_logits(
            model, batch['image'].to(device, non_blocking=True),
            target.shape[-2:])
        logits_parts.append(logits.cpu().to(torch.float16).numpy())
        targets_parts.append(target.numpy().astype(np.uint8, copy=False))
        sample_ids.extend(batch['sample_id'])
    return (np.concatenate(logits_parts), np.concatenate(targets_parts),
            sample_ids)


@torch.inference_mode()
def collect_teacher_pseudo(model, loader, device):
    classes = []
    confidences = []
    model.eval()
    for batch in loader:
        target = batch['mask']
        logits = common.forward_logits(
            model, batch['image'].to(device, non_blocking=True),
            target.shape[-2:])
        probability = logits.softmax(dim=1)
        confidence, prediction = probability.max(dim=1)
        classes.append(prediction.cpu().to(torch.uint8).numpy())
        confidences.append(confidence.cpu().to(torch.float16).numpy())
    return np.concatenate(classes), np.concatenate(confidences)


def confusion(prediction, target, valid):
    encoded = target[valid].astype(np.int64) * 3 \
        + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=9).reshape(3, 3)


def prediction_for_bias(logits, healthy_bias, lesion_bias):
    biased = logits.astype(np.float32, copy=True)
    biased[:, 1] += healthy_bias
    biased[:, 2] += lesion_bias
    return biased.argmax(axis=1).astype(np.uint8)


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=True)
    student = common.build_student(args.student_config, device)
    student.load_state_dict(checkpoint['state_dict'], strict=True)
    target_loader = make_loader(args.target_manifest, args)
    source_loader = make_loader(args.source_val_manifest, args)
    target_logits, _, target_ids = collect_student_logits(
        student, target_loader, device)
    source_logits, source_targets, source_ids = collect_student_logits(
        student, source_loader, device)
    del student

    teacher = common.build_student(args.teacher_config, device)
    teacher_checkpoint = torch.load(
        args.teacher_checkpoint, map_location='cpu', weights_only=False)
    teacher.load_state_dict(teacher_checkpoint['state_dict'], strict=True)
    teacher_class, teacher_confidence = collect_teacher_pseudo(
        teacher, target_loader, device)
    del teacher, teacher_checkpoint
    high_confidence = teacher_confidence >= args.teacher_confidence
    if not high_confidence.any():
        raise RuntimeError('No target teacher pixel reached confidence floor')

    rows = []
    best = None
    for healthy_bias in parse_grid(args.healthy_biases):
        for lesion_bias in parse_grid(args.lesion_biases):
            target_prediction = prediction_for_bias(
                target_logits, healthy_bias, lesion_bias)
            target_confusion = confusion(
                target_prediction, teacher_class, high_confidence)
            target_metrics = common.metrics_from_confusion(target_confusion)
            source_prediction = prediction_for_bias(
                source_logits, healthy_bias, lesion_bias)
            source_valid = source_targets < 3
            source_confusion = confusion(
                source_prediction, source_targets, source_valid)
            source_metrics = common.metrics_from_confusion(source_confusion)
            eligible = source_metrics['mIoU'] >= args.source_miou_floor
            row = {
                'healthy_logit_bias': healthy_bias,
                'lesion_logit_bias': lesion_bias,
                'eligible': eligible,
                'target_teacher_mIoU': target_metrics['mIoU'],
                'target_teacher_pixel_accuracy':
                    target_metrics['pixel_accuracy'],
                'target_teacher_IoU_background':
                    target_metrics['IoU_background'],
                'target_teacher_IoU_healthy_leaf':
                    target_metrics['IoU_healthy_leaf'],
                'target_teacher_IoU_lesion': target_metrics['IoU_lesion'],
                'source_validation_mIoU': source_metrics['mIoU'],
                'source_validation_IoU_lesion': source_metrics['IoU_lesion'],
            }
            rows.append(row)
            key = (target_metrics['mIoU'], source_metrics['mIoU'])
            if eligible and (best is None or key > best['selection_key']):
                best = {
                    **row,
                    'selection_key': key,
                    'target_teacher_confusion_matrix':
                        target_confusion.tolist(),
                    'source_validation_confusion_matrix':
                        source_confusion.tolist(),
                }
    if best is None:
        raise RuntimeError(
            f'No candidate met source mIoU floor {args.source_miou_floor}')
    rows.sort(
        key=lambda item: (item['eligible'], item['target_teacher_mIoU'],
                          item['source_validation_mIoU']), reverse=True)
    write_csv(output_dir / 'target_teacher_bias_grid.csv', rows)

    state = checkpoint['state_dict']
    state = {key: value.clone() for key, value in state.items()}
    bias_key = 'decode_head.conv_seg.bias'
    state[bias_key][1] += best['healthy_logit_bias']
    state[bias_key][2] += best['lesion_logit_bias']
    calibrated = dict(checkpoint)
    calibrated['state_dict'] = state
    calibrated['unlabeled_target_calibration'] = {
        key: value for key, value in best.items() if key != 'selection_key'}
    checkpoint_path = output_dir / 'best_unlabeled_teacher_calibrated.pth'
    torch.save(calibrated, checkpoint_path)

    summary = {
        'source_checkpoint': str(Path(args.checkpoint).resolve()),
        'teacher_checkpoint': str(Path(args.teacher_checkpoint).resolve()),
        'target_manifest': str(Path(args.target_manifest).resolve()),
        'target_unlabeled_images': len(target_ids),
        'source_validation_images': len(source_ids),
        'target_ground_truth_masks_read': False,
        'teacher_confidence_floor': args.teacher_confidence,
        'teacher_high_confidence_pixel_percent':
            float(100.0 * high_confidence.mean()),
        'source_validation_mIoU_floor': args.source_miou_floor,
        'selection_metric': 'target_teacher_mIoU',
        'selected': {key: value for key, value in best.items()
                     if key != 'selection_key'},
        'output_checkpoint': str(checkpoint_path.resolve()),
        'test_split_used': False,
    }
    (output_dir / 'calibration_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

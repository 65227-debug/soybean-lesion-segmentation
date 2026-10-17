#!/usr/bin/env python
"""Compare the lesion teacher before and after independent U-Net gating."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ('background', 'healthy_leaf', 'lesion')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dir', required=True)
    parser.add_argument('--leaf-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--output-dir', default='outputs/dadou_leaf_gate_ablation')
    parser.add_argument(
        '--leaf-threshold',
        type=float,
        default=None,
        help=('Use the saved uint16 leaf probability map at this threshold. '
              'By default, use the exported argmax mask.'))
    parser.add_argument(
        '--dilation-pixels',
        type=int,
        default=0,
        help='Optional spatial tolerance applied to the binary leaf gate.')
    return parser.parse_args()


def confusion(prediction, target, num_classes):
    encoded = target.reshape(-1) * num_classes + prediction.reshape(-1)
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes)


def metrics(matrix, names):
    true_positive = np.diag(matrix).astype(np.float64)
    ground_truth = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(union, np.nan), where=union > 0)
    dice_denom = ground_truth + predicted
    dice = np.divide(
        2 * true_positive,
        dice_denom,
        out=np.full_like(dice_denom, np.nan),
        where=dice_denom > 0)
    precision = np.divide(
        true_positive, predicted, out=np.full_like(predicted, np.nan), where=predicted > 0)
    recall = np.divide(
        true_positive,
        ground_truth,
        out=np.full_like(ground_truth, np.nan),
        where=ground_truth > 0)
    result = {
        'aAcc': float(100 * true_positive.sum() / matrix.sum()),
        'mIoU': float(100 * np.nanmean(iou)),
        'mDice': float(100 * np.nanmean(dice)),
    }
    for index, name in enumerate(names):
        result[f'IoU_{name}'] = float(100 * iou[index])
        result[f'Dice_{name}'] = float(100 * dice[index])
        result[f'Precision_{name}'] = float(100 * precision[index])
        result[f'Recall_{name}'] = float(100 * recall[index])
    return result


def main():
    args = parse_args()
    teacher_dir = Path(args.teacher_dir)
    leaf_dir = Path(args.leaf_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    gated_dir = output_dir / 'gated_masks'
    gated_dir.mkdir(parents=True, exist_ok=True)

    teacher_matrix = np.zeros((3, 3), dtype=np.int64)
    gated_matrix = np.zeros((3, 3), dtype=np.int64)
    leaf_matrix = np.zeros((2, 2), dtype=np.int64)
    per_image = []
    for gt_path in sorted(gt_dir.glob('*.png')):
        teacher = cv2.imread(
            str(teacher_dir / f'{gt_path.stem}.mask.png'), cv2.IMREAD_UNCHANGED)
        if args.leaf_threshold is None:
            leaf = cv2.imread(
                str(leaf_dir / f'{gt_path.stem}.mask.png'),
                cv2.IMREAD_UNCHANGED)
        else:
            leaf_probability = cv2.imread(
                str(leaf_dir / f'{gt_path.stem}.leaf_prob16.png'),
                cv2.IMREAD_UNCHANGED)
            if leaf_probability is None:
                raise FileNotFoundError(
                    f'Missing leaf probability for {gt_path.stem}')
            leaf = (leaf_probability.astype(np.float32) / 65535.0 >=
                    args.leaf_threshold).astype(np.uint8)
        target = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if teacher is None or leaf is None or target is None:
            raise FileNotFoundError(f'Missing prediction for {gt_path.stem}')
        output_size = (target.shape[1], target.shape[0])
        if teacher.shape != target.shape:
            teacher = cv2.resize(teacher, output_size, interpolation=cv2.INTER_NEAREST)
        if leaf.shape != target.shape:
            leaf = cv2.resize(leaf, output_size, interpolation=cv2.INTER_NEAREST)
        if args.dilation_pixels:
            radius = args.dilation_pixels
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            leaf = cv2.dilate(leaf, kernel)
        target_leaf = (target > 0).astype(np.uint8)
        gated = teacher.copy()
        gated[(leaf == 0) & (gated > 0)] = 0
        cv2.imwrite(str(gated_dir / f'{gt_path.stem}.png'), gated)

        teacher_one = confusion(teacher, target, 3)
        gated_one = confusion(gated, target, 3)
        teacher_matrix += teacher_one
        gated_matrix += gated_one
        leaf_matrix += confusion(leaf, target_leaf, 2)
        teacher_metrics = metrics(teacher_one, CLASS_NAMES)
        gated_metrics = metrics(gated_one, CLASS_NAMES)
        per_image.append({
            'sample_id': gt_path.stem,
            'teacher_lesion_IoU': teacher_metrics['IoU_lesion'],
            'gated_lesion_IoU': gated_metrics['IoU_lesion'],
            'lesion_IoU_change': (
                gated_metrics['IoU_lesion'] - teacher_metrics['IoU_lesion']),
        })

    teacher_result = metrics(teacher_matrix, CLASS_NAMES)
    gated_result = metrics(gated_matrix, CLASS_NAMES)
    leaf_result = metrics(leaf_matrix, ('background', 'leaf'))
    gate_result_key = (
        'teacher_plus_unet_hard_gate'
        if args.leaf_threshold is None and args.dilation_pixels == 0 else
        'teacher_plus_unet_probability_gate')
    summary = {
        'gate_parameters': {
            'leaf_probability_threshold': args.leaf_threshold,
            'uses_exported_argmax_mask': args.leaf_threshold is None,
            'dilation_pixels': args.dilation_pixels,
            'selection_split': 'val',
        },
        'teacher_only': teacher_result,
        'independent_unet_leaf': leaf_result,
        gate_result_key: gated_result,
        'gated_minus_teacher': {
            key: gated_result[key] - teacher_result[key]
            for key in teacher_result
        },
        'test_split_used': False,
    }
    with (output_dir / 'ablation_summary.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / 'per_image_lesion_iou.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

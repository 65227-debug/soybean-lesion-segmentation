#!/usr/bin/env python
"""Select dual-evidence pixel thresholds on pixel-annotated validation images."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from gating_core import GatingConfig, load_config, save_config


def parse_float_grid(value):
    return [float(item) for item in value.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--validation-base-maps', default='data/dadou/06_验证集基础图')
    parser.add_argument('--mask-dir', default='data/dadou_mmseg/val/masks')
    parser.add_argument(
        '--base-config',
        default='paper_code/dual_evidence_gating/default_config.json')
    parser.add_argument(
        '--output-dir', default='data/dadou/06_验证集门控调参')
    parser.add_argument(
        '--selected-config-output',
        default='paper_code/dual_evidence_gating/selected_config.json')
    parser.add_argument('--tau-p-grid', default='0.80,0.85,0.90,0.95')
    parser.add_argument('--tau-n-grid', default='0.05,0.10,0.15,0.20')
    parser.add_argument('--tau-s-grid', default='0.30,0.40,0.45,0.50,0.60')
    parser.add_argument('--tau-sn-grid', default='0.10,0.20,0.30')
    parser.add_argument('--min-positive-precision', type=float, default=0.80)
    parser.add_argument('--min-negative-precision', type=float, default=0.94)
    parser.add_argument('--min-positive-recall', type=float, default=0.10)
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def fbeta(precision, recall, beta=0.5):
    beta_squared = beta ** 2
    denominator = beta_squared * precision + recall
    return safe_div((1 + beta_squared) * precision * recall, denominator)


def calculate_metrics(p_seg, p_leaf, a_sem, ground_truth, config):
    inner = np.concatenate([item.astype(bool) for item in p_leaf['inner']])
    p_leaf_values = np.concatenate(p_leaf['probability'])
    p_seg_values = np.concatenate(p_seg)
    a_sem_values = np.concatenate(a_sem)
    gt = np.concatenate(ground_truth)

    positive = (
        (p_seg_values >= config.tau_p)
        & (a_sem_values >= config.tau_s)
        & inner)
    negative = (
        (p_seg_values <= config.tau_n)
        & (a_sem_values <= config.tau_sn)
        & inner)
    background = p_leaf_values <= config.background_leaf_threshold
    accepted = positive | negative | background

    gt_positive = gt == 2
    gt_negative = gt == 1
    gt_background = gt == 0
    positive_tp = int((positive & gt_positive).sum())
    negative_tp = int((negative & gt_negative).sum())
    background_tp = int((background & gt_background).sum())
    positive_precision = safe_div(positive_tp, int(positive.sum()))
    positive_recall = safe_div(positive_tp, int(gt_positive.sum()))
    negative_precision = safe_div(negative_tp, int(negative.sum()))
    negative_recall = safe_div(negative_tp, int(gt_negative.sum()))
    background_precision = safe_div(background_tp, int(background.sum()))
    background_recall = safe_div(background_tp, int(gt_background.sum()))
    correct = positive_tp + negative_tp + background_tp
    leaf_accepted = positive | negative
    gt_leaf = gt_positive | gt_negative
    leaf_accepted_ratio = safe_div(int(leaf_accepted.sum()), int(gt_leaf.sum()))
    accepted_accuracy = safe_div(correct, int(accepted.sum()))
    positive_f05 = fbeta(positive_precision, positive_recall, beta=0.5)
    negative_f05 = fbeta(negative_precision, negative_recall, beta=0.5)
    coverage_score = min(leaf_accepted_ratio / 0.50, 1.0)
    objective = (
        0.50 * positive_f05
        + 0.20 * negative_f05
        + 0.20 * accepted_accuracy
        + 0.10 * coverage_score)
    return {
        'tau_p': config.tau_p,
        'tau_n': config.tau_n,
        'tau_s': config.tau_s,
        'tau_sn': config.tau_sn,
        'positive_precision': positive_precision,
        'positive_recall': positive_recall,
        'positive_f0.5': positive_f05,
        'negative_precision': negative_precision,
        'negative_recall': negative_recall,
        'negative_f0.5': negative_f05,
        'background_precision': background_precision,
        'background_recall': background_recall,
        'accepted_pixel_accuracy': accepted_accuracy,
        'accepted_all_pixel_ratio': safe_div(int(accepted.sum()), gt.size),
        'accepted_leaf_pixel_ratio': leaf_accepted_ratio,
        'ignored_all_pixel_ratio': 1.0 - safe_div(int(accepted.sum()), gt.size),
        'objective': objective,
        'accepted_positive_pixels': int(positive.sum()),
        'accepted_negative_pixels': int(negative.sum()),
        'accepted_background_pixels': int(background.sum()),
    }


def teacher_baseline_metrics(p_seg, ground_truth):
    prediction = np.concatenate(p_seg) >= 0.5
    gt = np.concatenate(ground_truth) == 2
    true_positive = int((prediction & gt).sum())
    precision = safe_div(true_positive, int(prediction.sum()))
    recall = safe_div(true_positive, int(gt.sum()))
    return {
        'threshold': 0.5,
        'precision': precision,
        'recall': recall,
        'dice_f1': fbeta(precision, recall, beta=1.0),
        'f0.5': fbeta(precision, recall, beta=0.5),
    }


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_validation_data(base_maps_dir, mask_dir):
    base_maps_dir = Path(base_maps_dir)
    manifest_path = base_maps_dir / '三张基础图生成清单.csv'
    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    p_seg = []
    p_leaf_probability = []
    p_leaf_inner = []
    a_sem = []
    ground_truth = []
    sample_rows = []
    for row in manifest:
        numeric_path = base_maps_dir / row['numeric_npz']
        arrays = np.load(numeric_path)
        mask_path = Path(mask_dir) / f"{Path(row['filename']).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        target_shape = arrays['p_seg'].shape
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask, (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST)
        unique = set(np.unique(mask).tolist())
        if not unique <= {0, 1, 2}:
            raise ValueError(f'Unexpected mask values {unique} in {mask_path}')
        p_seg.append(arrays['p_seg'].reshape(-1))
        p_leaf_probability.append(arrays['p_leaf'].reshape(-1))
        p_leaf_inner.append(arrays['m_inner'].reshape(-1))
        a_sem.append(arrays['a_sem'].reshape(-1))
        ground_truth.append(mask.astype(np.uint8).reshape(-1))
        sample_rows.append(row)
    return {
        'p_seg': p_seg,
        'p_leaf': {
            'probability': p_leaf_probability,
            'inner': p_leaf_inner,
        },
        'a_sem': a_sem,
        'ground_truth': ground_truth,
        'rows': sample_rows,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.base_config)
    data = load_validation_data(args.validation_base_maps, args.mask_dir)

    results = []
    grids = itertools.product(
        parse_float_grid(args.tau_p_grid),
        parse_float_grid(args.tau_n_grid),
        parse_float_grid(args.tau_s_grid),
        parse_float_grid(args.tau_sn_grid))
    for tau_p, tau_n, tau_s, tau_sn in grids:
        if tau_n >= tau_p or tau_sn >= tau_s:
            continue
        config = GatingConfig(**{
            **base_config.__dict__,
            'tau_p': tau_p,
            'tau_n': tau_n,
            'tau_s': tau_s,
            'tau_sn': tau_sn,
        })
        results.append(calculate_metrics(
            data['p_seg'], data['p_leaf'], data['a_sem'],
            data['ground_truth'], config))
    results.sort(key=lambda row: row['objective'], reverse=True)
    for rank, row in enumerate(results, start=1):
        row['objective_rank'] = rank
        row['meets_precision_recall_constraints'] = (
            row['positive_precision'] >= args.min_positive_precision
            and row['negative_precision'] >= args.min_negative_precision
            and row['positive_recall'] >= args.min_positive_recall)
    eligible = [
        row for row in results if row['meets_precision_recall_constraints']]
    selected = eligible[0] if eligible else results[0]
    selected_config = GatingConfig(**{
        **base_config.__dict__,
        'tau_p': selected['tau_p'],
        'tau_n': selected['tau_n'],
        'tau_s': selected['tau_s'],
        'tau_sn': selected['tau_sn'],
    })

    write_csv(output_dir / '全部阈值组合.csv', results)
    save_config(
        args.selected_config_output, selected_config,
        extra={
            'selection': {
                'split': 'validation only',
                'validation_images': len(data['rows']),
                'classes': sorted({row['class_name'] for row in data['rows']}),
                'objective': (
                    '0.50*lesion_F0.5 + 0.20*healthy_F0.5 + '
                    '0.20*accepted_accuracy + 0.10*leaf_coverage_score'),
                'constraints': {
                    'min_positive_precision': args.min_positive_precision,
                    'min_negative_precision': args.min_negative_precision,
                    'min_positive_recall': args.min_positive_recall,
                },
                'eligible_combinations': len(eligible),
                'selected_metrics': selected,
                'test_split_used': False,
            }
        })
    summary = {
        'validation_images': len(data['rows']),
        'validation_class_counts': {
            name: sum(row['class_name'] == name for row in data['rows'])
            for name in sorted({row['class_name'] for row in data['rows']})
        },
        'grid_combinations_evaluated': len(results),
        'eligible_combinations': len(eligible),
        'teacher_0.5_baseline': teacher_baseline_metrics(
            data['p_seg'], data['ground_truth']),
        'selected': selected,
        'selected_config': str(Path(args.selected_config_output).resolve()),
        'test_split_used': False,
    }
    (output_dir / '阈值选择汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

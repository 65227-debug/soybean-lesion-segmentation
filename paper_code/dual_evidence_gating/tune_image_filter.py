#!/usr/bin/env python
"""Select the whole-image overlap threshold on the validation subset."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gating_core import calculate_image_statistics, gate_pixels, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--validation-base-maps', default='data/dadou/06_验证集基础图')
    parser.add_argument('--mask-dir', default='data/dadou_mmseg/val/masks')
    parser.add_argument(
        '--config',
        default='paper_code/dual_evidence_gating/selected_config.json')
    parser.add_argument(
        '--output-dir', default='data/dadou/06_验证集门控调参')
    parser.add_argument(
        '--overlap-grid',
        default='0,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10')
    parser.add_argument('--target-positive-precision', type=float, default=0.95)
    parser.add_argument('--min-image-retention', type=float, default=0.50)
    parser.add_argument(
        '--min-blur-score', type=float, default=5.0,
        help='Only rejects extremely blurred images; kept fixed during search.')
    return parser.parse_args()


def safe_div(a, b):
    return a / b if b else 0.0


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.validation_base_maps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    with (root / '三张基础图生成清单.csv').open(
            encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))

    image_records = []
    for row in rows:
        arrays = np.load(root / row['numeric_npz'])
        labels, _weights, masks = gate_pixels(
            arrays['p_seg'], arrays['p_leaf'], arrays['a_sem'], config)
        statistics = calculate_image_statistics(labels, masks)
        mask_path = Path(args.mask_dir) / f"{Path(row['filename']).stem}.png"
        ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        ground_truth = cv2.resize(
            ground_truth, (labels.shape[1], labels.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        positive = labels == 2
        true_positive = int((positive & (ground_truth == 2)).sum())
        image = cv2.imread(row['source_path'], cv2.IMREAD_GRAYSCALE)
        blur_score = float(cv2.Laplacian(image, cv2.CV_64F).var())
        image_records.append({
            'filename': row['filename'],
            'class_name': row['class_name'],
            **statistics,
            'predicted_positive_pixels': int(positive.sum()),
            'true_positive_pixels': true_positive,
            'positive_precision': safe_div(true_positive, int(positive.sum())),
            'semantic_label_probability': float(
                row['semantic_label_probability']),
            'blur_score': blur_score,
        })
    write_csv(output_dir / '验证集逐图门控质量.csv', image_records)

    search_rows = []
    for threshold in [float(x) for x in args.overlap_grid.split(',')]:
        accepted = []
        for row in image_records:
            keep = (
                row['semantic_segmentation_overlap_ratio'] >= threshold
                and row['positive_pixels'] >= config.min_positive_pixels
                and row['leaf_inner_area_percent']
                >= config.min_leaf_inner_area_percent
                and row['lesion_candidate_leaf_percent']
                <= config.max_lesion_candidate_leaf_percent
                and row['semantic_label_probability']
                >= config.min_semantic_confidence
                and row['blur_score'] >= args.min_blur_score)
            if keep:
                accepted.append(row)
        predicted = sum(row['predicted_positive_pixels'] for row in accepted)
        true_positive = sum(row['true_positive_pixels'] for row in accepted)
        aggregate_precision = safe_div(true_positive, predicted)
        retention = safe_div(len(accepted), len(image_records))
        search_rows.append({
            'min_overlap_ratio': threshold,
            'accepted_images': len(accepted),
            'image_retention': retention,
            'aggregate_positive_precision': aggregate_precision,
            'mean_per_image_positive_precision': (
                float(np.mean([row['positive_precision'] for row in accepted]))
                if accepted else 0.0),
            'accepted_positive_pixels': predicted,
            'meets_constraints': (
                aggregate_precision >= args.target_positive_precision
                and retention >= args.min_image_retention),
        })
    eligible = [row for row in search_rows if row['meets_constraints']]
    if not eligible:
        print('No image-filter threshold meets the requested constraints.',
              file=sys.stderr)
        selected = max(
            search_rows,
            key=lambda row: row['aggregate_positive_precision']
            * np.sqrt(row['image_retention']))
        fallback = True
    else:
        selected = max(
            eligible,
            key=lambda row: (
                row['image_retention'],
                row['aggregate_positive_precision'],
                -row['min_overlap_ratio']))
        fallback = False
    write_csv(output_dir / '整图重叠阈值搜索.csv', search_rows)

    config_path = Path(args.config)
    config_json = json.loads(config_path.read_text(encoding='utf-8'))
    config_json['gating']['min_overlap_ratio'] = selected['min_overlap_ratio']
    config_json['gating']['min_blur_score'] = args.min_blur_score
    config_json['image_filter_selection'] = {
        'split': 'validation only',
        'validation_images': len(image_records),
        'target_aggregate_positive_precision': args.target_positive_precision,
        'min_image_retention': args.min_image_retention,
        'selected': selected,
        'fallback_used': fallback,
        'test_split_used': False,
    }
    config_path.write_text(
        json.dumps(config_json, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {
        'selected_min_overlap_ratio': selected['min_overlap_ratio'],
        'selected_min_blur_score': args.min_blur_score,
        'selected_validation_metrics': selected,
        'fallback_used': fallback,
        'config_updated': str(config_path.resolve()),
        'test_split_used': False,
    }
    (output_dir / '整图筛选阈值汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

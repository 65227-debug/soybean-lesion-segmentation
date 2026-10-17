#!/usr/bin/env python
"""Paired image-level bootstrap intervals for F versus ablation groups."""

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', default='work_dirs/dadou_ablation_comparison')
    parser.add_argument('--reference', default='F')
    parser.add_argument('--groups', nargs='+', default=list('ABCDE'))
    parser.add_argument('--metrics', nargs='+', default=['mIoU', 'IoU_lesion'])
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260803)
    parser.add_argument(
        '--output',
        default='work_dirs/dadou_ablation_comparison/'
                'F_vs_ablation_paired_bootstrap.csv')
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {row['sample_id']: row for row in rows}


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    reference_rows = read_rows(
        input_dir / f'test_per_image_{args.reference}.csv')
    rng = np.random.default_rng(args.seed)
    results = []
    for group in args.groups:
        group_rows = read_rows(input_dir / f'test_per_image_{group}.csv')
        common_ids = sorted(set(reference_rows) & set(group_rows))
        for metric in args.metrics:
            differences = []
            for sample_id in common_ids:
                reference_value = float(reference_rows[sample_id][metric])
                group_value = float(group_rows[sample_id][metric])
                if np.isfinite(reference_value) and np.isfinite(group_value):
                    differences.append(reference_value - group_value)
            differences = np.asarray(differences, dtype=np.float64)
            indices = rng.integers(
                0, len(differences), size=(args.resamples, len(differences)))
            bootstrap_means = differences[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
            results.append({
                'comparison': f'{args.reference}-{group}',
                'metric': metric,
                'paired_images': len(differences),
                'mean_difference_percentage_points':
                    100.0 * differences.mean(),
                'ci95_lower_percentage_points': 100.0 * lower,
                'ci95_upper_percentage_points': 100.0 * upper,
                'ci_excludes_zero': bool(lower > 0 or upper < 0),
                'resamples': args.resamples,
                'seed': args.seed,
            })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(
            f"{row['comparison']} {row['metric']}: "
            f"{row['mean_difference_percentage_points']:+.2f} pp "
            f"[95% CI {row['ci95_lower_percentage_points']:+.2f}, "
            f"{row['ci95_upper_percentage_points']:+.2f}]")


if __name__ == '__main__':
    main()

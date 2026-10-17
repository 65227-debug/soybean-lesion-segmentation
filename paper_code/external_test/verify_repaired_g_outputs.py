#!/usr/bin/env python
"""Strictly verify repaired-G external evaluation artifacts."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


VARIANTS = (
    'G_old_calibrated', 'G_old_uncalibrated',
    'G4_source_uncalibrated', 'G_UDA_epoch12', 'G_repaired')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path(parse_args().output_dir).resolve()
    errors = []
    prediction_counts = {}
    for variant in VARIANTS:
        masks = sorted((root / 'predictions' / variant / 'masks').glob('*.png'))
        overlays = sorted(
            (root / 'predictions' / variant / 'overlays').glob('*.jpg'))
        prediction_counts[variant] = {
            'masks': len(masks), 'overlays': len(overlays)}
        if len(masks) != 80 or len(overlays) != 80:
            errors.append(f'{variant}: expected 80 masks/overlays')
        for path in masks:
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (512, 512):
                errors.append(f'invalid mask shape: {path}')
                continue
            values = set(int(value) for value in np.unique(mask))
            if not values <= {0, 1, 2}:
                errors.append(f'invalid mask values {values}: {path}')

    expected_csv_rows = {
        '01_修复前后总体指标.csv': 7,
        '02_修复后分病种指标.csv': 28,
        '03_修复前后逐图指标.csv': 560,
        '04_排除Healthy不完整标注后的60张指标.csv': 7,
        '05_修复G配对Bootstrap.csv': 6,
    }
    actual_csv_rows = {}
    for name, expected in expected_csv_rows.items():
        actual = len(csv_rows(root / name))
        actual_csv_rows[name] = actual
        if actual != expected:
            errors.append(f'{name}: expected {expected}, got {actual}')

    qualitative = len(list((root / 'qualitative_all_80').glob('*.jpg')))
    if qualitative != 80:
        errors.append(f'qualitative images: expected 80, got {qualitative}')
    prohibited = [str(path) for path in root.rglob('*')
                  if path.suffix.lower() in {'.pth', '.pt', '.ckpt'}]
    if prohibited:
        errors.append(f'model files found in paper output: {prohibited}')
    adaptation = json.loads((
        root / 'training_and_calibration_records/'
        'adaptation_dataset_summary.json').read_text(encoding='utf-8'))
    if adaptation['adaptation_test_sha256_overlap'] != 0:
        errors.append('adaptation/test SHA-256 overlap is non-zero')
    if adaptation['scene_id_overlap_count'] != 0:
        errors.append('adaptation/test scene overlap is non-zero')

    report = {
        'ok': not errors,
        'errors': errors,
        'prediction_counts': prediction_counts,
        'qualitative_images': qualitative,
        'csv_data_rows': actual_csv_rows,
        'adaptation_test_sha256_overlap':
            adaptation['adaptation_test_sha256_overlap'],
        'adaptation_test_scene_overlap': adaptation['scene_id_overlap_count'],
        'model_weight_files_inside_result': prohibited,
    }
    (root / '07_严格核验报告.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

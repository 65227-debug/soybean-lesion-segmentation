#!/usr/bin/env python
"""Audit pseudo-label and weight outputs before student training."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', default='data/dadou/06_07_08_双证据门控')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_dir)
    with (root / '全部图像质量与门控清单.csv').open(
            encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    accepted = 0
    for row in rows:
        candidate_path = root / row['candidate_label_path']
        final_path = root / row['pseudo_label_path']
        weight_path = root / row['pixel_weight_path']
        numeric_path = root / row['numeric_npz_path']
        required_paths = (
            candidate_path, final_path, weight_path, numeric_path)
        missing_paths = [path for path in required_paths if not path.is_file()]
        errors.extend(f'missing:{path}' for path in missing_paths)
        if missing_paths:
            continue
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
        final = cv2.imread(str(final_path), cv2.IMREAD_UNCHANGED)
        weight = cv2.imread(str(weight_path), cv2.IMREAD_UNCHANGED)
        if candidate.shape != (512, 512) or final.shape != (512, 512):
            errors.append(f'shape:{row["filename"]}')
        if set(np.unique(candidate).tolist()) - {0, 1, 2, 255}:
            errors.append(f'candidate_values:{row["filename"]}')
        if set(np.unique(final).tolist()) - {0, 1, 2, 255}:
            errors.append(f'final_values:{row["filename"]}')
        if weight.dtype != np.uint16 or weight.shape != final.shape:
            errors.append(f'weight_format:{row["filename"]}')
        numeric = np.load(numeric_path)
        ignored = numeric['pseudo_label'] == 255
        if np.any(numeric['pixel_weight'][ignored] != 0):
            errors.append(f'ignored_nonzero_weight:{row["filename"]}')
        is_accepted = row['image_accepted'].lower() == 'true'
        if is_accepted:
            accepted += 1
        elif not np.all(final == 255) or np.any(weight != 0):
            errors.append(f'rejected_not_zeroed:{row["filename"]}')
    report = {
        'images_audited': len(rows),
        'accepted_images': accepted,
        'rejected_images': len(rows) - accepted,
        'errors': len(errors),
        'first_errors': errors[:50],
        'audit_passed': len(errors) == 0,
    }
    (root / '输出完整性审计.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f'Audit failed with {len(errors)} errors')


if __name__ == '__main__':
    main()

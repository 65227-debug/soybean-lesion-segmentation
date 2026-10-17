#!/usr/bin/env python
"""Tune corrected F gates using only pixel-annotated validation images."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from corrected_gating import CorrectedGatingConfig, save_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-maps', required=True)
    parser.add_argument('--mask-dir', default='data/dadou_mmseg/val/masks')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--selected-config', required=True)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def prf(mask, truth):
    true_positive = int((mask & truth).sum())
    precision = true_positive / int(mask.sum()) if mask.any() else 0.0
    recall = true_positive / int(truth.sum()) if truth.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'accepted_pixels': int(mask.sum()),
    }


def choose(rows, min_precision):
    eligible = [row for row in rows if row['precision'] >= min_precision]
    pool = eligible or rows
    selected = max(pool, key=lambda row: (row['f1'], row['precision']))
    return selected, bool(eligible)


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    base_dir = Path(args.base_maps)
    manifest = read_csv(base_dir / '三张基础图生成清单.csv')
    teacher, leaf, semantic, truth = [], [], [], []
    for row in manifest:
        arrays = np.load(base_dir / row['numeric_npz'])
        if 'p_teacher' not in arrays:
            raise ValueError('Base maps must contain p_teacher')
        mask = cv2.imread(
            str(Path(args.mask_dir) / f"{Path(row['filename']).stem}.png"),
            cv2.IMREAD_UNCHANGED)
        shape = arrays['p_leaf'].shape
        if mask.shape != shape:
            mask = cv2.resize(
                mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        teacher.append(arrays['p_teacher'].astype(np.float32))
        leaf.append(arrays['p_leaf'].astype(np.float32))
        semantic.append(arrays['a_sem'].astype(np.float32))
        truth.append(mask.astype(np.uint8))
    teacher = np.concatenate([x.reshape(3, -1) for x in teacher], axis=1)
    leaf = np.concatenate([x.reshape(-1) for x in leaf])
    semantic = np.concatenate([x.reshape(-1) for x in semantic])
    truth = np.concatenate([x.reshape(-1) for x in truth])

    # The stored inner map already uses threshold 0.5 and erosion radius 5.
    inner = np.concatenate([
        np.load(base_dir / row['numeric_npz'])['m_inner'].reshape(-1).astype(bool)
        for row in manifest
    ])
    p_background, p_healthy, p_lesion = teacher

    lesion_rows = []
    for tau_p, tau_s in itertools.product(
            (0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
            (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)):
        row = {'lesion_teacher_threshold': tau_p,
               'lesion_semantic_threshold': tau_s}
        row.update(prf(
            inner & (p_lesion >= tau_p) & (semantic >= tau_s), truth == 2))
        lesion_rows.append(row)
    healthy_rows = []
    for tau_h, tau_n in itertools.product(
            (0.60, 0.70, 0.80, 0.90, 0.95),
            (0.05, 0.10, 0.15, 0.20)):
        row = {'healthy_teacher_threshold': tau_h,
               'lesion_negative_threshold': tau_n}
        row.update(prf(
            inner & (p_healthy >= tau_h) & (p_lesion <= tau_n), truth == 1))
        healthy_rows.append(row)
    background_rows = []
    for tau_bg, tau_leaf in itertools.product(
            (0.60, 0.70, 0.80, 0.90, 0.95),
            (0.03, 0.05, 0.10, 0.15)):
        row = {'background_teacher_threshold': tau_bg,
               'background_leaf_threshold': tau_leaf}
        row.update(prf(
            (p_background >= tau_bg) & (leaf <= tau_leaf), truth == 0))
        background_rows.append(row)

    selected_lesion, lesion_met = choose(lesion_rows, 0.90)
    selected_healthy, healthy_met = choose(healthy_rows, 0.97)
    selected_background, background_met = choose(background_rows, 0.995)
    config = CorrectedGatingConfig(
        lesion_teacher_threshold=selected_lesion['lesion_teacher_threshold'],
        lesion_semantic_threshold=selected_lesion['lesion_semantic_threshold'],
        healthy_teacher_threshold=selected_healthy['healthy_teacher_threshold'],
        lesion_negative_threshold=selected_healthy['lesion_negative_threshold'],
        background_teacher_threshold=selected_background[
            'background_teacher_threshold'],
        background_leaf_threshold=selected_background[
            'background_leaf_threshold'])
    selection = {
        'split': 'validation only',
        'validation_images': len(manifest),
        'objective': 'maximize per-class F1 subject to precision constraints',
        'precision_constraints': {
            'lesion': 0.90, 'healthy': 0.97, 'background': 0.995},
        'constraints_met': {
            'lesion': lesion_met, 'healthy': healthy_met,
            'background': background_met},
        'selected_lesion': selected_lesion,
        'selected_healthy': selected_healthy,
        'selected_background': selected_background,
        'test_split_used': False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'lesion_grid.csv', lesion_rows)
    write_csv(output_dir / 'healthy_grid.csv', healthy_rows)
    write_csv(output_dir / 'background_grid.csv', background_rows)
    save_config(args.selected_config, config, selection)
    (output_dir / 'selection_summary.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'gating': config.__dict__, **selection},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


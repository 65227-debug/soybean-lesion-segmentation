#!/usr/bin/env python
"""Evaluate LayerCAM localization on available pixel-annotated validation images."""

import argparse
import csv
import json
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam-dir', default='outputs/dadou_semantic_layercam_val')
    parser.add_argument('--val-manifest', default='data/dadou_semantic/val_manifest.csv')
    parser.add_argument('--exact-manifest', default='data/dadou/metadata/精标总清单.csv')
    parser.add_argument('--gt-dir', default='data/dadou_mmseg/val/masks')
    parser.add_argument('--top-fraction', type=float, default=0.20)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def source_basename(row):
    if row.get('source_filename'):
        return row['source_filename']
    return PureWindowsPath(row['source_image_path']).name


def safe_mean(values):
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    cam_dir = Path(args.cam_dir)
    gt_dir = Path(args.gt_dir)
    validation = read_csv(args.val_manifest)
    exact = read_csv(args.exact_manifest)
    source_to_annotation = {
        source_basename(row): row['annotation_filename'] for row in exact
    }
    per_image = []
    skipped = []
    for row in validation:
        source_name = Path(row['linked_path']).name
        annotation_name = source_to_annotation.get(source_name)
        if annotation_name is None:
            skipped.append({'source': source_name, 'reason': 'not_in_exact_manifest'})
            continue
        sample_id = Path(annotation_name).stem
        gt_path = gt_dir / f'{sample_id}.png'
        cam_path = (
            cam_dir / row['class_name'] /
            f'{Path(source_name).stem}.multilayer_layercam16.png')
        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        cam = cv2.imread(str(cam_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            skipped.append({'source': source_name, 'reason': 'gt_mask_not_available'})
            continue
        if cam is None:
            skipped.append({'source': source_name, 'reason': 'cam_not_available'})
            continue
        cam = cam.astype(np.float32) / 65535.0
        if cam.shape != gt.shape:
            cam = cv2.resize(
                cam, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        leaf = gt > 0
        lesion = gt == 2
        background = gt == 0
        total_energy = float(cam.sum()) + 1e-12
        lesion_area_fraction = float(lesion.mean())
        lesion_energy_fraction = float(cam[lesion].sum() / total_energy)
        leaf_energy_fraction = float(cam[leaf].sum() / total_energy)
        maximum = np.unravel_index(int(np.argmax(cam)), cam.shape)
        threshold = float(np.quantile(cam, 1.0 - args.top_fraction))
        top = cam >= threshold
        top_pixels = max(int(top.sum()), 1)
        lesion_pixels = max(int(lesion.sum()), 1)
        per_image.append({
            'source_filename': source_name,
            'sample_id': sample_id,
            'class_name': row['class_name'],
            'lesion_area_fraction': lesion_area_fraction,
            'lesion_energy_fraction': lesion_energy_fraction,
            'leaf_energy_fraction': leaf_energy_fraction,
            'lesion_energy_enrichment': (
                lesion_energy_fraction / max(lesion_area_fraction, 1e-12)),
            'mean_cam_lesion': float(cam[lesion].mean()) if lesion.any() else 0.0,
            'mean_cam_healthy_leaf': float(cam[leaf & ~lesion].mean()),
            'mean_cam_background': float(cam[background].mean()),
            'cam_argmax_in_lesion': bool(lesion[maximum]),
            'cam_argmax_in_leaf': bool(leaf[maximum]),
            'top_cam_lesion_precision': float((top & lesion).sum() / top_pixels),
            'top_cam_lesion_recall': float((top & lesion).sum() / lesion_pixels),
        })

    output_csv = cam_dir / 'layercam_localization_per_image.csv'
    with output_csv.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary = {
        'annotated_images_evaluated': len(per_image),
        'validation_images_without_available_mask': len(skipped),
        'top_cam_fraction': args.top_fraction,
        'mean_lesion_area_percent': 100 * safe_mean([
            row['lesion_area_fraction'] for row in per_image]),
        'mean_cam_energy_in_lesion_percent': 100 * safe_mean([
            row['lesion_energy_fraction'] for row in per_image]),
        'mean_cam_energy_in_leaf_percent': 100 * safe_mean([
            row['leaf_energy_fraction'] for row in per_image]),
        'mean_lesion_energy_enrichment': safe_mean([
            row['lesion_energy_enrichment'] for row in per_image]),
        'cam_argmax_in_lesion_percent': 100 * safe_mean([
            row['cam_argmax_in_lesion'] for row in per_image]),
        'cam_argmax_in_leaf_percent': 100 * safe_mean([
            row['cam_argmax_in_leaf'] for row in per_image]),
        'mean_top_cam_lesion_precision_percent': 100 * safe_mean([
            row['top_cam_lesion_precision'] for row in per_image]),
        'mean_top_cam_lesion_recall_percent': 100 * safe_mean([
            row['top_cam_lesion_recall'] for row in per_image]),
        'skipped': skipped,
        'test_split_used': False,
    }
    with (cam_dir / 'layercam_localization_summary.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

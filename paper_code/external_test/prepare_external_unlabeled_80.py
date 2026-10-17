#!/usr/bin/env python
"""Prepare the 80 non-test MH-SoyaHealthVision images for label-free adaptation.

The original selection contains 160 images (40 per category).  Eighty images
have already been frozen as the annotated external test set.  This script
selects the complementary 80 images by source SHA-256/name, resizes them to
512x512, and writes dummy all-valid masks/weights for online teacher
distillation.  It never reads an external ground-truth mask.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CATEGORY_ORDER = (
    'Healthy', 'Frogeye_leaf_spot', 'Rust', 'Septoria_brown_spot')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--selection-manifest',
        default='data/external_annotations/'
                'metadata/selection_manifest.csv')
    parser.add_argument(
        '--complete-images-root',
        default=str(PROJECT_ROOT / (
            'data/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset '
            'for Integrated Crop Health Assessment/'
            'External_Segmentation_Test_160_待标注/images')))
    parser.add_argument(
        '--frozen-test-manifest',
        default=str(PROJECT_ROOT / (
            'data/MH-SoyaHealthVision_External_Test_80_3class/'
            'external_test_manifest.csv')))
    parser.add_argument(
        '--output-root',
        default=str(PROJECT_ROOT / (
            'data/MH-SoyaHealthVision_External_Unlabeled_80_UDA')))
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    selection_manifest = Path(args.selection_manifest).resolve()
    complete_images_root = Path(args.complete_images_root).resolve()
    frozen_test_manifest = Path(args.frozen_test_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    images_dir = output_root / 'images_512'
    masks_dir = output_root / 'dummy_masks_512'
    weights_dir = output_root / 'weights_512'
    for directory in (images_dir, masks_dir, weights_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = [row for row in read_csv(selection_manifest)
                if row['selected'].strip().lower() == 'true']
    frozen = read_csv(frozen_test_manifest)
    if len(selected) != 160 or len(frozen) != 80:
        raise RuntimeError(
            f'Expected selected/test counts 160/80, got '
            f'{len(selected)}/{len(frozen)}')

    frozen_hashes = {row['sha256'] for row in frozen}
    frozen_keys = {
        (row['class_name'], Path(row['source_image_path']).name)
        for row in frozen}
    adaptation_rows = []
    for row in selected:
        key = (row['class_name'], row['output_name'])
        if row['sha256'] in frozen_hashes or key in frozen_keys:
            continue
        source = complete_images_root / row['class_name'] / row['output_name']
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_hash = sha256(source)
        if actual_hash != row['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {source}')
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Unreadable image: {source}')
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        sample_id = f"UDA__{row['class_name']}__{source.stem}"
        image_path = images_dir / f'{sample_id}.png'
        mask_path = masks_dir / f'{sample_id}.png'
        weight_path = weights_dir / f'{sample_id}.png'
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f'Failed to write {image_path}')
        # Online strong-teacher distillation uses only shape/validity here.
        cv2.imwrite(str(mask_path), np.zeros((512, 512), dtype=np.uint8))
        cv2.imwrite(
            str(weight_path), np.full((512, 512), 65535, dtype=np.uint16))
        adaptation_rows.append({
            'sample_id': sample_id,
            'sample_type': 'external_unlabeled_adaptation',
            'split': 'adaptation_train_only',
            'class_name': row['class_name'],
            'scene_id': row['scene_id'],
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'weight_path': str(weight_path),
            'source_image_path': str(source),
            'sha256': actual_hash,
        })

    counts = Counter(row['class_name'] for row in adaptation_rows)
    if len(adaptation_rows) != 80:
        raise RuntimeError(
            f'Expected 80 complementary images, got {len(adaptation_rows)}')
    if any(counts[name] != 20 for name in CATEGORY_ORDER):
        raise RuntimeError(f'Unbalanced complementary set: {dict(counts)}')
    if {row['sha256'] for row in adaptation_rows} & frozen_hashes:
        raise RuntimeError('Frozen test image leaked into adaptation set')

    adaptation_rows.sort(
        key=lambda item: (CATEGORY_ORDER.index(item['class_name']),
                          item['sample_id']))
    manifest_path = output_root / 'pseudo_train_manifest.csv'
    write_csv(manifest_path, adaptation_rows)
    frozen_scenes = {row['scene_id'] for row in frozen}
    adaptation_scenes = {row['scene_id'] for row in adaptation_rows}
    summary = {
        'source_selected_images': len(selected),
        'frozen_annotated_test_images_excluded': len(frozen),
        'unlabeled_adaptation_images': len(adaptation_rows),
        'images_per_category': dict(counts),
        'image_size': [512, 512],
        'external_ground_truth_masks_read': False,
        'adaptation_test_sha256_overlap': 0,
        'adaptation_scenes': len(adaptation_scenes),
        'frozen_test_scenes': len(frozen_scenes),
        'scene_id_overlap_count': len(adaptation_scenes & frozen_scenes),
        'selection_manifest': str(selection_manifest),
        'selection_manifest_sha256': sha256(selection_manifest),
        'frozen_test_manifest': str(frozen_test_manifest),
        'frozen_test_manifest_sha256': sha256(frozen_test_manifest),
        'adaptation_manifest': str(manifest_path),
        'adaptation_manifest_sha256': sha256(manifest_path),
        'protocol': (
            'label-free target-domain adaptation; the frozen 80 masks are '
            'excluded and are not read'),
    }
    (output_root / 'dataset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

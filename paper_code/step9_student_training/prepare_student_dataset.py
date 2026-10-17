#!/usr/bin/env python
"""Prepare aligned 512x512 supervised and pseudo-labelled student data."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervised-root', default='data/dadou_mmseg')
    parser.add_argument(
        '--gating-root', default='data/dadou/06_07_08_双证据门控')
    parser.add_argument('--output-root', default='data/dadou_student')
    parser.add_argument('--image-size', type=int, default=512)
    return parser.parse_args()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_supervised_split(source_root, output_root, split, image_size):
    image_dir = Path(source_root) / split / 'images'
    mask_dir = Path(source_root) / split / 'masks'
    output_image_dir = Path(output_root) / 'supervised' / split / 'images'
    output_mask_dir = Path(output_root) / 'supervised' / split / 'masks'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pixel_counts = Counter()
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = mask_dir / f'{image_path.stem}.png'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f'Unreadable pair: {image_path}, {mask_path}')
        if set(np.unique(mask).tolist()) - {0, 1, 2}:
            raise ValueError(f'Unexpected labels in {mask_path}')
        image = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        output_image = output_image_dir / f'{image_path.stem}.jpg'
        output_mask = output_mask_dir / f'{image_path.stem}.png'
        cv2.imwrite(str(output_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(output_mask), mask.astype(np.uint8))
        for label in (0, 1, 2):
            pixel_counts[label] += int((mask == label).sum())
        rows.append({
            'sample_id': image_path.stem,
            'sample_type': 'supervised',
            'split': split,
            'image_path': str(output_image.resolve()),
            'mask_path': str(output_mask.resolve()),
            'weight_path': '',
            'source_image_path': str(image_path.resolve()),
        })
    return rows, pixel_counts


def prepare_pseudo_split(gating_root, output_root, image_size):
    gating_root = Path(gating_root)
    accepted_manifest = gating_root / '可用于学生训练.csv'
    with accepted_manifest.open(encoding='utf-8-sig', newline='') as handle:
        accepted_rows = list(csv.DictReader(handle))
    output_image_dir = Path(output_root) / 'pseudo' / 'images'
    output_mask_dir = Path(output_root) / 'pseudo' / 'masks'
    output_weight_dir = Path(output_root) / 'pseudo' / 'weights'
    for directory in (output_image_dir, output_mask_dir, output_weight_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    pixel_counts = Counter()
    for row in accepted_rows:
        source_image = Path(row['source_path'])
        source_mask = gating_root / row['pseudo_label_path']
        source_weight = gating_root / row['pixel_weight_path']
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
        weight = cv2.imread(str(source_weight), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or weight is None:
            raise ValueError(
                f'Unreadable pseudo sample: {source_image}, {source_mask}, '
                f'{source_weight}')
        image = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if mask.shape != (image_size, image_size):
            mask = cv2.resize(
                mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        if weight.shape != (image_size, image_size):
            weight = cv2.resize(
                weight, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        unique = set(np.unique(mask).tolist())
        if unique - {0, 1, 2, 255}:
            raise ValueError(f'Unexpected pseudo labels in {source_mask}: {unique}')
        sample_id = f"{row['class_name']}__{source_image.stem}"
        output_image = output_image_dir / f'{sample_id}.jpg'
        output_mask = output_mask_dir / f'{sample_id}.png'
        output_weight = output_weight_dir / f'{sample_id}.png'
        cv2.imwrite(str(output_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(output_mask), mask.astype(np.uint8))
        cv2.imwrite(str(output_weight), weight.astype(np.uint16))
        for label in (0, 1, 2, 255):
            pixel_counts[label] += int((mask == label).sum())
        rows.append({
            'sample_id': sample_id,
            'sample_type': 'pseudo',
            'split': 'train',
            'class_name': row['class_name'],
            'image_path': str(output_image.resolve()),
            'mask_path': str(output_mask.resolve()),
            'weight_path': str(output_weight.resolve()),
            'source_image_path': str(source_image.resolve()),
        })
    return rows, pixel_counts


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        'image_size': [args.image_size, args.image_size],
        'supervised': {},
        'pseudo': {},
        'test_split_used_for_training_or_model_selection': False,
    }
    for split in ('train', 'val', 'test'):
        rows, pixels = prepare_supervised_split(
            args.supervised_root, output_root, split, args.image_size)
        manifest_path = output_root / f'supervised_{split}_manifest.csv'
        write_csv(manifest_path, rows)
        summary['supervised'][split] = {
            'images': len(rows),
            'manifest': str(manifest_path.resolve()),
            'pixel_counts': {str(key): value for key, value in pixels.items()},
        }
    pseudo_rows, pseudo_pixels = prepare_pseudo_split(
        args.gating_root, output_root, args.image_size)
    pseudo_manifest = output_root / 'pseudo_train_manifest.csv'
    write_csv(pseudo_manifest, pseudo_rows)
    summary['pseudo'] = {
        'images': len(pseudo_rows),
        'manifest': str(pseudo_manifest.resolve()),
        'pixel_counts': {
            str(key): value for key, value in pseudo_pixels.items()
        },
        'weight_encoding': 'weight = uint16_pixel / 65535.0',
    }
    (output_root / 'dataset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

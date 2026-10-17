#!/usr/bin/env python
"""Convert LabelMe soybean polygons into mmsegmentation masks.

Original images and JSON annotations are never modified. Lesions are clipped
to the union of leaf polygons, enforcing the three-class label contract:
0 background, 1 visible healthy leaf tissue, 2 lesion.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SPLIT_FOLDERS = {
    'train': '01_训练集_其余127张',
    'val': '02_验证集_42张',
    'test': '03_测试集_49张_请最后标',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='data/dadou')
    parser.add_argument('--output-root', default='data/dadou_mmseg')
    return parser.parse_args()


def rasterize(shapes, height, width):
    leaf = np.zeros((height, width), dtype=np.uint8)
    lesion = np.zeros_like(leaf)
    counts = Counter()
    for shape in shapes:
        label = shape.get('label')
        if label not in ('leaf', 'lesion'):
            continue
        points = shape.get('points', [])
        if len(points) < 3:
            continue
        polygon = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(leaf if label == 'leaf' else lesion, [polygon], 1)
        counts[label] += 1

    outside_pixels = int(np.count_nonzero(lesion & (leaf == 0)))
    clipped_lesion = lesion & leaf
    mask = np.zeros_like(leaf)
    mask[leaf > 0] = 1
    mask[clipped_lesion > 0] = 2
    return mask, counts, outside_pixels


def safe_symlink(source, destination):
    relative_source = os.path.relpath(source, destination.parent)
    if destination.is_symlink():
        if os.readlink(destination) == relative_source:
            return
        raise RuntimeError(f'Existing symlink has a different target: {destination}')
    if destination.exists():
        raise RuntimeError(f'Refusing to overwrite existing file: {destination}')
    destination.symlink_to(relative_source)


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    split_stats = {}
    train_pixel_counts = np.zeros(3, dtype=np.int64)

    for split, folder_name in SPLIT_FOLDERS.items():
        source_dir = source_root / folder_name
        image_dir = source_dir / 'images'
        annotation_dir = source_dir / 'annotations_json'
        output_images = output_root / split / 'images'
        output_masks = output_root / split / 'masks'
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)

        included = excluded = 0
        for json_path in sorted(annotation_dir.glob('*.json')):
            with json_path.open(encoding='utf-8') as handle:
                annotation = json.load(handle)

            image_path = image_dir / annotation.get(
                'imagePath', json_path.with_suffix('.jpg').name)
            if not image_path.is_file():
                candidates = list(image_dir.glob(f'{json_path.stem}.*'))
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        f'Cannot uniquely find image for {json_path}')
                image_path = candidates[0]

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Cannot read image: {image_path}')
            height, width = image.shape[:2]
            mask, shape_counts, outside_pixels = rasterize(
                annotation.get('shapes', []), height, width)

            status = 'included'
            if shape_counts['leaf'] == 0 or not np.any(mask == 1):
                status = 'excluded_no_leaf'
                excluded += 1
            else:
                mask_path = output_masks / f'{json_path.stem}.png'
                if not cv2.imwrite(str(mask_path), mask):
                    raise RuntimeError(f'Cannot write mask: {mask_path}')
                safe_symlink(image_path, output_images / image_path.name)
                included += 1
                if split == 'train':
                    train_pixel_counts += np.bincount(
                        mask.reshape(-1), minlength=3).astype(np.int64)

            class_pixels = np.bincount(mask.reshape(-1), minlength=3)
            records.append({
                'split': split,
                'sample_id': json_path.stem,
                'image_path': str(image_path),
                'annotation_path': str(json_path),
                'width': width,
                'height': height,
                'leaf_polygons': shape_counts['leaf'],
                'lesion_polygons': shape_counts['lesion'],
                'lesion_outside_leaf_pixels_clipped': outside_pixels,
                'background_pixels': int(class_pixels[0]),
                'healthy_leaf_pixels': int(class_pixels[1]),
                'lesion_pixels': int(class_pixels[2]),
                'status': status,
            })
        split_stats[split] = {'included': included, 'excluded': excluded}

    manifest_path = output_root / 'conversion_manifest.csv'
    with manifest_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    frequencies = train_pixel_counts / train_pixel_counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-12))
    weights /= weights.mean()
    stats = {
        'label_contract': {
            '0': 'background',
            '1': 'healthy_leaf',
            '2': 'lesion',
        },
        'split_samples': split_stats,
        'train_pixel_counts': train_pixel_counts.tolist(),
        'train_pixel_frequencies': frequencies.tolist(),
        'suggested_inverse_sqrt_class_weights_mean_1': weights.tolist(),
        'lesion_policy': 'lesion polygons are intersected with the union of leaf polygons',
        'source_is_unchanged': True,
    }
    with (output_root / 'dataset_statistics.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()

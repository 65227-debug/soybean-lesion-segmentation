#!/usr/bin/env python
"""Convert the frozen 80-image LabelMe subset to 3-class 512px masks."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASSES = ('background', 'healthy_leaf', 'lesion')
PALETTE_BGR = np.asarray(((0, 0, 0), (0, 180, 0), (30, 30, 220)),
                         dtype=np.uint8)
EXPECTED_CATEGORIES = {
    'Healthy', 'Frogeye_leaf_spot', 'Rust', 'Septoria_brown_spot'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotated-root', required=True)
    parser.add_argument('--selection-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--size', type=int, default=512)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def polygon_mask(shapes, label, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        if shape.get('label') != label:
            continue
        if shape.get('shape_type') != 'polygon':
            raise ValueError(f'Only polygons are supported: {shape!r}')
        points = np.rint(np.asarray(shape['points'], dtype=np.float64))
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)


def main():
    args = parse_args()
    annotated_root = Path(args.annotated_root).resolve()
    selection_manifest = Path(args.selection_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output is non-empty: {output_dir}')
    if args.size <= 0:
        raise ValueError('--size must be positive')

    with selection_manifest.open(encoding='utf-8-sig', newline='') as handle:
        selected_rows = {
            row['output_name']: row for row in csv.DictReader(handle)
            if row.get('selected', '').lower() == 'true'
        }
    image_root = annotated_root / 'images'
    pairs = []
    for class_dir in sorted(path for path in image_root.iterdir()
                            if path.is_dir()):
        for image_path in sorted(class_dir.glob('*.jpg')):
            json_path = image_path.with_suffix('.json')
            if not json_path.is_file():
                raise FileNotFoundError(f'Missing JSON: {json_path}')
            pairs.append((class_dir.name, image_path, json_path))
    category_counts = Counter(category for category, _, _ in pairs)
    if set(category_counts) != EXPECTED_CATEGORIES:
        raise RuntimeError(f'Unexpected categories: {dict(category_counts)}')
    if category_counts != Counter({name: 20 for name in EXPECTED_CATEGORIES}):
        raise RuntimeError(f'Expected 20 per category: {dict(category_counts)}')

    image_dir = output_dir / 'images_512'
    mask_dir = output_dir / 'masks_512'
    colour_dir = output_dir / 'ground_truth_colour'
    overlay_dir = output_dir / 'ground_truth_overlay'
    json_dir = output_dir / 'labelme_json_original'
    for directory in (image_dir, mask_dir, colour_dir, overlay_dir, json_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    pixels = Counter()
    total_outside = 0
    total_lesion = 0
    for index, (category, image_path, json_path) in enumerate(pairs, start=1):
        data = json.loads(json_path.read_text(encoding='utf-8'))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable image: {image_path}')
        height, width = image.shape[:2]
        if (int(data.get('imageHeight', -1)), int(data.get('imageWidth', -1))) \
                != (height, width):
            raise RuntimeError(f'JSON dimensions mismatch: {json_path}')
        shapes = data.get('shapes', [])
        leaf = polygon_mask(shapes, 'leaf', height, width)
        lesion = polygon_mask(shapes, 'lesion', height, width)
        if not leaf.any():
            raise RuntimeError(f'No leaf pixels: {json_path}')
        if category == 'Healthy' and lesion.any():
            raise RuntimeError(f'Healthy image has lesion pixels: {json_path}')
        if category != 'Healthy' and not lesion.any():
            raise RuntimeError(f'Disease image has no lesion pixels: {json_path}')

        outside = lesion & ~leaf
        total_outside += int(outside.sum())
        total_lesion += int(lesion.sum())
        target = np.zeros((height, width), dtype=np.uint8)
        target[leaf] = 1
        target[lesion] = 2
        resized_image = cv2.resize(
            image, (args.size, args.size), interpolation=cv2.INTER_AREA)
        resized_target = cv2.resize(
            target, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        values = set(np.unique(resized_target).tolist())
        if not values <= {0, 1, 2}:
            raise RuntimeError(f'Invalid class values {values}: {json_path}')

        sample_id = f'{category}__{image_path.stem}'
        image_out = image_dir / f'{sample_id}.png'
        mask_out = mask_dir / f'{sample_id}.png'
        colour_out = colour_dir / f'{sample_id}.png'
        overlay_out = overlay_dir / f'{sample_id}.jpg'
        if not cv2.imwrite(str(image_out), resized_image):
            raise OSError(image_out)
        if not cv2.imwrite(str(mask_out), resized_target):
            raise OSError(mask_out)
        colour = PALETTE_BGR[resized_target]
        cv2.imwrite(str(colour_out), colour)
        overlay = cv2.addWeighted(resized_image, 0.65, colour, 0.35, 0.0)
        cv2.imwrite(str(overlay_out), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        shutil.copy2(json_path, json_dir / f'{sample_id}.json')

        manifest_row = selected_rows.get(image_path.name)
        if manifest_row is None:
            raise RuntimeError(
                f'Annotated image absent from frozen 160 selection: {image_path}')
        current_hash = sha256(image_path)
        if current_hash != manifest_row['sha256']:
            raise RuntimeError(
                f'Image hash differs from frozen selection: {image_path}')
        counts = {value: int(np.sum(resized_target == value))
                  for value in range(3)}
        pixels.update(counts)
        rows.append({
            'sample_id': sample_id,
            'sample_type': 'external_test',
            'split': 'external_test_only',
            'class_name': category,
            'scene_id': manifest_row['scene_id'],
            'image_path': str(image_out),
            'mask_path': str(mask_out),
            'source_image_path': str(image_path),
            'source_json_path': str(json_path),
            'sha256': current_hash,
            'original_width': width,
            'original_height': height,
            'background_percent': 100.0 * counts[0] / resized_target.size,
            'healthy_leaf_percent': 100.0 * counts[1] / resized_target.size,
            'lesion_percent': 100.0 * counts[2] / resized_target.size,
            'lesion_outside_leaf_original_pixels': int(outside.sum()),
            'lesion_original_pixels': int(lesion.sum()),
        })

    write_csv(output_dir / 'external_test_manifest.csv', rows)
    total_pixels = sum(pixels.values())
    summary = {
        'dataset': 'MH-SoyaHealthVision balanced external test subset',
        'source_dataset_candidate_pool': 160,
        'frozen_external_test_images': len(rows),
        'images_per_category': dict(sorted(category_counts.items())),
        'classes': list(CLASSES),
        'class_values': {'background': 0, 'healthy_leaf': 1, 'lesion': 2},
        'output_size': [args.size, args.size],
        'pixel_distribution_percent': {
            CLASSES[value]: 100.0 * pixels[value] / total_pixels
            for value in range(3)},
        'lesion_pixels_outside_leaf_annotation_percent': (
            100.0 * total_outside / total_lesion if total_lesion else 0.0),
        'all_images_match_original_frozen_selection_sha256': True,
        'training_or_model_selection_use': False,
        'conversion_rule': (
            'rasterize leaf=1 first; rasterize lesion=2 second; '
            'unlabelled pixels=background=0; resize masks with nearest-neighbor'),
    }
    (output_dir / 'dataset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

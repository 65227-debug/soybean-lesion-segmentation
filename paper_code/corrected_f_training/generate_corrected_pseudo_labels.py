#!/usr/bin/env python
"""Generate corrected, scene-auditable pseudo labels for experiment F."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from corrected_gating import IGNORE, gate_pixels, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-maps', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    base_dir = Path(args.base_maps)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'
    weight_dir = output_dir / 'weights'
    for directory in (image_dir, mask_dir, weight_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    rows = read_csv(base_dir / '三张基础图生成清单.csv')
    manifest, rejected = [], []
    pixels = Counter()
    scenes = Counter()
    classes = Counter()
    for index, row in enumerate(rows, start=1):
        arrays = np.load(base_dir / row['numeric_npz'])
        labels, weights, masks = gate_pixels(
            arrays['p_teacher'].astype(np.float32),
            arrays['p_leaf'].astype(np.float32),
            arrays['a_sem'].astype(np.float32), config)
        reasons = []
        if 100.0 * masks['inner'].mean() < config.min_leaf_inner_area_percent:
            reasons.append('leaf_area_too_small')
        if int(masks['lesion'].sum()) < config.min_positive_pixels:
            reasons.append('too_few_lesion_pixels')
        confidence = float(row.get('semantic_label_probability') or 1.0)
        if confidence < config.min_semantic_confidence:
            reasons.append('semantic_image_confidence_too_low')
        if reasons:
            rejected.append({
                'filename': row['filename'], 'class_name': row['class_name'],
                'scene_id': row.get('scene_id', ''),
                'reasons': '|'.join(reasons)})
            continue
        sample_id = f"{row['class_name']}__{Path(row['filename']).stem}"
        image_path = image_dir / f'{sample_id}.jpg'
        mask_path = mask_dir / f'{sample_id}.png'
        weight_path = weight_dir / f'{sample_id}.png'
        image = cv2.imread(row['source_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {row['source_path']}")
        image = cv2.resize(image, (labels.shape[1], labels.shape[0]),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), labels)
        cv2.imwrite(
            str(weight_path),
            np.rint(np.clip(weights, 0, 1) * 65535).astype(np.uint16))
        scene_id = row.get('scene_id', '') or sample_id
        manifest.append({
            'sample_id': sample_id, 'sample_type': 'pseudo', 'split': 'train',
            'class_name': row['class_name'], 'scene_id': scene_id,
            'image_path': str(image_path.resolve()),
            'mask_path': str(mask_path.resolve()),
            'weight_path': str(weight_path.resolve()),
            'source_image_path': row['source_path'],
        })
        scenes[scene_id] += 1
        classes[row['class_name']] += 1
        for value in (0, 1, 2, 255):
            pixels[value] += int((labels == value).sum())
        if index % 100 == 0 or index == len(rows):
            print(f'[{index}/{len(rows)}] accepted={len(manifest)}', flush=True)
    if not manifest:
        raise RuntimeError('No corrected pseudo labels were accepted')
    write_csv(output_dir / 'pseudo_train_manifest.csv', manifest)
    if rejected:
        write_csv(output_dir / 'rejected_manifest.csv', rejected)
    total = sum(pixels.values())
    summary = {
        'input_images': len(rows), 'accepted_images': len(manifest),
        'rejected_images': len(rejected), 'accepted_by_class': dict(classes),
        'accepted_unique_scenes': len(scenes),
        'maximum_images_from_one_scene': max(scenes.values()),
        'pixel_distribution_percent': {
            str(label): 100.0 * pixels[label] / total
            for label in (0, 1, 2, 255)},
        'scene_balancing': 'performed by inverse-scene-frequency sampler',
        'test_split_used': False,
    }
    (output_dir / 'pseudo_label_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


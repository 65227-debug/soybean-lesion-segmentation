#!/usr/bin/env python
"""Build a scene-disjoint soybean disease weak-label classification set."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
CLASSES = ('frogeye', 'target_spot')
PROTECTED_ROLES = {'exact_train', 'exact_val', 'exact_test'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='data/新建文件夹')
    parser.add_argument(
        '--metadata', default='data/dadou/metadata/ASDID弱监督池.csv')
    parser.add_argument('--supervised-root', default='data/dadou')
    parser.add_argument('--output-root', default='data/dadou_semantic')
    parser.add_argument(
        '--near-duplicate-distance',
        type=int,
        default=2,
        help='Maximum 64-bit pHash Hamming distance to a supervised image.')
    parser.add_argument(
        '--verify-sha256', action='store_true', help='Verify selected files.')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8]
    median = np.median(coefficients.reshape(-1)[1:])
    bits = coefficients > median
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming(left, right):
    return bin(left ^ right).count('1')


def image_files(root):
    return sorted(
        path for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def ensure_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f'Conflicting output: {destination}')
        return
    os.symlink(source.resolve(), destination)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    metadata_path = Path(args.metadata)
    supervised_root = Path(args.supervised_root)
    output_root = Path(args.output_root)

    with metadata_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))

    protected = [row for row in rows if row['role'] in PROTECTED_ROLES]
    protected_scenes = {row['scene_id'] for row in protected}
    protected_hashes = {row['sha256'] for row in protected}

    supervised_paths = []
    for split_dir in sorted(supervised_root.glob('*')):
        candidate = split_dir / 'images'
        if candidate.is_dir():
            supervised_paths.extend(image_files(candidate))
    supervised_actual_hashes = {sha256(path) for path in supervised_paths}
    supervised_phashes = [(path, perceptual_hash(path)) for path in supervised_paths]
    protected_hashes.update(supervised_actual_hashes)

    candidates = sorted(
        (row for row in rows if row['role'] == 'weak_train_candidate'),
        key=lambda row: (row['category_label'], row['scene_id'], row['filename']))
    selected = []
    excluded = []
    seen_hashes = set()
    for row in candidates:
        record = dict(row)
        source = source_root / row['category_label'] / row['filename']
        reason = ''
        nearest_distance = ''
        if row['category_label'] not in CLASSES:
            reason = 'unsupported_class'
        elif not source.is_file():
            reason = 'missing_source_file'
        elif row['scene_id'] in protected_scenes:
            reason = 'same_scene_as_supervised'
        elif row['sha256'] in protected_hashes:
            reason = 'exact_duplicate_of_supervised'
        elif row['sha256'] in seen_hashes:
            reason = 'internal_exact_duplicate'
        else:
            candidate_phash = perceptual_hash(source)
            if supervised_phashes:
                nearest_path, nearest_hash = min(
                    supervised_phashes,
                    key=lambda item: hamming(candidate_phash, item[1]))
                nearest_distance = hamming(candidate_phash, nearest_hash)
                record['nearest_supervised_image'] = str(nearest_path.resolve())
                record['nearest_supervised_phash_distance'] = nearest_distance
                if nearest_distance <= args.near_duplicate_distance:
                    reason = 'perceptual_duplicate_of_supervised'
            if not reason and args.verify_sha256:
                actual_hash = sha256(source)
                if actual_hash != row['sha256']:
                    reason = 'sha256_metadata_mismatch'

        record['source_local'] = str(source.resolve()) if source.exists() else str(source)
        record['exclusion_reason'] = reason
        if reason:
            excluded.append(record)
            continue
        seen_hashes.add(row['sha256'])
        class_index = CLASSES.index(row['category_label'])
        destination = output_root / 'train' / row['category_label'] / row['filename']
        ensure_link(source, destination)
        record['class_name'] = row['category_label']
        record['class_index'] = class_index
        record['linked_path'] = str(destination.resolve())
        selected.append(record)

    validation = []
    for row in sorted(
            (item for item in rows if item['role'] == 'exact_val'),
            key=lambda item: (item['category_label'], item['filename'])):
        source = source_root / row['category_label'] / row['filename']
        if not source.is_file():
            raise FileNotFoundError(source)
        if args.verify_sha256 and sha256(source) != row['sha256']:
            raise ValueError(f'SHA-256 mismatch: {source}')
        destination = output_root / 'val' / row['category_label'] / row['filename']
        ensure_link(source, destination)
        record = dict(row)
        record['class_name'] = row['category_label']
        record['class_index'] = CLASSES.index(row['category_label'])
        record['source_local'] = str(source.resolve())
        record['linked_path'] = str(destination.resolve())
        validation.append(record)

    train_fields = list(selected[0])
    excluded_fields = sorted({key for row in excluded for key in row})
    val_fields = list(validation[0])
    write_csv(output_root / 'train_manifest.csv', selected, train_fields)
    write_csv(output_root / 'val_manifest.csv', validation, val_fields)
    write_csv(output_root / 'excluded_manifest.csv', excluded, excluded_fields)

    test_rows = [dict(row) for row in rows if row['role'] == 'exact_test']
    write_csv(
        output_root / 'held_out_test_manifest_do_not_use.csv', test_rows,
        list(test_rows[0]))

    summary = {
        'classes': list(CLASSES),
        'candidate_rows': len(candidates),
        'selected_weak_train': len(selected),
        'selected_by_class': dict(Counter(row['class_name'] for row in selected)),
        'selected_unique_scenes': len({row['scene_id'] for row in selected}),
        'validation_images': len(validation),
        'validation_by_class': dict(Counter(row['class_name'] for row in validation)),
        'excluded_candidates': len(excluded),
        'excluded_by_reason': dict(Counter(row['exclusion_reason'] for row in excluded)),
        'protected_supervised_images_on_disk': len(supervised_paths),
        'protected_metadata_rows': len(protected),
        'near_duplicate_phash_distance': args.near_duplicate_distance,
        'sha256_verified': args.verify_sha256,
        'test_split_used': False,
    }
    with (output_root / 'audit_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

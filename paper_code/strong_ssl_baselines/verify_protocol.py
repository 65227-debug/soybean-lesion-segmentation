#!/usr/bin/env python
"""Audit the frozen SSL protocol without loading weak labels or pseudo masks."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path):
    with resolve(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_path(row):
    return resolve(row['image_path'])


def source_path(row):
    value = row.get('source_image_path') or row['image_path']
    return resolve(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', default=str(Path(__file__).with_name('aligned_protocol.json')))
    parser.add_argument(
        '--output', default='paper_materials/强半监督基线复现_20260827/00_协议审计.json')
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text(encoding='utf-8'))
    manifests = {
        'labeled_train': cfg['labeled_train_manifest'],
        'validation': cfg['validation_manifest'],
        'test': cfg['test_manifest'],
        'unlabeled': cfg['unlabeled_manifest'],
    }
    rows = {name: read_rows(path) for name, path in manifests.items()}
    expected = {'labeled_train': 108, 'validation': 27, 'test': 37, 'unlabeled': 1390}
    problems = []
    details = {}
    split_hashes = {}
    for name, split_rows in rows.items():
        ids = [row['sample_id'] for row in split_rows]
        paths = [image_path(row) for row in split_rows]
        sources = [source_path(row) for row in split_rows]
        missing = [str(path) for path in paths + sources if not path.is_file()]
        if len(split_rows) != expected[name]:
            problems.append(f'{name}: expected {expected[name]}, got {len(split_rows)}')
        if len(ids) != len(set(ids)):
            problems.append(f'{name}: duplicate sample_id')
        if missing:
            problems.append(f'{name}: {len(missing)} missing paths')
        dimensions = Counter()
        unreadable = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable.append(str(path))
            else:
                dimensions[f'{image.shape[1]}x{image.shape[0]}'] += 1
        hashes = {sha256(path): str(path) for path in sources if path.is_file()}
        split_hashes[name] = set(hashes)
        details[name] = {
            'manifest': str(resolve(manifests[name])),
            'rows': len(split_rows),
            'unique_sample_ids': len(set(ids)),
            'unique_source_sha256': len(hashes),
            'image_dimensions': dict(dimensions),
            'missing_paths': missing,
            'unreadable_paths': unreadable,
        }
        if unreadable:
            problems.append(f'{name}: {len(unreadable)} unreadable images')
    names = list(rows)
    overlaps = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            common = split_hashes[left] & split_hashes[right]
            overlaps.append({'left': left, 'right': right, 'sha256_overlap': len(common)})
            if common:
                problems.append(f'{left}/{right}: {len(common)} byte-identical source images')
    unlabel_columns = set(rows['unlabeled'][0]) if rows['unlabeled'] else set()
    allowed = set(cfg['unlabeled_columns_allowed'])
    used_columns = {'sample_id', 'image_path', 'source_image_path'}
    if not used_columns <= unlabel_columns:
        problems.append('unlabeled manifest misses one of the required image-only columns')
    result = {
        'protocol_name': cfg['protocol_name'],
        'status': 'PASS' if not problems else 'FAIL',
        'details': details,
        'content_hash_overlaps': overlaps,
        'unlabeled_manifest_columns_present_but_not_loaded_by_training': sorted(unlabel_columns - allowed),
        'unlabeled_columns_used_by_training': sorted(used_columns),
        'total_optimizer_steps': cfg['total_optimizer_steps'],
        'problems': problems,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Audit the MH-SoyaHealthVision 160-image LabelMe annotation set."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotated-root', required=True)
    parser.add_argument('--original-root', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    annotated_root = Path(args.annotated_root).resolve()
    original_root = Path(args.original_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = original_root / 'metadata' / 'selection_manifest.csv'
    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        selected = [
            row for row in csv.DictReader(handle)
            if row.get('selected', '').lower() == 'true'
        ]
    if len(selected) != 160:
        raise RuntimeError(f'Expected 160 selected rows, found {len(selected)}')

    rows = []
    invalid_annotations = []
    label_counts = Counter()
    class_counts = Counter()
    status_counts = Counter()
    for row in selected:
        class_name = row['class_name']
        filename = row['output_name']
        stem = Path(filename).stem
        original_image = original_root / 'images' / class_name / filename
        annotated_image = annotated_root / 'images' / class_name / filename
        annotation_json = annotated_root / 'images' / class_name / f'{stem}.json'
        issues = []
        labels = Counter()
        shape_count = 0
        dimensions_match = False
        image_hash_matches = False

        if not original_image.is_file():
            issues.append('original_source_image_missing')
        if not annotated_image.is_file():
            issues.append('annotated_image_missing')
        if not annotation_json.is_file():
            issues.append('annotation_json_missing')

        if original_image.is_file() and annotated_image.is_file():
            image_hash_matches = sha256(original_image) == sha256(annotated_image)
            if not image_hash_matches:
                issues.append('annotated_image_differs_from_original')

        if annotation_json.is_file():
            try:
                data = json.loads(annotation_json.read_text(encoding='utf-8'))
                image = cv2.imread(str(annotated_image), cv2.IMREAD_COLOR)
                if image is None:
                    issues.append('annotated_image_unreadable')
                else:
                    height, width = image.shape[:2]
                    dimensions_match = (
                        int(data.get('imageHeight', -1)) == height
                        and int(data.get('imageWidth', -1)) == width)
                    if not dimensions_match:
                        issues.append('json_image_dimensions_mismatch')
                shapes = data.get('shapes', [])
                shape_count = len(shapes)
                if not shapes:
                    issues.append('annotation_has_no_shapes')
                for shape in shapes:
                    label = shape.get('label', '')
                    labels[label] += 1
                    label_counts[label] += 1
                    if label not in {'leaf', 'lesion'}:
                        issues.append(f'unexpected_label:{label}')
                    if shape.get('shape_type') != 'polygon':
                        issues.append(
                            f"unexpected_shape_type:{shape.get('shape_type')}")
                    points = shape.get('points', [])
                    if len(points) < 3:
                        issues.append('polygon_has_fewer_than_3_points')
                if class_name == 'Healthy' and labels.get('lesion', 0):
                    issues.append('healthy_image_contains_lesion_polygon')
                if not labels.get('leaf', 0):
                    issues.append('leaf_polygon_missing')
            except Exception as error:
                issues.append(f'json_parse_error:{error!r}')

        if annotation_json.is_file() and annotated_image.is_file() and not issues:
            status = 'valid_annotated_pair'
            class_counts[class_name] += 1
        else:
            status = 'needs_annotation_or_repair'
        status_counts[status] += 1
        output = {
            'class_name': class_name,
            'filename': filename,
            'scene_id': row.get('scene_id', ''),
            'status': status,
            'issues': '|'.join(dict.fromkeys(issues)),
            'original_image_path': str(original_image),
            'annotated_image_path': str(annotated_image),
            'annotation_json_path': str(annotation_json),
            'original_image_exists': original_image.is_file(),
            'annotated_image_exists': annotated_image.is_file(),
            'annotation_json_exists': annotation_json.is_file(),
            'image_hash_matches_original': image_hash_matches,
            'json_image_dimensions_match': dimensions_match,
            'shape_count': shape_count,
            'leaf_polygon_count': labels.get('leaf', 0),
            'lesion_polygon_count': labels.get('lesion', 0),
        }
        rows.append(output)
        if status != 'valid_annotated_pair':
            invalid_annotations.append(output)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in (
        ('全部160张标注核查.csv', rows),
        ('待补标或修复清单.csv', invalid_annotations),
    ):
        with (output_dir / name).open(
                'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(contents)

    summary = {
        'expected_selected_images': 160,
        'valid_annotated_pairs': status_counts['valid_annotated_pair'],
        'needs_annotation_or_repair': status_counts[
            'needs_annotation_or_repair'],
        'valid_by_class': dict(sorted(class_counts.items())),
        'annotation_shape_label_counts': dict(sorted(label_counts.items())),
        'annotated_root': str(annotated_root),
        'original_complete_160_root': str(original_root),
        'safe_to_run_final_frozen_external_test': (
            status_counts['valid_annotated_pair'] == 160),
        'model_inference_performed': False,
        'reason_not_evaluated': (
            'External test is incomplete; partial evaluation would expose test '
            'performance before the planned one-time 160-image evaluation.'
            if status_counts['valid_annotated_pair'] != 160 else ''),
    }
    (output_dir / '标注核查汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

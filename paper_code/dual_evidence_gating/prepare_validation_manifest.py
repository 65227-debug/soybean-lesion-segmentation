#!/usr/bin/env python
"""Prepare a validation manifest without touching the final test split."""

import argparse
import csv
from pathlib import Path


CLASS_MAP = {'FLS': 'frogeye', 'TLS': 'target_spot'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', default='data/dadou_mmseg/val/images')
    parser.add_argument('--mask-dir', default='data/dadou_mmseg/val/masks')
    parser.add_argument(
        '--master-manifest', default='data/dadou/metadata/精标总清单.csv')
    parser.add_argument(
        '--output', default='data/dadou/06_验证集基础图/validation_manifest.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    scene_by_filename = {}
    master_path = Path(args.master_manifest)
    if master_path.is_file():
        with master_path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                scene_by_filename[Path(row['annotation_filename']).stem] = row.get(
                    'scene_id', '')
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    rows = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        disease_code = image_path.stem.rsplit('_', 1)[-1]
        if disease_code not in CLASS_MAP:
            continue
        mask_path = mask_dir / f'{image_path.stem}.png'
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        rows.append({
            'filename': image_path.name,
            'class_name': CLASS_MAP[disease_code],
            'scene_id': scene_by_filename.get(image_path.stem, ''),
            'linked_path': str(image_path.resolve()),
            'mask_path': str(mask_path.resolve()),
        })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        name: sum(row['class_name'] == name for row in rows)
        for name in CLASS_MAP.values()
    }
    print(f'Wrote {len(rows)} validation rows to {output_path.resolve()}')
    print(counts)


if __name__ == '__main__':
    main()

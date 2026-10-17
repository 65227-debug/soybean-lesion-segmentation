#!/usr/bin/env python
"""Export aligned originals and publication-ready base-map visualizations.

The formal fifth-step NPZ files retain float32 Pseg, Asem, and Minner data.
This utility converts Pseg and Asem to true-colour JET heatmaps with a fixed
0--1 colour scale, renders Minner as a green binary region, and exports both
separate panels and a labelled four-panel comparison for every sample.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--base-dir',
        default='data/dadou/05_三张基础图_修正版',
        help='Formal base-map result directory containing manifest and NPZs.')
    parser.add_argument(
        '--output-dir',
        default='data/dadou/07_论文四图_原图_Pseg_Asem_Minner',
        help='Destination directory.')
    parser.add_argument('--size', type=int, default=512)
    return parser.parse_args()


def probability_heatmap(array):
    """Map [0, 1] values to a three-channel BGR JET heatmap."""
    values = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(values, cv2.COLORMAP_JET)


def minner_colour(mask):
    """Render trustworthy leaf interior as green on a black background."""
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask.astype(bool)] = (0, 190, 0)
    return result


def labelled_panel(image, label):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (148, 38), (255, 255, 255), -1)
    cv2.putText(
        result, label, (9, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (20, 20, 20), 2, cv2.LINE_AA)
    return result


def write_image(path, image, jpeg_quality=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if jpeg_quality is not None:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f'Failed to write image: {path}')


def main():
    args = parse_args()
    if args.size <= 0:
        raise ValueError('--size must be positive')

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    manifest_path = base_dir / '三张基础图生成清单.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No samples in {manifest_path}')

    directories = {
        'original': output_dir / '01_原图_512x512',
        'pseg': output_dir / '02_Pseg_彩色热力图_JET',
        'asem': output_dir / '03_Asem_彩色热力图_JET',
        'minner': output_dir / '04_Minner_叶片可信区域',
        'combined': output_dir / '05_原图_Pseg_Asem_Minner_四联图',
    }
    failures = []
    exported = 0
    class_counts = Counter()

    for index, row in enumerate(rows, start=1):
        class_name = row['class_name']
        stem = Path(row['filename']).stem
        source_path = Path(row['source_path'])
        npz_path = base_dir / row['numeric_npz']
        try:
            original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError(f'Unreadable source image: {source_path}')
            original = cv2.resize(
                original, (args.size, args.size), interpolation=cv2.INTER_AREA)

            with np.load(npz_path) as arrays:
                pseg = arrays['p_seg'].astype(np.float32)
                asem = arrays['a_sem'].astype(np.float32)
                minner = arrays['m_inner'].astype(bool)
            if not (pseg.shape == asem.shape == minner.shape):
                raise ValueError(
                    f'Array shapes differ in {npz_path}: '
                    f'{pseg.shape}, {asem.shape}, {minner.shape}')
            if pseg.shape != (args.size, args.size):
                pseg = cv2.resize(
                    pseg, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
                asem = cv2.resize(
                    asem, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
                minner = cv2.resize(
                    minner.astype(np.uint8), (args.size, args.size),
                    interpolation=cv2.INTER_NEAREST).astype(bool)

            pseg_heat = probability_heatmap(pseg)
            asem_heat = probability_heatmap(asem)
            minner_view = minner_colour(minner)
            relative = Path(class_name) / f'{stem}.png'

            write_image(directories['original'] / relative, original)
            write_image(directories['pseg'] / relative, pseg_heat)
            write_image(directories['asem'] / relative, asem_heat)
            write_image(directories['minner'] / relative, minner_view)

            panels = [
                labelled_panel(original, 'Original'),
                labelled_panel(pseg_heat, 'Pseg'),
                labelled_panel(asem_heat, 'Asem'),
                labelled_panel(minner_view, 'Minner'),
            ]
            write_image(directories['combined'] / relative,
                        np.concatenate(panels, axis=1))
            exported += 1
            class_counts[class_name] += 1
        except Exception as error:
            failures.append({
                'sample_index': index,
                'class_name': class_name,
                'filename': row.get('filename', ''),
                'source_path': str(source_path),
                'numeric_npz': str(npz_path),
                'error': repr(error),
            })

        if index % 100 == 0 or index == len(rows):
            print(
                f'[{index}/{len(rows)}] exported={exported} '
                f'failed={len(failures)}', flush=True)

    summary = {
        'source_base_maps': str(base_dir.resolve()),
        'source_manifest': str(manifest_path.resolve()),
        'output_directory': str(output_dir.resolve()),
        'requested_samples': len(rows),
        'exported_samples': exported,
        'failed_samples': len(failures),
        'class_counts': dict(sorted(class_counts.items())),
        'image_size': [args.size, args.size],
        'pseg_visualization': 'OpenCV COLORMAP_JET; fixed input scale [0, 1]',
        'asem_visualization': 'OpenCV COLORMAP_JET; fixed input scale [0, 1]',
        'minner_visualization': 'binary mask; green foreground, black background',
        'separate_images_per_sample': 4,
        'combined_four_panel_per_sample': 1,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / '导出汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if failures:
        with (output_dir / '导出失败清单.csv').open(
                'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0]))
            writer.writeheader()
            writer.writerows(failures)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(
            f'{len(failures)} samples failed; see 导出失败清单.csv')


if __name__ == '__main__':
    main()

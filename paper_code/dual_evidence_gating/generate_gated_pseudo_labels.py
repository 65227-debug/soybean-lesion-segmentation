#!/usr/bin/env python
"""Run pixel gating, weighting, and whole-image quality filtering."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from gating_core import (
    BACKGROUND, HEALTHY, IGNORE, LESION, apply_image_filter,
    assess_image_quality, calculate_image_statistics, colorize_labels,
    gate_pixels, load_config)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-maps', default='data/dadou/05_三张基础图')
    parser.add_argument(
        '--source-manifest', default='data/dadou_semantic/train_manifest.csv')
    parser.add_argument(
        '--config',
        default='paper_code/dual_evidence_gating/selected_config.json')
    parser.add_argument(
        '--output-dir', default='data/dadou/06_07_08_双证据门控')
    parser.add_argument('--preview-per-class', type=int, default=8)
    return parser.parse_args()


def to_uint16(array):
    return np.clip(np.rint(array * 65535), 0, 65535).astype(np.uint16)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metadata_key(row):
    return row.get('class_name', ''), Path(
        row.get('linked_path', row.get('source_path', ''))).name


def make_preview(image, candidate, final, weights, accepted, reasons):
    height, width = candidate.shape
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    candidate_color = colorize_labels(candidate)
    final_color = colorize_labels(final)
    weight_heat = cv2.applyColorMap(
        np.rint(weights * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    panels = [image, candidate_color, final_color, weight_heat]
    labels = ['Image', 'Candidate', 'Final', 'Weight']
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (175, 34), (255, 255, 255), -1)
        cv2.putText(
            panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (20, 20, 20), 2, cv2.LINE_AA)
    canvas = np.concatenate(panels, axis=1)
    status = 'ACCEPTED' if accepted else 'REJECTED: ' + '|'.join(reasons)
    cv2.rectangle(canvas, (0, height - 40), (canvas.shape[1], height),
                  (255, 255, 255), -1)
    cv2.putText(
        canvas, status[:150], (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (0, 100, 0) if accepted else (0, 0, 200), 2, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    base_maps = Path(args.base_maps)
    output_dir = Path(args.output_dir)
    output_dirs = {
        'candidate': output_dir / '01_门控候选伪标签',
        'final': output_dir / '02_整图筛选后伪标签',
        'weight': output_dir / '03_像素权重图',
        'numeric': output_dir / '04_训练数值_npz',
        'preview': output_dir / '05_抽样预览',
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    base_rows = read_csv(base_maps / '三张基础图生成清单.csv')
    source_rows = read_csv(args.source_manifest)
    source_by_key = {metadata_key(row): row for row in source_rows}
    preview_counts = Counter()
    output_rows = []
    reason_counts = Counter()
    accepted_counts = Counter()
    label_pixel_counts = Counter()
    total_final_pixels = 0

    for index, base_row in enumerate(base_rows, start=1):
        class_name = base_row['class_name']
        stem = Path(base_row['filename']).stem
        key = (class_name, Path(base_row['source_path']).name)
        source_row = source_by_key.get(key, {})
        arrays = np.load(base_maps / base_row['numeric_npz'])
        p_seg = arrays['p_seg'].astype(np.float32)
        p_leaf = arrays['p_leaf'].astype(np.float32)
        a_sem = arrays['a_sem'].astype(np.float32)
        healthy_image = class_name.lower() in {'healthy', 'hea'}
        candidate, candidate_weights, masks = gate_pixels(
            p_seg, p_leaf, a_sem, config, healthy_image=healthy_image)
        statistics = calculate_image_statistics(candidate, masks)
        metadata = {
            **source_row,
            'semantic_label_probability': base_row.get(
                'semantic_label_probability', ''),
        }
        image_accepted, reasons = assess_image_quality(
            statistics, metadata, config, healthy_image=healthy_image)
        final, final_weights = apply_image_filter(
            candidate, candidate_weights, image_accepted)
        for reason in reasons:
            reason_counts[reason] += 1
        if image_accepted:
            accepted_counts[class_name] += 1
        for label in (BACKGROUND, HEALTHY, LESION, IGNORE):
            label_pixel_counts[label] += int((final == label).sum())
        total_final_pixels += final.size

        class_dir = Path(class_name)
        candidate_path = output_dirs['candidate'] / class_dir / f'{stem}.png'
        final_path = output_dirs['final'] / class_dir / f'{stem}.png'
        weight_path = output_dirs['weight'] / class_dir / f'{stem}.png'
        numeric_path = output_dirs['numeric'] / class_dir / f'{stem}.npz'
        preview_path = output_dirs['preview'] / class_dir / f'{stem}.jpg'
        for path in (candidate_path, final_path, weight_path, numeric_path,
                     preview_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(candidate_path), candidate)
        cv2.imwrite(str(final_path), final)
        cv2.imwrite(str(weight_path), to_uint16(final_weights))
        np.savez_compressed(
            numeric_path,
            candidate_label=candidate.astype(np.uint8),
            candidate_pixel_weight=candidate_weights.astype(np.float32),
            pseudo_label=final.astype(np.uint8),
            pixel_weight=final_weights.astype(np.float32),
            image_accepted=np.asarray(image_accepted, dtype=np.uint8))

        save_preview = preview_counts[class_name] < args.preview_per_class
        if save_preview:
            original = cv2.imread(base_row['source_path'], cv2.IMREAD_COLOR)
            if original is not None:
                preview = make_preview(
                    original, candidate, final, final_weights,
                    image_accepted, reasons)
                cv2.imwrite(
                    str(preview_path), preview,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                preview_counts[class_name] += 1

        output_row = {
            'sample_index': index,
            'filename': base_row['filename'],
            'class_name': class_name,
            'scene_id': base_row.get('scene_id', ''),
            'source_path': base_row['source_path'],
            'image_accepted': image_accepted,
            'rejection_reasons': '|'.join(reasons),
            **statistics,
            'semantic_label_probability': base_row.get(
                'semantic_label_probability', ''),
            'blur_score': source_row.get('blur_score', ''),
            'brightness_mean': source_row.get('brightness_mean', ''),
            'candidate_label_path': str(candidate_path.relative_to(output_dir)),
            'pseudo_label_path': str(final_path.relative_to(output_dir)),
            'pixel_weight_path': str(weight_path.relative_to(output_dir)),
            'numeric_npz_path': str(numeric_path.relative_to(output_dir)),
            'preview_path': (
                str(preview_path.relative_to(output_dir)) if save_preview else ''),
        }
        output_rows.append(output_row)
        if index % 100 == 0 or index == len(base_rows):
            print(
                f'[{index}/{len(base_rows)}] accepted='
                f'{sum(accepted_counts.values())} rejected='
                f'{index - sum(accepted_counts.values())}', flush=True)

    write_csv(output_dir / '全部图像质量与门控清单.csv', output_rows)
    accepted_rows = [row for row in output_rows if row['image_accepted']]
    rejected_rows = [row for row in output_rows if not row['image_accepted']]
    write_csv(
        output_dir / '可用于学生训练.csv', accepted_rows,
        fieldnames=list(output_rows[0]))
    write_csv(
        output_dir / '整图排除.csv', rejected_rows,
        fieldnames=list(output_rows[0]))
    summary = {
        'steps': [
            '6: dual-evidence positive/negative/ignore pixel gating',
            '7: confidence-aware pixel weights',
            '8: whole-image quality filtering',
        ],
        'config': str(Path(args.config).resolve()),
        'input_images': len(output_rows),
        'accepted_images': len(accepted_rows),
        'rejected_images': len(rejected_rows),
        'acceptance_percent': 100.0 * len(accepted_rows) / len(output_rows),
        'accepted_by_class': dict(accepted_counts),
        'rejected_by_class': {
            name: sum(
                (not row['image_accepted']) and row['class_name'] == name
                for row in output_rows)
            for name in sorted({row['class_name'] for row in output_rows})
        },
        'rejection_reason_counts': dict(reason_counts),
        'final_pixel_distribution_percent': {
            'background_0': 100.0 * label_pixel_counts[BACKGROUND] / total_final_pixels,
            'healthy_1': 100.0 * label_pixel_counts[HEALTHY] / total_final_pixels,
            'lesion_2': 100.0 * label_pixel_counts[LESION] / total_final_pixels,
            'ignore_255': 100.0 * label_pixel_counts[IGNORE] / total_final_pixels,
        },
        'continuous_weight_encoding': 'weight = uint16_pixel / 65535.0',
        'test_split_used': False,
    }
    (output_dir / '第六七八步汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Prepare B/D/E ablation pseudo labels from the frozen base teachers."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from mmseg.apis import inference_model, init_model


IGNORE = 255
BACKGROUND = 0
HEALTHY = 1
LESION = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-maps', default='data/dadou/05_三张基础图')
    parser.add_argument(
        '--teacher-config',
        default=('work_dirs/dadou_deeplabv3plus_r50_teacher/'
                 'deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py'))
    parser.add_argument(
        '--teacher-checkpoint',
        default=('work_dirs/dadou_deeplabv3plus_r50_teacher/'
                 'best_mIoU_epoch_23_inference.pth'))
    parser.add_argument(
        '--config', default='paper_code/ablation_abdec/ablation_config.json')
    parser.add_argument('--output-root', default='data/dadou_abdec')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=None)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_uint16(array):
    return np.clip(np.rint(array * 65535.0), 0, 65535).astype(np.uint16)


def teacher_probabilities(model, image_bgr):
    result = inference_model(model, image_bgr)
    logits = result.seg_logits.data
    if logits.ndim == 4:
        logits = logits.squeeze(0)
    probabilities = torch.softmax(logits, dim=0).detach().float().cpu().numpy()
    if probabilities.shape[1:] != image_bgr.shape[:2]:
        probabilities = np.stack([
            cv2.resize(item, (image_bgr.shape[1], image_bgr.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
            for item in probabilities
        ])
    return probabilities.astype(np.float32)


def labels_b(probabilities, threshold):
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0)
    labels = np.full(prediction.shape, IGNORE, dtype=np.uint8)
    accepted = confidence >= threshold
    labels[accepted] = prediction[accepted]
    weights = np.where(accepted, confidence, 0).astype(np.float32)
    return labels, weights


def labels_d(p_seg, p_leaf, inner, config):
    labels = np.full(p_seg.shape, IGNORE, dtype=np.uint8)
    weights = np.zeros(p_seg.shape, dtype=np.float32)
    background = p_leaf <= config['background_leaf_threshold']
    healthy = (p_seg <= config['tau_n']) & inner
    lesion = (p_seg >= config['tau_p']) & inner
    labels[background] = BACKGROUND
    labels[healthy] = HEALTHY
    labels[lesion] = LESION
    weights[background] = (
        (1.0 - p_leaf[background]) * (1.0 - p_seg[background]))
    weights[healthy] = 1.0 - p_seg[healthy]
    weights[lesion] = p_seg[lesion]
    return labels, np.clip(weights, 0, 1)


def labels_e(probabilities, a_sem, config):
    p_seg = probabilities[LESION]
    labels = np.full(p_seg.shape, IGNORE, dtype=np.uint8)
    weights = np.zeros(p_seg.shape, dtype=np.float32)
    lesion = (p_seg >= config['tau_p']) & (a_sem >= config['tau_s'])
    non_lesion = (p_seg <= config['tau_n']) & (a_sem <= config['tau_sn'])
    non_lesion_prediction = probabilities[:2].argmax(axis=0).astype(np.uint8)
    non_lesion_confidence = probabilities[:2].max(axis=0)
    labels[non_lesion] = non_lesion_prediction[non_lesion]
    labels[lesion] = LESION
    weights[non_lesion] = (
        non_lesion_confidence[non_lesion] * (1.0 - a_sem[non_lesion]))
    weights[lesion] = p_seg[lesion] * a_sem[lesion]
    return labels, np.clip(weights, 0, 1)


def main():
    args = parse_args()
    configuration = json.loads(Path(args.config).read_text(encoding='utf-8'))
    pseudo_config = configuration['pseudo_generation']
    rows = read_csv(Path(args.base_maps) / '三张基础图生成清单.csv')
    if args.max_images is not None:
        rows = rows[:args.max_images]
    output_root = Path(args.output_root)
    image_dir = output_root / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for mode in ('B', 'D', 'E'):
        (output_root / mode / 'masks').mkdir(parents=True, exist_ok=True)
        (output_root / mode / 'weights').mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    teacher = init_model(
        args.teacher_config, args.teacher_checkpoint, device=device)
    manifests = {mode: [] for mode in ('B', 'D', 'E')}
    counts = {mode: Counter() for mode in ('B', 'D', 'E')}
    lesion_probability_difference = []
    for index, row in enumerate(rows, start=1):
        class_name = row['class_name']
        sample_id = f"{class_name}__{Path(row['filename']).stem}"
        source_image = Path(row['source_path'])
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable image: {source_image}')
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        output_image = image_dir / f'{sample_id}.jpg'
        cv2.imwrite(str(output_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        arrays = np.load(Path(args.base_maps) / row['numeric_npz'])
        p_seg = arrays['p_seg'].astype(np.float32)
        p_leaf = arrays['p_leaf'].astype(np.float32)
        inner = arrays['m_inner'].astype(bool)
        a_sem = arrays['a_sem'].astype(np.float32)
        probabilities = teacher_probabilities(teacher, image)
        lesion_probability_difference.append(float(
            np.mean(np.abs(probabilities[LESION] - p_seg))))
        mode_outputs = {
            'B': labels_b(
                probabilities, pseudo_config['confidence_threshold_b']),
            'D': labels_d(p_seg, p_leaf, inner, pseudo_config),
            'E': labels_e(probabilities, a_sem, pseudo_config),
        }
        for mode, (labels, weights) in mode_outputs.items():
            mask_path = output_root / mode / 'masks' / f'{sample_id}.png'
            weight_path = output_root / mode / 'weights' / f'{sample_id}.png'
            cv2.imwrite(str(mask_path), labels)
            cv2.imwrite(str(weight_path), to_uint16(weights))
            for label in (BACKGROUND, HEALTHY, LESION, IGNORE):
                counts[mode][label] += int((labels == label).sum())
            manifests[mode].append({
                'sample_id': sample_id,
                'sample_type': 'pseudo',
                'split': 'train',
                'class_name': class_name,
                'image_path': str(output_image.resolve()),
                'mask_path': str(mask_path.resolve()),
                'weight_path': str(weight_path.resolve()),
                'source_image_path': str(source_image.resolve()),
            })
        if index % 100 == 0 or index == len(rows):
            print(f'[{index}/{len(rows)}] prepared B/D/E', flush=True)

    for mode, manifest_rows in manifests.items():
        write_csv(output_root / f'pseudo_{mode}_manifest.csv', manifest_rows)
    total_pixels = len(rows) * 512 * 512
    summary = {
        'images': len(rows),
        'modes': {},
        'thresholds': pseudo_config,
        'teacher_lesion_probability_mean_absolute_difference_from_step5': (
            float(np.mean(lesion_probability_difference))),
        'test_split_used': False,
    }
    for mode in ('B', 'D', 'E'):
        summary['modes'][mode] = {
            'manifest': str((output_root / f'pseudo_{mode}_manifest.csv').resolve()),
            'pixel_counts': {str(k): counts[mode][k] for k in (0, 1, 2, 255)},
            'pixel_percent': {
                str(k): 100.0 * counts[mode][k] / total_pixels
                for k in (0, 1, 2, 255)
            },
        }
    (output_root / 'pseudo_generation_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

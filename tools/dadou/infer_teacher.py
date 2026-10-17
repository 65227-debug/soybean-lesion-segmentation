#!/usr/bin/env python
"""Run the trained teacher and export masks plus lesion probability maps."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.logging.history_buffer import HistoryBuffer
from mmseg.apis import inference_model, init_model


PALETTE_BGR = np.asarray([[0, 0, 0], [0, 180, 0], [30, 30, 220]], dtype=np.uint8)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('input', help='Image file or image directory')
    parser.add_argument('--output-dir', default='outputs/dadou_teacher')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def collect_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # MMEngine checkpoints contain HistoryBuffer in their metadata. PyTorch
    # 2.6+ defaults to a restricted loader, so explicitly allow this trusted
    # MMEngine type for checkpoints produced by this local training run.
    torch.serialization.add_safe_globals([HistoryBuffer])
    model = init_model(args.config, args.checkpoint, device=args.device)

    records = []
    for image_path in collect_images(args.input):
        result = inference_model(model, str(image_path))
        prediction = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        logits = result.seg_logits.data.squeeze()
        lesion_probability = torch.softmax(logits, dim=0)[2].cpu().numpy()

        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f'Cannot read {image_path}')
        if prediction.shape != original.shape[:2]:
            prediction = cv2.resize(
                prediction,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        if lesion_probability.shape != original.shape[:2]:
            lesion_probability = cv2.resize(
                lesion_probability,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        color_mask = PALETTE_BGR[prediction]
        overlay = cv2.addWeighted(original, 0.65, color_mask, 0.35, 0)
        probability_u16 = np.clip(
            np.rint(lesion_probability * 65535), 0, 65535).astype(np.uint16)

        stem = image_path.stem
        cv2.imwrite(str(output_dir / f'{stem}.mask.png'), prediction)
        cv2.imwrite(str(output_dir / f'{stem}.color.png'), color_mask)
        cv2.imwrite(str(output_dir / f'{stem}.lesion_prob16.png'), probability_u16)
        cv2.imwrite(str(output_dir / f'{stem}.overlay.jpg'), overlay)

        records.append({
            'sample_id': stem,
            'height': original.shape[0],
            'width': original.shape[1],
            'predicted_background_percent': float(np.mean(prediction == 0) * 100),
            'predicted_healthy_leaf_percent': float(np.mean(prediction == 1) * 100),
            'predicted_lesion_percent': float(np.mean(prediction == 2) * 100),
            'mean_lesion_probability': float(lesion_probability.mean()),
        })

    with (output_dir / 'prediction_summary.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Exported {len(records)} samples to {output_dir.resolve()}')


if __name__ == '__main__':
    main()

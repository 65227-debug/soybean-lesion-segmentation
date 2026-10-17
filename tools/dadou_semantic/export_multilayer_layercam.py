#!/usr/bin/env python
"""Export multi-layer LayerCAM maps from an EfficientNet-B0 classifier."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--manifest', required=True)
    parser.add_argument(
        '--output-dir', default='outputs/soybean_semantic_layercam')
    parser.add_argument(
        '--target', choices=('label', 'prediction'), default='label')
    parser.add_argument('--max-images', type=int, default=None)
    return parser.parse_args()


class LayerCamCollector:
    def __init__(self, modules):
        self.activations = {}
        self.gradients = {}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_hook(self._forward_hook(name)))

    def _forward_hook(self, name):
        def hook(_module, _inputs, output):
            self.activations[name] = output
            output.register_hook(
                lambda gradient, key=name: self.gradients.__setitem__(
                    key, gradient))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def normalize_cam(cam):
    cam = cam - cam.min()
    maximum = cam.max()
    if maximum > 0:
        cam = cam / maximum
    return cam


def build_model(checkpoint, device):
    class_to_idx = checkpoint['class_to_idx']
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(class_to_idx))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model


def generate_cams(model, collector, tensor, target_index, layer_names):
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probabilities = logits.softmax(dim=1)
    logits[0, target_index].backward()
    layer_cams = {}
    for name in layer_names:
        activation = collector.activations[name]
        positive_gradient = F.relu(collector.gradients[name])
        cam = F.relu((positive_gradient * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=tensor.shape[-2:], mode='bilinear', align_corners=False)
        layer_cams[name] = normalize_cam(cam[0, 0].detach()).cpu().numpy()
    fused = normalize_cam(np.mean(list(layer_cams.values()), axis=0))
    return probabilities[0].detach().cpu().numpy(), layer_cams, fused


def save_cam_outputs(image_path, output_stem, layer_cams, fused):
    original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise ValueError(f'Unreadable image: {image_path}')
    height, width = original_bgr.shape[:2]
    for layer_name, cam in layer_cams.items():
        resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(
            str(output_stem.with_name(
                f'{output_stem.name}.{layer_name}.layercam16.png')),
            np.round(resized * 65535).astype(np.uint16))
    fused_resized = cv2.resize(
        fused, (width, height), interpolation=cv2.INTER_LINEAR)
    fused_path = output_stem.with_name(
        f'{output_stem.name}.multilayer_layercam16.png')
    cv2.imwrite(str(fused_path), np.round(fused_resized * 65535).astype(np.uint16))
    heatmap = cv2.applyColorMap(
        np.round(fused_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)
    overlay_path = output_stem.with_name(
        f'{output_stem.name}.multilayer_overlay.jpg')
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return overlay_path


def make_contact_sheet(items, output_path, columns=4, tile_width=320):
    if not items:
        return
    tile_height = 250
    rows = math.ceil(len(items) / columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, np.uint8)
    for index, item in enumerate(items):
        image = cv2.imread(str(item['overlay_path']), cv2.IMREAD_COLOR)
        if image is None:
            continue
        available_height = tile_height - 34
        scale = min(tile_width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image, (max(1, int(image.shape[1] * scale)),
                    max(1, int(image.shape[0] * scale))))
        row, column = divmod(index, columns)
        x = column * tile_width + (tile_width - resized.shape[1]) // 2
        y = row * tile_height + 34 + (available_height - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        label = f"GT:{item['true_class']} P:{item['predicted_class']}"
        cv2.putText(
            canvas, label, (column * tile_width + 6, row * tile_height + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1,
            cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint, device)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    feature_indices = tuple(checkpoint.get(
        'layercam_feature_indices', (3, 5, 8)))
    layer_names = [f'features_{index}' for index in feature_indices]
    modules = {
        name: model.features[index]
        for name, index in zip(layer_names, feature_indices)
    }
    collector = LayerCamCollector(modules)
    input_size = int(checkpoint.get('input_size', 224))
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    if args.max_images is not None:
        manifest = manifest[:args.max_images]
    prediction_rows = []
    contact_items = []
    for row in manifest:
        image_path = Path(row['linked_path'])
        true_index = class_to_idx[row['class_name']]
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.enable_grad():
            initial_logits = model(tensor)
            predicted_index = int(initial_logits.argmax(dim=1).item())
            target_index = true_index if args.target == 'label' else predicted_index
            probabilities, layer_cams, fused = generate_cams(
                model, collector, tensor, target_index, layer_names)
        class_dir = output_dir / row['class_name']
        class_dir.mkdir(parents=True, exist_ok=True)
        output_stem = class_dir / image_path.stem
        overlay_path = save_cam_outputs(
            image_path, output_stem, layer_cams, fused)
        prediction = int(np.argmax(probabilities))
        result = {
            'filename': image_path.name,
            'image_path': str(image_path.resolve()),
            'true_class': row['class_name'],
            'predicted_class': idx_to_class[prediction],
            'correct': prediction == true_index,
            'cam_target_class': idx_to_class[target_index],
            'layercam_feature_indices': ','.join(map(str, feature_indices)),
            'fused_cam_mean': float(fused.mean()),
            'fused_cam_max': float(fused.max()),
        }
        for index, name in idx_to_class.items():
            result[f'probability_{name}'] = float(probabilities[index])
        prediction_rows.append(result)
        contact_items.append({
            'overlay_path': overlay_path,
            'true_class': row['class_name'],
            'predicted_class': idx_to_class[prediction],
        })

    collector.close()
    with (output_dir / 'layercam_predictions.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    make_contact_sheet(contact_items, output_dir / 'layercam_contact_sheet.jpg')
    summary = {
        'model': 'EfficientNet-B0',
        'method': 'Multi-layer LayerCAM',
        'feature_indices': list(feature_indices),
        'fusion': 'mean of independently normalized LayerCAM maps',
        'cam_target': args.target,
        'images_exported': len(prediction_rows),
        'classification_accuracy_percent': 100 * np.mean(
            [row['correct'] for row in prediction_rows]),
        'test_split_used': False,
    }
    with (output_dir / 'layercam_summary.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

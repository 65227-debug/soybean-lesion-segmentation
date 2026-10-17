#!/usr/bin/env python
"""Generate the three aligned base maps used by dual-evidence gating.

For every weakly labelled image this script exports:

1. the lesion probability from the DeepLabV3+ segmentation teacher;
2. the eroded trustworthy leaf interior from the independent U-Net;
3. the label-targeted, multi-layer LayerCAM from EfficientNet-B0.

All outputs are aligned to a common square resolution. Continuous maps are
stored both as lossless uint16 PNGs and as float32 arrays in compressed NPZ
files so later threshold searches do not depend on visualization images.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

from mmseg.apis import inference_model, init_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', default='data/dadou_semantic/train_manifest.csv')
    parser.add_argument(
        '--teacher-config',
        default=('work_dirs/dadou_deeplabv3plus_r50_teacher/'
                 'deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py'))
    parser.add_argument(
        '--teacher-checkpoint',
        default=('work_dirs/dadou_deeplabv3plus_r50_teacher/'
                 'best_mIoU_epoch_23_inference.pth'))
    parser.add_argument(
        '--leaf-config',
        default=('work_dirs/dadou_unet_leaf/'
                 'unet-s5-d16_fcn_1xb4-60e_dadou-leaf-512x512.py'))
    parser.add_argument(
        '--leaf-checkpoint',
        default=('work_dirs/dadou_unet_leaf/'
                 'best_IoU_leaf_epoch_46_inference.pth'))
    parser.add_argument(
        '--semantic-checkpoint',
        default='work_dirs/dadou_efficientnet_b0_semantic/best_macro_f1.pth')
    parser.add_argument(
        '--output-dir', default='data/dadou/05_三张基础图')
    parser.add_argument('--output-size', type=int, default=512)
    parser.add_argument('--leaf-threshold', type=float, default=0.5)
    parser.add_argument('--erosion-radius', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument(
        '--preview-limit', type=int, default=24,
        help='Number of four-panel JPEG previews to save; -1 saves all.')
    parser.add_argument(
        '--no-npz', action='store_true',
        help='Do not save compressed float32 arrays used by later gating.')
    return parser.parse_args()


class LayerCamCollector:
    """Collect feature activations and their gradients for LayerCAM."""

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


def normalize_map(array):
    array = array - array.min()
    maximum = float(array.max())
    if maximum > 0:
        array = array / maximum
    return array


def build_semantic_model(checkpoint, device):
    class_to_idx = checkpoint['class_to_idx']
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(class_to_idx))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model


def generate_layercam(model, collector, tensor, target_index, layer_names,
                      output_size):
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probabilities = logits.softmax(dim=1)
    logits[0, target_index].backward()
    layer_cams = []
    for name in layer_names:
        activation = collector.activations[name]
        positive_gradient = F.relu(collector.gradients[name])
        cam = F.relu((positive_gradient * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(output_size, output_size), mode='bilinear',
            align_corners=False)
        layer_cams.append(
            normalize_map(cam[0, 0].detach().float().cpu().numpy()))
    fused = normalize_map(np.mean(layer_cams, axis=0)).astype(np.float32)
    return probabilities[0].detach().float().cpu().numpy(), fused


def mmseg_probabilities(model, image_bgr):
    result = inference_model(model, image_bgr)
    logits = result.seg_logits.data.squeeze(0) \
        if result.seg_logits.data.ndim == 4 else result.seg_logits.data
    return torch.softmax(logits, dim=0).detach().float().cpu().numpy()


def to_uint16(array):
    return np.clip(np.rint(array * 65535.0), 0, 65535).astype(np.uint16)


def make_preview(image_bgr, lesion_probability, leaf_inner, semantic_cam):
    size = lesion_probability.shape[0]
    image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    lesion_heat = cv2.applyColorMap(
        np.rint(lesion_probability * 255).astype(np.uint8), cv2.COLORMAP_JET)
    leaf_view = np.zeros_like(image_bgr)
    leaf_view[leaf_inner > 0] = (0, 190, 0)
    semantic_heat = cv2.applyColorMap(
        np.rint(semantic_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panels = [image_bgr, lesion_heat, leaf_view, semantic_heat]
    labels = ['Image', 'P_seg', 'M_inner', 'A_sem']
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (142, 34), (255, 255, 255), -1)
        cv2.putText(
            panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (20, 20, 20), 2, cv2.LINE_AA)
    return np.concatenate(panels, axis=1)


def ensure_paths(args):
    required = {
        'manifest': Path(args.manifest),
        'teacher config': Path(args.teacher_config),
        'teacher checkpoint': Path(args.teacher_checkpoint),
        'leaf config': Path(args.leaf_config),
        'leaf checkpoint': Path(args.leaf_checkpoint),
        'semantic checkpoint': Path(args.semantic_checkpoint),
    }
    missing = [f'{name}: {path}' for name, path in required.items()
               if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_paths(args)
    if args.output_size <= 0:
        raise ValueError('--output-size must be positive')
    if args.erosion_radius < 0:
        raise ValueError('--erosion-radius cannot be negative')
    if not 0 <= args.leaf_threshold <= 1:
        raise ValueError('--leaf-threshold must be within [0, 1]')

    with Path(args.manifest).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.max_images is not None:
        rows = rows[:args.max_images]
    if not rows:
        raise ValueError('The input manifest contains no images')

    output_dir = Path(args.output_dir)
    map_dirs = {
        'lesion': output_dir / '01_病斑概率图_Pseg',
        'leaf': output_dir / '02_叶片可信区域图_Minner',
        'semantic': output_dir / '03_病种语义响应图_Asem',
        'numeric': output_dir / '04_门控数值_npz',
        'preview': output_dir / '05_抽样预览',
    }
    for directory in map_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    teacher = init_model(
        args.teacher_config, args.teacher_checkpoint, device=str(device))
    leaf_model = init_model(
        args.leaf_config, args.leaf_checkpoint, device=str(device))
    semantic_checkpoint = torch.load(
        args.semantic_checkpoint, map_location=device, weights_only=True)
    semantic_model = build_semantic_model(semantic_checkpoint, device)
    class_to_idx = semantic_checkpoint['class_to_idx']
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    feature_indices = tuple(semantic_checkpoint.get(
        'layercam_feature_indices', (3, 5, 8)))
    layer_names = [f'features_{index}' for index in feature_indices]
    collector = LayerCamCollector({
        name: semantic_model.features[index]
        for name, index in zip(layer_names, feature_indices)
    })
    semantic_input_size = int(semantic_checkpoint.get('input_size', 224))
    semantic_transform = transforms.Compose([
        transforms.Resize((semantic_input_size, semantic_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if args.erosion_radius:
        width = 2 * args.erosion_radius + 1
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (width, width))
    else:
        erosion_kernel = None

    output_rows = []
    failures = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        image_path = Path(row['linked_path'])
        class_name = row['class_name']
        try:
            if class_name not in class_to_idx:
                raise ValueError(
                    f'Class {class_name!r} is absent from semantic checkpoint')
            original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if original_bgr is None:
                raise ValueError(f'Unreadable image: {image_path}')
            original_height, original_width = original_bgr.shape[:2]
            aligned_bgr = cv2.resize(
                original_bgr, (args.output_size, args.output_size),
                interpolation=cv2.INTER_AREA)

            teacher_probabilities = mmseg_probabilities(
                teacher, aligned_bgr)
            lesion_probability = teacher_probabilities[2]
            leaf_probability = mmseg_probabilities(
                leaf_model, aligned_bgr)[1]
            if lesion_probability.shape != (args.output_size, args.output_size):
                lesion_probability = cv2.resize(
                    lesion_probability, (args.output_size, args.output_size),
                    interpolation=cv2.INTER_LINEAR)
            if teacher_probabilities.shape[1:] != (
                    args.output_size, args.output_size):
                teacher_probabilities = np.stack([
                    cv2.resize(
                        probability, (args.output_size, args.output_size),
                        interpolation=cv2.INTER_LINEAR)
                    for probability in teacher_probabilities
                ])
            if leaf_probability.shape != (args.output_size, args.output_size):
                leaf_probability = cv2.resize(
                    leaf_probability, (args.output_size, args.output_size),
                    interpolation=cv2.INTER_LINEAR)
            lesion_probability = np.clip(
                lesion_probability, 0, 1).astype(np.float32)
            teacher_probabilities = np.clip(
                teacher_probabilities, 0, 1).astype(np.float32)
            leaf_probability = np.clip(
                leaf_probability, 0, 1).astype(np.float32)
            leaf_mask = (leaf_probability >= args.leaf_threshold).astype(np.uint8)
            leaf_inner = cv2.erode(leaf_mask, erosion_kernel) \
                if erosion_kernel is not None else leaf_mask

            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            semantic_tensor = semantic_transform(
                Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
            semantic_probabilities, semantic_cam = generate_layercam(
                semantic_model, collector, semantic_tensor,
                class_to_idx[class_name], layer_names, args.output_size)

            class_subdir = Path(class_name)
            stem = image_path.stem
            lesion_path = map_dirs['lesion'] / class_subdir / f'{stem}.png'
            leaf_path = map_dirs['leaf'] / class_subdir / f'{stem}.png'
            semantic_path = map_dirs['semantic'] / class_subdir / f'{stem}.png'
            numeric_path = map_dirs['numeric'] / class_subdir / f'{stem}.npz'
            preview_path = map_dirs['preview'] / class_subdir / f'{stem}.jpg'
            for path in (lesion_path, leaf_path, semantic_path, numeric_path,
                         preview_path):
                path.parent.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(lesion_path), to_uint16(lesion_probability)):
                raise OSError(f'Failed to write {lesion_path}')
            if not cv2.imwrite(str(leaf_path), leaf_inner * 255):
                raise OSError(f'Failed to write {leaf_path}')
            if not cv2.imwrite(str(semantic_path), to_uint16(semantic_cam)):
                raise OSError(f'Failed to write {semantic_path}')
            if not args.no_npz:
                np.savez_compressed(
                    numeric_path,
                    p_seg=lesion_probability.astype(np.float32),
                    p_teacher=teacher_probabilities.astype(np.float32),
                    p_leaf=leaf_probability.astype(np.float32),
                    m_inner=leaf_inner.astype(np.uint8),
                    a_sem=semantic_cam.astype(np.float32))
            save_preview = args.preview_limit < 0 or index <= args.preview_limit
            if save_preview:
                preview = make_preview(
                    aligned_bgr, lesion_probability, leaf_inner, semantic_cam)
                cv2.imwrite(
                    str(preview_path), preview,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

            predicted_index = int(np.argmax(semantic_probabilities))
            result_row = {
                'sample_index': index,
                'filename': image_path.name,
                'class_name': class_name,
                'scene_id': row.get('scene_id', ''),
                'source_path': str(image_path.resolve()),
                'original_height': original_height,
                'original_width': original_width,
                'map_height': args.output_size,
                'map_width': args.output_size,
                'lesion_probability_mean': float(lesion_probability.mean()),
                'lesion_probability_max': float(lesion_probability.max()),
                'lesion_area_at_0.5_percent': float(
                    100 * np.mean(lesion_probability >= 0.5)),
                'leaf_area_before_erosion_percent': float(100 * leaf_mask.mean()),
                'leaf_inner_area_percent': float(100 * leaf_inner.mean()),
                'semantic_cam_mean': float(semantic_cam.mean()),
                'semantic_predicted_class': idx_to_class[predicted_index],
                'semantic_label_probability': float(
                    semantic_probabilities[class_to_idx[class_name]]),
                'semantic_prediction_correct': predicted_index == class_to_idx[class_name],
                'pseg_png': str(lesion_path.relative_to(output_dir)),
                'minner_png': str(leaf_path.relative_to(output_dir)),
                'asem_png': str(semantic_path.relative_to(output_dir)),
                'numeric_npz': (
                    str(numeric_path.relative_to(output_dir))
                    if not args.no_npz else ''),
                'preview_jpg': (
                    str(preview_path.relative_to(output_dir))
                    if save_preview else ''),
            }
            for class_index, name in idx_to_class.items():
                result_row[f'semantic_probability_{name}'] = float(
                    semantic_probabilities[class_index])
            output_rows.append(result_row)
        except Exception as error:  # continue so one corrupt image is auditable
            failures.append({
                'sample_index': index,
                'filename': row.get('filename', image_path.name),
                'class_name': class_name,
                'source_path': str(image_path),
                'error': repr(error),
            })
        if index % 25 == 0 or index == len(rows):
            elapsed = time.time() - started
            print(
                f'[{index}/{len(rows)}] exported={len(output_rows)} '
                f'failed={len(failures)} elapsed={elapsed:.1f}s', flush=True)
            write_csv(output_dir / '三张基础图生成清单.csv', output_rows)
            write_csv(output_dir / '生成失败清单.csv', failures)

    collector.close()
    elapsed = time.time() - started
    summary = {
        'task': '第五步：生成三张基础图',
        'input_manifest': str(Path(args.manifest).resolve()),
        'requested_images': len(rows),
        'successfully_exported': len(output_rows),
        'failed_images': len(failures),
        'class_counts': {
            name: sum(row['class_name'] == name for row in output_rows)
            for name in class_to_idx
        },
        'output_size': [args.output_size, args.output_size],
        'continuous_png_encoding': 'probability = uint16_pixel / 65535.0',
        'numeric_encoding': (
            'compressed NPZ: p_seg float32, p_teacher[3,H,W] float32, '
            'p_leaf float32, '
            'm_inner uint8, a_sem float32'
            if not args.no_npz else 'disabled'),
        'leaf_threshold': args.leaf_threshold,
        'leaf_erosion_radius_pixels_at_output_size': args.erosion_radius,
        'semantic_cam_target': 'weak image-level disease label',
        'semantic_method': 'mean fusion of independently normalized LayerCAM',
        'semantic_feature_indices': list(feature_indices),
        'teacher_checkpoint': str(Path(args.teacher_checkpoint).resolve()),
        'leaf_checkpoint': str(Path(args.leaf_checkpoint).resolve()),
        'semantic_checkpoint': str(Path(args.semantic_checkpoint).resolve()),
        'held_out_test_split_used': False,
        'elapsed_seconds': elapsed,
    }
    if output_rows:
        summary.update({
            'mean_lesion_probability': float(np.mean([
                row['lesion_probability_mean'] for row in output_rows])),
            'mean_leaf_inner_area_percent': float(np.mean([
                row['leaf_inner_area_percent'] for row in output_rows])),
            'mean_semantic_cam_response': float(np.mean([
                row['semantic_cam_mean'] for row in output_rows])),
            'semantic_training_pool_accuracy_percent': float(100 * np.mean([
                row['semantic_prediction_correct'] for row in output_rows])),
        })
    with (output_dir / '生成汇总.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(
            f'{len(failures)} images failed; see 生成失败清单.csv')


if __name__ == '__main__':
    main()

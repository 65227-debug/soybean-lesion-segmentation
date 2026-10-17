"""Shared utilities for protocol-aligned strong SSL baselines."""

import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[1]
STEP9_DIR = CODE_DIR.parent / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as base  # noqa: E402

IMAGENET_MEAN = base.IMAGENET_MEAN
IMAGENET_STD = base.IMAGENET_STD
CLASS_NAMES = base.CLASS_NAMES


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    return json.loads(resolve(path).read_text(encoding='utf-8'))


def read_rows(path):
    with resolve(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def image_only_unlabeled_rows(path):
    """Return a hard allow-list view; weak-label and pseudo-label fields vanish."""
    rows = read_rows(path)
    return [
        {
            'sample_id': row['sample_id'],
            'image_path': row['image_path'],
            'source_image_path': row.get('source_image_path', ''),
        }
        for row in rows
    ]


def set_determinism(seed):
    base.seed_everything(seed)
    torch.use_deterministic_algorithms(False)


def normalize(image):
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def apply_geometry(image, cfg):
    if random.random() < cfg['horizontal_flip_probability']:
        image = np.flip(image, axis=1)
    if random.random() < cfg['vertical_flip_probability']:
        image = np.flip(image, axis=0)
    degrees = float(cfg['rotation_degrees'])
    if degrees > 0:
        angle = random.uniform(-degrees, degrees)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(image)


def apply_geometry_pair(image, mask, cfg, mask_border_value=255):
    if random.random() < cfg['horizontal_flip_probability']:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() < cfg['vertical_flip_probability']:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    degrees = float(cfg['rotation_degrees'])
    if degrees > 0:
        angle = random.uniform(-degrees, degrees)
        height, width = mask.shape
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(
            mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=mask_border_value)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def color_jitter(image, strength, hue_strength):
    image = image.astype(np.float32) / 255.0
    brightness = 1.0 + random.uniform(-strength, strength)
    contrast = 1.0 + random.uniform(-strength, strength)
    saturation = 1.0 + random.uniform(-strength, strength)
    image *= brightness
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast + mean
    gray = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2GRAY)[..., None]
    image = gray + saturation * (image - gray)
    image = np.clip(image, 0, 1)
    if hue_strength > 0:
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        shift = int(round(random.uniform(-hue_strength, hue_strength) * 180))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return image


def strong_augment(image, cfg):
    if random.random() < cfg['color_jitter_probability']:
        result = color_jitter(
            image, cfg['color_jitter_strength'], cfg['hue_strength'])
    else:
        result = image.astype(np.float32) / 255.0
    if random.random() < cfg['random_grayscale_probability']:
        gray = cv2.cvtColor(np.clip(result, 0, 1), cv2.COLOR_RGB2GRAY)
        result = np.repeat(gray[..., None], 3, axis=2)
    if random.random() < cfg['gaussian_blur_probability']:
        sigma = random.uniform(*cfg['gaussian_blur_sigma'])
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
    result = (np.clip(result, 0, 1) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(
        np.ascontiguousarray(result.transpose(2, 0, 1))).float()


def cutmix_box(height, width, cfg, device):
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    if random.random() > cfg['cutmix_probability']:
        return mask
    fraction = random.uniform(*cfg['cutmix_area_fraction'])
    ratio = random.uniform(*cfg['cutmix_aspect_ratio'])
    area = fraction * height * width
    box_width = max(1, int(round(math.sqrt(area / ratio))))
    box_height = max(1, int(round(math.sqrt(area * ratio))))
    box_width = min(box_width, width)
    box_height = min(box_height, height)
    x = random.randint(0, width - box_width)
    y = random.randint(0, height - box_height)
    mask[y:y + box_height, x:x + box_width] = True
    return mask


class WeakStrongUnlabeledDataset(Dataset):
    def __init__(self, manifest, base_augmentation, strong_augmentation,
                 two_strong=True):
        self.rows = image_only_unlabeled_rows(manifest)
        self.base_augmentation = base_augmentation
        self.strong_augmentation = strong_augmentation
        self.two_strong = two_strong

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_bgr = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unreadable image: {row['image_path']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = apply_geometry(image, self.base_augmentation)
        result = {
            'weak': normalize(image),
            'strong1': strong_augment(image.copy(), self.strong_augmentation),
            'sample_id': row['sample_id'],
            'image_path': row['image_path'],
        }
        if self.two_strong:
            result['strong2'] = strong_augment(
                image.copy(), self.strong_augmentation)
        return result


class ImageOnlyDataset(Dataset):
    def __init__(self, manifest):
        self.rows = image_only_unlabeled_rows(manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_bgr = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unreadable image: {row['image_path']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return normalize(image), row['sample_id'], row['image_path']


class HardPseudoDataset(Dataset):
    """ST++ dataset built only from newly generated hard pseudo masks."""
    def __init__(self, rows, pseudo_mask_dir, base_augmentation,
                 strong_augmentation, cutout_probability=0.5):
        self.rows = list(rows)
        self.pseudo_mask_dir = Path(pseudo_mask_dir)
        self.base_augmentation = base_augmentation
        self.strong_augmentation = strong_augmentation
        self.cutout_probability = cutout_probability

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_bgr = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(self.pseudo_mask_dir / f"{row['sample_id']}.png"),
            cv2.IMREAD_UNCHANGED)
        if image_bgr is None or mask is None:
            raise ValueError(f"Unreadable pseudo sample: {row['sample_id']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image, mask = apply_geometry_pair(
            image, mask, self.base_augmentation, mask_border_value=255)
        image = strong_augment(image, self.strong_augmentation)
        if random.random() < self.cutout_probability:
            height, width = mask.shape
            box_config = dict(self.strong_augmentation)
            box_config['cutmix_probability'] = 1.0
            box = cutmix_box(
                height, width, box_config, torch.device('cpu'))
            image[:, box] = 0.0
            mask[box.numpy()] = 255
        return {
            'image': image,
            'mask': torch.from_numpy(mask.astype(np.int64)),
            'sample_id': row['sample_id'],
        }


def make_loader(dataset, batch_size, shuffle, cfg, generator, drop_last):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=cfg['workers'], pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg['workers'] > 0,
        worker_init_fn=base.seed_worker, generator=generator)


def build_model(cfg, device):
    return base.build_student(resolve(cfg['model_config']), device)


def forward_logits(model, images, size=None):
    size = size or images.shape[-2:]
    return base.forward_logits(model, images, size)


def forward_with_feature_perturbation(model, images, dropout_probability):
    """UniMatch feature perturbation, adapted to MMSeg feature tuples."""
    features = model.extract_feat(images)
    doubled = tuple(torch.cat(
        [feature, F.dropout2d(
            feature, p=dropout_probability, training=True)], dim=0)
        for feature in features)
    logits = model.decode_head.forward(doubled)
    logits = F.interpolate(
        logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
    return logits.chunk(2, dim=0)


def supervised_loss(logits, targets, class_weights):
    return (
        F.cross_entropy(logits, targets, weight=class_weights)
        + base.soft_dice_loss(logits, targets))


def masked_ce(logits, targets, mask, class_weights=None, denominator='all'):
    loss = F.cross_entropy(
        logits, targets, weight=class_weights, reduction='none')
    mask = mask.float()
    if denominator == 'accepted':
        divisor = mask.sum().clamp_min(1.0)
    else:
        divisor = torch.tensor(mask.numel(), device=mask.device, dtype=mask.dtype)
    return (loss * mask).sum() / divisor


def paired_cutmix(images, masks, confidences, mix_images, mix_masks,
                  mix_confidences, boxes):
    output_images = images.clone()
    output_masks = masks.clone()
    output_confidences = confidences.clone()
    for index, box in enumerate(boxes):
        expanded = box.unsqueeze(0).expand_as(output_images[index])
        output_images[index][expanded] = mix_images[index][expanded]
        output_masks[index][box] = mix_masks[index][box]
        output_confidences[index][box] = mix_confidences[index][box]
    return output_images, output_masks, output_confidences


def evaluate_split(model, cfg, manifest, device, save_dir=None):
    dataset = base.StudentSegmentationDataset(resolve(manifest), training=False)
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = make_loader(
        dataset, cfg['validation_batch_size'], False, cfg, generator, False)
    weights = torch.tensor(cfg['class_weights'], dtype=torch.float32, device=device)
    return base.evaluate(model, loader, device, weights, save_dir=save_dir)


def save_checkpoint(path, model, epoch, validation, method, extra=None):
    payload = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': validation,
        'meta': {
            'method': method,
            'architecture': 'DeepLabV3+ MobileNetV3-Large',
            'classes': CLASS_NAMES,
            'input_size': [512, 512],
            'ignore_index': 255,
            'protocol': 'Dadou-SSL-108L-1390U-aligned-v1',
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.write_csv(path, rows)


def update_confusion(confusion, prediction, target):
    base.update_confusion(confusion, prediction, target)


def metrics_from_confusion(confusion):
    return base.metrics_from_confusion(confusion)


def cosine_lr(optimizer, step, total_steps, initial_lr, minimum_lr):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    lr = minimum_lr + 0.5 * (initial_lr - minimum_lr) * (
        1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr

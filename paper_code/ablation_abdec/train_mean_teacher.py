#!/usr/bin/env python
"""Train the C-group EMA Mean Teacher baseline without dual evidence."""

import argparse
import copy
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


STEP9_DIR = Path(__file__).resolve().parents[1] / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervised-root', default='data/dadou_student')
    parser.add_argument(
        '--unlabeled-manifest', default='data/dadou_abdec/pseudo_B_manifest.csv')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument(
        '--training-config',
        default='paper_code/ablation_abdec/ablation_config.json')
    parser.add_argument('--work-dir', default='work_dirs/dadou_ablation_C')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def color_jitter(image, strength):
    brightness = 1.0 + random.uniform(-strength, strength)
    contrast = 1.0 + random.uniform(-strength, strength)
    saturation = 1.0 + random.uniform(-strength, strength)
    image = image * brightness
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast + mean
    gray = cv2.cvtColor(
        np.clip(image, 0, 1), cv2.COLOR_RGB2GRAY)[..., None]
    image = gray + saturation * (image - gray)
    return np.clip(image, 0, 1)


class MeanTeacherUnlabeledDataset(Dataset):
    def __init__(self, manifest, augmentation, strong_color_jitter):
        self.rows = read_rows(manifest)
        self.augmentation = augmentation
        self.strong_color_jitter = strong_color_jitter

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_bgr = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unreadable image: {row['image_path']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if random.random() < self.augmentation['horizontal_flip_probability']:
            image = np.flip(image, axis=1)
        if random.random() < self.augmentation['vertical_flip_probability']:
            image = np.flip(image, axis=0)
        rotation = self.augmentation['rotation_degrees']
        if rotation > 0:
            angle = random.uniform(-rotation, rotation)
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D(
                (width / 2.0, height / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101)
        image = image.astype(np.float32) / 255.0
        weak = color_jitter(image.copy(), 0.05)
        strong = color_jitter(image.copy(), self.strong_color_jitter)
        weak = (weak - common.IMAGENET_MEAN) / common.IMAGENET_STD
        strong = (strong - common.IMAGENET_MEAN) / common.IMAGENET_STD
        return {
            'weak': torch.from_numpy(
                np.ascontiguousarray(weak.transpose(2, 0, 1))).float(),
            'strong': torch.from_numpy(
                np.ascontiguousarray(strong.transpose(2, 0, 1))).float(),
            'sample_id': row['sample_id'],
        }


@torch.no_grad()
def update_ema(teacher, student, decay):
    for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(), alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(
            teacher.buffers(), student.buffers()):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.mul_(decay).add_(
                student_buffer.detach(), alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


def make_loader(dataset, batch_size, shuffle, config, generator, drop_last):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=config['workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['workers'] > 0,
        worker_init_fn=common.seed_worker, generator=generator)


def train_epoch(student, teacher, supervised_loader, unlabeled_loader,
                optimizer, scaler, device, class_weights, lambda_u, config):
    student.train()
    teacher.eval()
    supervised_iterator = iter(supervised_loader)
    unlabeled_iterator = iter(unlabeled_loader)
    totals = {
        'supervised_ce': 0.0, 'supervised_dice': 0.0,
        'supervised_loss': 0.0, 'pseudo_loss': 0.0,
        'total_loss': 0.0, 'pseudo_mean_weight': 0.0,
        'pseudo_accepted_pixel_percent': 0.0,
    }
    confusion = np.zeros((3, 3), dtype=np.int64)
    threshold = config['mean_teacher']['confidence_threshold']
    for _ in range(config['steps_per_epoch']):
        supervised, supervised_iterator = common.next_or_restart(
            supervised_iterator, supervised_loader)
        unlabeled, unlabeled_iterator = common.next_or_restart(
            unlabeled_iterator, unlabeled_loader)
        supervised_images = supervised['image'].to(device, non_blocking=True)
        supervised_masks = supervised['mask'].to(device, non_blocking=True)
        weak = unlabeled['weak'].to(device, non_blocking=True)
        strong = unlabeled['strong'].to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            teacher_logits = common.forward_logits(
                teacher, weak, weak.shape[-2:])
            teacher_probability = teacher_logits.softmax(dim=1)
            confidence, pseudo_labels = teacher_probability.max(dim=1)
            accepted = confidence >= threshold
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            combined = torch.cat([supervised_images, strong], dim=0)
            logits = common.forward_logits(
                student, combined, supervised_masks.shape[-2:])
            supervised_logits = logits[:supervised_images.shape[0]]
            strong_logits = logits[supervised_images.shape[0]:]
            supervised_ce = F.cross_entropy(
                supervised_logits, supervised_masks, weight=class_weights)
            supervised_dice = common.soft_dice_loss(
                supervised_logits, supervised_masks)
            supervised_loss = supervised_ce + supervised_dice
            per_pixel = F.cross_entropy(
                strong_logits, pseudo_labels, weight=class_weights,
                reduction='none')
            consistency_weights = confidence * accepted.float()
            pseudo_loss = (
                (per_pixel * consistency_weights).sum()
                / (consistency_weights.sum() + 1e-6))
            total_loss = supervised_loss + lambda_u * pseudo_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            student.parameters(), config['gradient_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        update_ema(
            teacher, student, config['mean_teacher']['ema_decay'])
        totals['supervised_ce'] += supervised_ce.item()
        totals['supervised_dice'] += supervised_dice.item()
        totals['supervised_loss'] += supervised_loss.item()
        totals['pseudo_loss'] += pseudo_loss.item()
        totals['total_loss'] += total_loss.item()
        totals['pseudo_mean_weight'] += float(
            confidence[accepted].mean().item()) if accepted.any() else 0.0
        totals['pseudo_accepted_pixel_percent'] += float(
            100 * accepted.float().mean().item())
        common.update_confusion(
            confusion, supervised_logits.detach().argmax(dim=1),
            supervised_masks)
    steps = config['steps_per_epoch']
    result = {key: value / steps for key, value in totals.items()}
    result.update({
        f'train_{key}': value
        for key, value in common.metrics_from_confusion(confusion).items()
    })
    return result


def main():
    args = parse_args()
    config = json.loads(Path(args.training_config).read_text(encoding='utf-8'))
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.steps_per_epoch is not None:
        config['steps_per_epoch'] = args.steps_per_epoch
    if args.workers is not None:
        config['workers'] = args.workers
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, work_dir / 'mmseg_student_config.py')
    (work_dir / 'effective_training_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    common.seed_everything(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    supervised_dataset = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_train_manifest.csv',
        training=True, pseudo=False, augmentation=config['augmentation'])
    validation_dataset = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_val_manifest.csv',
        training=False)
    unlabeled_dataset = MeanTeacherUnlabeledDataset(
        args.unlabeled_manifest, config['augmentation'],
        config['mean_teacher']['strong_color_jitter'])
    supervised_loader = make_loader(
        supervised_dataset, config['supervised_batch_size'], True, config,
        generator, True)
    unlabeled_loader = make_loader(
        unlabeled_dataset, config['pseudo_batch_size'], True, config,
        generator, True)
    validation_loader = make_loader(
        validation_dataset, config['validation_batch_size'], False, config,
        generator, False)
    student = common.build_student(args.model_config, device)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_epochs'],
        eta_min=config['minimum_learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    class_weights = torch.tensor(
        config['class_weights'], dtype=torch.float32, device=device)
    history = []
    best_miou = -1.0
    best_epoch = 0
    best_metrics = None
    best_path = work_dir / 'best_validation_mIoU.pth'
    started = time.time()
    for epoch in range(1, config['max_epochs'] + 1):
        lambda_u = common.lambda_u_for_epoch(epoch, config['max_epochs'])
        row = {
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'lambda_u': lambda_u,
            **train_epoch(
                student, teacher, supervised_loader, unlabeled_loader,
                optimizer, scaler, device, class_weights, lambda_u, config),
        }
        validation, _ = common.evaluate(
            teacher, validation_loader, device, class_weights)
        row.update({
            f'val_{key}': value for key, value in validation.items()
            if key != 'confusion_matrix'
        })
        history.append(row)
        common.write_csv(work_dir / 'epoch_metrics.csv', history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation['mIoU'] > best_miou:
            best_miou = validation['mIoU']
            best_epoch = epoch
            best_metrics = validation
            common.save_checkpoint(
                best_path, teacher, epoch, validation,
                'DeepLabV3+ MobileNetV3-Large Mean Teacher EMA')
        common.save_checkpoint(
            work_dir / 'last_checkpoint.pth', teacher, epoch, validation,
            'DeepLabV3+ MobileNetV3-Large Mean Teacher EMA')
        scheduler.step()
    common.plot_history(history, work_dir, best_epoch)
    summary = {
        'group': 'C',
        'model': 'DeepLabV3+ MobileNetV3-Large',
        'method': 'EMA Mean Teacher consistency baseline',
        'ema_decay': config['mean_teacher']['ema_decay'],
        'confidence_threshold': config['mean_teacher']['confidence_threshold'],
        'supervised_train_images': len(supervised_dataset),
        'unlabeled_train_images': len(unlabeled_dataset),
        'validation_images': len(validation_dataset),
        'epochs_completed': len(history),
        'steps_per_epoch': config['steps_per_epoch'],
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'validation_model': 'EMA teacher',
        'training_elapsed_seconds': time.time() - started,
        'model_selection': 'validation mIoU only',
        'test_split_loaded': False,
    }
    (work_dir / 'training_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

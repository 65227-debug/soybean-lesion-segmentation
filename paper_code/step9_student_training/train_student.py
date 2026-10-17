#!/usr/bin/env python
"""Train DeepLabV3+ MobileNetV3 with supervised and weighted pseudo labels."""

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ('background', 'healthy_leaf', 'lesion')
PALETTE_BGR = np.asarray(
    [[0, 0, 0], [0, 180, 0], [30, 30, 220]], dtype=np.uint8)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/dadou_student')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument(
        '--training-config',
        default='paper_code/step9_student_training/training_config.json')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/dadou_student_deeplabv3plus_mobilenetv3')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class StudentSegmentationDataset(Dataset):
    def __init__(self, manifest, training=False, pseudo=False,
                 augmentation=None):
        self.rows = read_manifest(manifest)
        self.training = training
        self.pseudo = pseudo
        self.augmentation = augmentation or {}

    def __len__(self):
        return len(self.rows)

    def _augment(self, image, mask, weight):
        if random.random() < self.augmentation['horizontal_flip_probability']:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            weight = np.flip(weight, axis=1)
        if random.random() < self.augmentation['vertical_flip_probability']:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
            weight = np.flip(weight, axis=0)
        rotation = self.augmentation['rotation_degrees']
        if rotation > 0:
            angle = random.uniform(-rotation, rotation)
            height, width = mask.shape
            matrix = cv2.getRotationMatrix2D(
                (width / 2.0, height / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = cv2.warpAffine(
                mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255 if self.pseudo else 0)
            weight = cv2.warpAffine(
                weight, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        image = image.astype(np.float32) / 255.0
        brightness = 1.0 + random.uniform(
            -self.augmentation['brightness_jitter'],
            self.augmentation['brightness_jitter'])
        contrast = 1.0 + random.uniform(
            -self.augmentation['contrast_jitter'],
            self.augmentation['contrast_jitter'])
        saturation = 1.0 + random.uniform(
            -self.augmentation['saturation_jitter'],
            self.augmentation['saturation_jitter'])
        image *= brightness
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast + mean
        gray = cv2.cvtColor(
            np.clip(image, 0, 1), cv2.COLOR_RGB2GRAY)[..., None]
        image = gray + saturation * (image - gray)
        return (
            np.clip(image, 0, 1),
            np.ascontiguousarray(mask),
            np.ascontiguousarray(weight))

    def __getitem__(self, index):
        row = self.rows[index]
        image_bgr = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        if image_bgr is None or mask is None:
            raise ValueError(
                f"Unreadable sample: {row['image_path']}, {row['mask_path']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.pseudo:
            weight_u16 = cv2.imread(row['weight_path'], cv2.IMREAD_UNCHANGED)
            if weight_u16 is None:
                raise ValueError(f"Unreadable weight: {row['weight_path']}")
            weight = weight_u16.astype(np.float32) / 65535.0
        else:
            weight = np.ones(mask.shape, dtype=np.float32)
        if self.training:
            image, mask, weight = self._augment(image, mask, weight)
        else:
            image = image.astype(np.float32) / 255.0
        weight[mask == 255] = 0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return {
            'image': image_tensor,
            'mask': torch.from_numpy(mask.astype(np.int64)),
            'weight': torch.from_numpy(weight.astype(np.float32)),
            'sample_id': row['sample_id'],
            'image_path': row['image_path'],
        }


def build_student(model_config, device):
    init_default_scope('mmseg')
    config = Config.fromfile(model_config)
    model = MODELS.build(config.model)
    model.init_weights()
    model.to(device)
    return model


def forward_logits(model, images, output_size):
    features = model.extract_feat(images)
    logits = model.decode_head.forward(features)
    return F.interpolate(
        logits, size=output_size, mode='bilinear', align_corners=False)


def soft_dice_loss(logits, targets, ignore_index=255, epsilon=1e-6):
    valid = targets != ignore_index
    safe_targets = targets.clone()
    safe_targets[~valid] = 0
    one_hot = F.one_hot(
        safe_targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probabilities = logits.softmax(dim=1) * valid_float
    one_hot *= valid_float
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def weighted_pseudo_ce(logits, targets, pixel_weights, class_weights,
                       ignore_index=255):
    per_pixel = F.cross_entropy(
        logits, targets, weight=class_weights, ignore_index=ignore_index,
        reduction='none')
    valid_weights = pixel_weights * (targets != ignore_index).float()
    return (per_pixel * valid_weights).sum() / (valid_weights.sum() + 1e-6)


def lambda_u_for_epoch(epoch, max_epochs):
    if epoch <= 10:
        return 0.2 * (epoch - 1) / 9.0
    if epoch <= 30:
        return 0.2 + 0.3 * (epoch - 10) / 20.0
    if max_epochs <= 30:
        return 0.5
    return 0.5 + 0.2 * (epoch - 30) / (max_epochs - 30)


def update_confusion(confusion, prediction, target, num_classes=3):
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + prediction[valid]
    confusion += torch.bincount(
        encoded, minlength=num_classes ** 2).reshape(
            num_classes, num_classes).cpu().numpy()


def metrics_from_confusion(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - true_positive
    iou = np.divide(
        true_positive, union, out=np.full_like(true_positive, np.nan),
        where=union > 0)
    dice = np.divide(
        2 * true_positive, ground_truth + predicted,
        out=np.full_like(true_positive, np.nan),
        where=(ground_truth + predicted) > 0)
    precision = np.divide(
        true_positive, predicted, out=np.full_like(true_positive, np.nan),
        where=predicted > 0)
    recall = np.divide(
        true_positive, ground_truth, out=np.full_like(true_positive, np.nan),
        where=ground_truth > 0)
    result = {
        'mIoU': float(np.nanmean(iou)),
        'mDice': float(np.nanmean(dice)),
        'pixel_accuracy': float(true_positive.sum() / confusion.sum()),
    }
    for index, name in enumerate(CLASS_NAMES):
        result[f'IoU_{name}'] = float(iou[index])
        result[f'Dice_{name}'] = float(dice[index])
        result[f'Precision_{name}'] = float(precision[index])
        result[f'Recall_{name}'] = float(recall[index])
    return result


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_epoch(model, supervised_loader, pseudo_loader, optimizer, scaler,
                    device, class_weights, lambda_u, gradient_clip_norm):
    model.train()
    supervised_iterator = iter(supervised_loader)
    pseudo_iterator = iter(pseudo_loader)
    steps = max(len(supervised_loader), len(pseudo_loader))
    totals = {
        'supervised_ce': 0.0,
        'supervised_dice': 0.0,
        'supervised_loss': 0.0,
        'pseudo_loss': 0.0,
        'total_loss': 0.0,
        'pseudo_mean_weight': 0.0,
        'pseudo_accepted_pixel_percent': 0.0,
    }
    confusion = np.zeros((3, 3), dtype=np.int64)
    for _ in range(steps):
        supervised, supervised_iterator = next_or_restart(
            supervised_iterator, supervised_loader)
        pseudo, pseudo_iterator = next_or_restart(
            pseudo_iterator, pseudo_loader)
        supervised_images = supervised['image'].to(device, non_blocking=True)
        supervised_masks = supervised['mask'].to(device, non_blocking=True)
        pseudo_images = pseudo['image'].to(device, non_blocking=True)
        pseudo_masks = pseudo['mask'].to(device, non_blocking=True)
        pseudo_weights = pseudo['weight'].to(device, non_blocking=True)
        combined_images = torch.cat([supervised_images, pseudo_images], dim=0)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            combined_logits = forward_logits(
                model, combined_images, supervised_masks.shape[-2:])
            supervised_logits = combined_logits[:supervised_images.shape[0]]
            pseudo_logits = combined_logits[supervised_images.shape[0]:]
            supervised_ce = F.cross_entropy(
                supervised_logits, supervised_masks, weight=class_weights)
            supervised_dice = soft_dice_loss(
                supervised_logits, supervised_masks)
            supervised_loss = supervised_ce + supervised_dice
            pseudo_loss = weighted_pseudo_ce(
                pseudo_logits, pseudo_masks, pseudo_weights, class_weights)
            total_loss = supervised_loss + lambda_u * pseudo_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        totals['supervised_ce'] += supervised_ce.item()
        totals['supervised_dice'] += supervised_dice.item()
        totals['supervised_loss'] += supervised_loss.item()
        totals['pseudo_loss'] += pseudo_loss.item()
        totals['total_loss'] += total_loss.item()
        valid_pseudo = pseudo_masks != 255
        totals['pseudo_mean_weight'] += float(
            pseudo_weights[valid_pseudo].mean().item())
        totals['pseudo_accepted_pixel_percent'] += float(
            100 * valid_pseudo.float().mean().item())
        prediction = supervised_logits.detach().argmax(dim=1)
        update_confusion(confusion, prediction, supervised_masks)
    result = {key: value / steps for key, value in totals.items()}
    result.update({
        f'train_{key}': value
        for key, value in metrics_from_confusion(confusion).items()
    })
    return result


@torch.no_grad()
def evaluate(model, loader, device, class_weights, save_dir=None):
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    losses = []
    per_image_rows = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / 'masks').mkdir(parents=True, exist_ok=True)
        (save_dir / 'overlays').mkdir(parents=True, exist_ok=True)
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)
        with torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            logits = forward_logits(model, images, targets.shape[-2:])
            loss = (
                F.cross_entropy(logits, targets, weight=class_weights)
                + soft_dice_loss(logits, targets))
        losses.append(loss.item())
        predictions = logits.argmax(dim=1)
        update_confusion(confusion, predictions, targets)
        for index in range(predictions.shape[0]):
            single_confusion = np.zeros((3, 3), dtype=np.int64)
            update_confusion(
                single_confusion, predictions[index], targets[index])
            metrics = metrics_from_confusion(single_confusion)
            per_image_rows.append({
                'sample_id': batch['sample_id'][index],
                **metrics,
            })
            if save_dir is not None:
                prediction = predictions[index].cpu().numpy().astype(np.uint8)
                stem = batch['sample_id'][index]
                cv2.imwrite(str(save_dir / 'masks' / f'{stem}.png'), prediction)
                original = cv2.imread(
                    batch['image_path'][index], cv2.IMREAD_COLOR)
                color = PALETTE_BGR[prediction]
                overlay = cv2.addWeighted(original, 0.65, color, 0.35, 0)
                cv2.imwrite(
                    str(save_dir / 'overlays' / f'{stem}.jpg'), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    metrics = metrics_from_confusion(confusion)
    metrics['loss'] = float(np.mean(losses))
    metrics['confusion_matrix'] = confusion.tolist()
    return metrics, per_image_rows


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, epoch, metrics, architecture):
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'meta': {
            'architecture': architecture,
            'classes': CLASS_NAMES,
            'palette_bgr': PALETTE_BGR.tolist(),
            'input_size': [512, 512],
            'ignore_index': 255,
        },
    }, path)


def plot_history(rows, work_dir, best_epoch):
    epochs = [row['epoch'] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=180)
    axes[0].plot(epochs, [row['supervised_loss'] for row in rows],
                 label='Supervised loss')
    axes[0].plot(epochs, [row['pseudo_loss'] for row in rows],
                 label='Pseudo loss')
    axes[0].plot(epochs, [row['total_loss'] for row in rows],
                 label='Total loss')
    axes[0].set(title='Training losses', xlabel='Epoch', ylabel='Loss')
    axes[1].plot(epochs, [100 * row['val_mIoU'] for row in rows],
                 label='Validation mIoU')
    axes[1].plot(epochs, [100 * row['val_IoU_lesion'] for row in rows],
                 label='Validation lesion IoU')
    axes[1].plot(epochs, [100 * row['val_mDice'] for row in rows],
                 label='Validation mDice')
    axes[1].set(title='Validation metrics', xlabel='Epoch', ylabel='Score (%)')
    axes[2].plot(epochs, [row['lambda_u'] for row in rows], label='lambda_u')
    axes[2].plot(epochs, [row['learning_rate'] for row in rows],
                 label='Learning rate')
    axes[2].set(title='Schedules', xlabel='Epoch', ylabel='Value')
    for axis in axes:
        axis.axvline(
            best_epoch, color='black', linestyle='--', linewidth=1,
            label=f'Best epoch ({best_epoch})')
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ('.png', '.pdf'):
        figure.savefig(
            Path(work_dir) / f'training_curves{suffix}', bbox_inches='tight')
    plt.close(figure)


def make_loaders(data_root, config, generator):
    augmentation = config['augmentation']
    supervised_train = StudentSegmentationDataset(
        Path(data_root) / 'supervised_train_manifest.csv', training=True,
        pseudo=False, augmentation=augmentation)
    pseudo_train = StudentSegmentationDataset(
        Path(data_root) / 'pseudo_train_manifest.csv', training=True,
        pseudo=True, augmentation=augmentation)
    validation = StudentSegmentationDataset(
        Path(data_root) / 'supervised_val_manifest.csv', training=False)
    common = dict(
        num_workers=config['workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['workers'] > 0,
        worker_init_fn=seed_worker,
        generator=generator)
    supervised_loader = DataLoader(
        supervised_train, batch_size=config['supervised_batch_size'],
        shuffle=True, drop_last=True, **common)
    pseudo_loader = DataLoader(
        pseudo_train, batch_size=config['pseudo_batch_size'],
        shuffle=True, drop_last=True, **common)
    validation_loader = DataLoader(
        validation, batch_size=config['validation_batch_size'],
        shuffle=False, drop_last=False, **common)
    return supervised_loader, pseudo_loader, validation_loader


def main():
    args = parse_args()
    training_config_path = Path(args.training_config)
    config = json.loads(training_config_path.read_text(encoding='utf-8'))
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.workers is not None:
        config['workers'] = args.workers
    if args.smoke_test:
        config['max_epochs'] = 1
        config['workers'] = min(config['workers'], 2)
        config['early_stopping_patience'] = 1
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, work_dir / 'mmseg_student_config.py')
    (work_dir / 'effective_training_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    seed_everything(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    supervised_loader, pseudo_loader, validation_loader = make_loaders(
        args.data_root, config, generator)
    model = build_student(args.model_config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_epochs'],
        eta_min=config['minimum_learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    class_weights = torch.tensor(
        config['class_weights'], dtype=torch.float32, device=device)

    initial_metrics, _ = evaluate(
        model, validation_loader, device, class_weights)
    (work_dir / 'initial_validation_metrics.json').write_text(
        json.dumps(initial_metrics, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps({'epoch': 0, 'validation': initial_metrics},
                     ensure_ascii=False), flush=True)

    history = []
    best_miou = -1.0
    best_epoch = 0
    best_metrics = None
    patience_anchor = -1.0
    epochs_without_meaningful_improvement = 0
    best_path = work_dir / 'best_validation_mIoU.pth'
    started = time.time()
    for epoch in range(1, config['max_epochs'] + 1):
        lambda_u = lambda_u_for_epoch(epoch, config['max_epochs'])
        learning_rate = optimizer.param_groups[0]['lr']
        train_metrics = train_one_epoch(
            model, supervised_loader, pseudo_loader, optimizer, scaler,
            device, class_weights, lambda_u, config['gradient_clip_norm'])
        validation_metrics, _ = evaluate(
            model, validation_loader, device, class_weights)
        row = {
            'epoch': epoch,
            'learning_rate': learning_rate,
            'lambda_u': lambda_u,
            **train_metrics,
            **{f'val_{key}': value for key, value in validation_metrics.items()
               if key != 'confusion_matrix'},
        }
        history.append(row)
        write_csv(work_dir / 'epoch_metrics.csv', history)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        current_miou = validation_metrics['mIoU']
        if current_miou > best_miou:
            best_miou = current_miou
            best_epoch = epoch
            best_metrics = validation_metrics
            save_checkpoint(
                best_path, model, epoch, validation_metrics,
                'DeepLabV3+ MobileNetV3-Large')
        if (current_miou >= patience_anchor
                + config['early_stopping_min_delta']):
            patience_anchor = current_miou
            epochs_without_meaningful_improvement = 0
        else:
            epochs_without_meaningful_improvement += 1
        save_checkpoint(
            work_dir / 'last_checkpoint.pth', model, epoch,
            validation_metrics, 'DeepLabV3+ MobileNetV3-Large')
        scheduler.step()
        if (epochs_without_meaningful_improvement
                >= config['early_stopping_patience']):
            print(f'Early stopping at epoch {epoch}', flush=True)
            break

    elapsed = time.time() - started
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test_dataset = None
    test_metrics = None
    if not args.smoke_test:
        test_dataset = StudentSegmentationDataset(
            Path(args.data_root) / 'supervised_test_manifest.csv',
            training=False)
        test_loader = DataLoader(
            test_dataset, batch_size=config['validation_batch_size'],
            shuffle=False, num_workers=config['workers'],
            pin_memory=device.type == 'cuda',
            persistent_workers=config['workers'] > 0,
            worker_init_fn=seed_worker)
        test_metrics, test_per_image = evaluate(
            model, test_loader, device, class_weights,
            save_dir=work_dir / 'test_predictions')
        write_csv(work_dir / 'test_per_image_metrics.csv', test_per_image)
    plot_history(history, work_dir, best_epoch)

    summary = {
        'model': 'DeepLabV3+ MobileNetV3-Large',
        'mobilenetv3_imagenet_pretrained': True,
        'parameters': parameter_count,
        'input_size': [config['image_size'], config['image_size']],
        'training_method': 'fixed teacher, dual-evidence weighted pseudo labels',
        'supervised_train_images': len(supervised_loader.dataset),
        'pseudo_train_images': len(pseudo_loader.dataset),
        'validation_images': len(validation_loader.dataset),
        'test_images': len(test_dataset) if test_dataset is not None else 0,
        'epochs_completed': len(history),
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'test_metrics': test_metrics,
        'lambda_u_final_reached': history[-1]['lambda_u'],
        'training_elapsed_seconds': elapsed,
        'model_selection': 'validation mIoU only',
        'test_split_loaded_after_best_checkpoint_selection': not args.smoke_test,
        'test_split_used_for_training_or_model_selection': False,
        'smoke_test': args.smoke_test,
    }
    (work_dir / 'training_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    test_report = (
        f"- 测试mIoU：{100 * test_metrics['mIoU']:.2f}%\n"
        f"- 测试病斑IoU：{100 * test_metrics['IoU_lesion']:.2f}%\n"
        f"- 测试mDice：{100 * test_metrics['mDice']:.2f}%"
        if test_metrics is not None else '- 冒烟测试未加载最终测试集')
    report = f"""# 第九步学生模型训练结果

- 模型：DeepLabV3+ + MobileNetV3-Large
- 初始化：MobileNetV3 ImageNet预训练，DeepLabV3+解码头随机初始化
- 人工精标训练图：{len(supervised_loader.dataset)}张
- 双证据门控伪标签：{len(pseudo_loader.dataset)}张
- 验证图：{len(validation_loader.dataset)}张
- 最终测试图：{len(test_dataset) if test_dataset is not None else 0}张（只在最佳权重确定后加载）
- 最佳epoch：{best_epoch}
- 验证mIoU：{100 * best_metrics['mIoU']:.2f}%
- 验证病斑IoU：{100 * best_metrics['IoU_lesion']:.2f}%
{test_report}
- 参数量：{parameter_count / 1e6:.3f}M

损失定义：`L = L_sup_CE + L_sup_Dice + lambda_u(t) * L_pseudo_weighted_CE`。
模型仅按验证mIoU选择，测试集没有参与早停、阈值或超参数选择。
"""
    (work_dir / '论文训练结果.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

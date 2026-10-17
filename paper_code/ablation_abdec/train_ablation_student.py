#!/usr/bin/env python
"""Train A/B/D/E students with a common architecture and update budget."""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


STEP9_DIR = Path(__file__).resolve().parents[1] / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('A', 'B', 'D', 'E'))
    parser.add_argument('--supervised-root', default='data/dadou_student')
    parser.add_argument('--pseudo-root', default='data/dadou_abdec')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument(
        '--training-config',
        default='paper_code/ablation_abdec/ablation_config.json')
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, config, generator, drop_last):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=config['workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['workers'] > 0,
        worker_init_fn=common.seed_worker, generator=generator)


def make_loaders(args, config, generator):
    augmentation = config['augmentation']
    supervised = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_train_manifest.csv',
        training=True, pseudo=False, augmentation=augmentation)
    validation = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_val_manifest.csv',
        training=False)
    supervised_loader = make_loader(
        supervised, config['supervised_batch_size'], True, config,
        generator, True)
    validation_loader = make_loader(
        validation, config['validation_batch_size'], False, config,
        generator, False)
    pseudo_loader = None
    if args.mode != 'A':
        pseudo = common.StudentSegmentationDataset(
            Path(args.pseudo_root) / f'pseudo_{args.mode}_manifest.csv',
            training=True, pseudo=True, augmentation=augmentation)
        pseudo_loader = make_loader(
            pseudo, config['pseudo_batch_size'], True, config,
            generator, True)
    return supervised_loader, pseudo_loader, validation_loader


def train_epoch(model, supervised_loader, pseudo_loader, optimizer, scaler,
                device, class_weights, lambda_u, config):
    model.train()
    supervised_iterator = iter(supervised_loader)
    pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None
    totals = {
        'supervised_ce': 0.0, 'supervised_dice': 0.0,
        'supervised_loss': 0.0, 'pseudo_loss': 0.0,
        'total_loss': 0.0, 'pseudo_mean_weight': 0.0,
        'pseudo_accepted_pixel_percent': 0.0,
    }
    confusion = np.zeros((3, 3), dtype=np.int64)
    steps = config['steps_per_epoch']
    for _ in range(steps):
        supervised, supervised_iterator = common.next_or_restart(
            supervised_iterator, supervised_loader)
        supervised_images = supervised['image'].to(device, non_blocking=True)
        supervised_masks = supervised['mask'].to(device, non_blocking=True)
        pseudo = None
        if pseudo_loader is not None:
            pseudo, pseudo_iterator = common.next_or_restart(
                pseudo_iterator, pseudo_loader)
            pseudo_images = pseudo['image'].to(device, non_blocking=True)
            pseudo_masks = pseudo['mask'].to(device, non_blocking=True)
            pseudo_weights = pseudo['weight'].to(device, non_blocking=True)
            combined_images = torch.cat(
                [supervised_images, pseudo_images], dim=0)
        else:
            combined_images = supervised_images
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            combined_logits = common.forward_logits(
                model, combined_images, supervised_masks.shape[-2:])
            supervised_logits = combined_logits[:supervised_images.shape[0]]
            supervised_ce = F.cross_entropy(
                supervised_logits, supervised_masks, weight=class_weights)
            supervised_dice = common.soft_dice_loss(
                supervised_logits, supervised_masks)
            supervised_loss = supervised_ce + supervised_dice
            if pseudo is not None:
                pseudo_logits = combined_logits[supervised_images.shape[0]:]
                pseudo_loss = common.weighted_pseudo_ce(
                    pseudo_logits, pseudo_masks, pseudo_weights,
                    class_weights)
            else:
                pseudo_loss = supervised_loss.new_zeros(())
            total_loss = supervised_loss + lambda_u * pseudo_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config['gradient_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        totals['supervised_ce'] += supervised_ce.item()
        totals['supervised_dice'] += supervised_dice.item()
        totals['supervised_loss'] += supervised_loss.item()
        totals['pseudo_loss'] += pseudo_loss.item()
        totals['total_loss'] += total_loss.item()
        if pseudo is not None:
            valid = pseudo_masks != 255
            totals['pseudo_mean_weight'] += float(
                pseudo_weights[valid].mean().item())
            totals['pseudo_accepted_pixel_percent'] += float(
                100 * valid.float().mean().item())
        common.update_confusion(
            confusion, supervised_logits.detach().argmax(dim=1),
            supervised_masks)
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
    if args.seed is not None:
        config['seed'] = args.seed
    work_dir = Path(args.work_dir or f'work_dirs/dadou_ablation_{args.mode}')
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, work_dir / 'mmseg_student_config.py')
    (work_dir / 'effective_training_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    common.seed_everything(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    supervised_loader, pseudo_loader, validation_loader = make_loaders(
        args, config, generator)
    model = common.build_student(args.model_config, device)
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
    history = []
    best_miou = -1.0
    best_epoch = 0
    best_metrics = None
    best_path = work_dir / 'best_validation_mIoU.pth'
    started = time.time()
    for epoch in range(1, config['max_epochs'] + 1):
        lambda_u = (
            0.0 if args.mode == 'A'
            else common.lambda_u_for_epoch(epoch, config['max_epochs']))
        row = {
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'lambda_u': lambda_u,
            **train_epoch(
                model, supervised_loader, pseudo_loader, optimizer, scaler,
                device, class_weights, lambda_u, config),
        }
        validation, _ = common.evaluate(
            model, validation_loader, device, class_weights)
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
                best_path, model, epoch, validation,
                f'DeepLabV3+ MobileNetV3-Large ablation {args.mode}')
        common.save_checkpoint(
            work_dir / 'last_checkpoint.pth', model, epoch, validation,
            f'DeepLabV3+ MobileNetV3-Large ablation {args.mode}')
        scheduler.step()
    common.plot_history(history, work_dir, best_epoch)
    summary = {
        'group': args.mode,
        'model': 'DeepLabV3+ MobileNetV3-Large',
        'parameters': parameter_count,
        'supervised_train_images': len(supervised_loader.dataset),
        'pseudo_train_images': (
            len(pseudo_loader.dataset) if pseudo_loader is not None else 0),
        'validation_images': len(validation_loader.dataset),
        'epochs_completed': len(history),
        'steps_per_epoch': config['steps_per_epoch'],
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'training_elapsed_seconds': time.time() - started,
        'model_selection': 'validation mIoU only',
        'test_split_loaded': False,
    }
    (work_dir / 'training_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

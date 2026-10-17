#!/usr/bin/env python
"""Train corrected F with isolated BN and scene/class-balanced pseudo loss."""

import argparse
import contextlib
import csv
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, WeightedRandomSampler


STEP9_DIR = Path(__file__).resolve().parents[1] / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervised-root', default='data/dadou_student')
    parser.add_argument('--pseudo-root', required=True)
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument(
        '--training-config',
        default='paper_code/corrected_f_training/training_config.json')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--maximum-lambda-u', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--init-checkpoint', default=None,
        help='Optional supervised checkpoint for two-stage SSL fine-tuning.')
    parser.add_argument(
        '--skip-test', action='store_true',
        help='Do not load images or calculate test metrics (for smoke checks).')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def make_loader(dataset, batch_size, shuffle, workers, generator, drop_last,
                sampler=None):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler, drop_last=drop_last, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        worker_init_fn=common.seed_worker, generator=generator)


def make_loaders(args, config):
    augmentation = config['augmentation']
    supervised = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_train_manifest.csv',
        training=True, pseudo=False, augmentation=augmentation)
    pseudo_manifest = Path(args.pseudo_root) / 'pseudo_train_manifest.csv'
    pseudo = common.StudentSegmentationDataset(
        pseudo_manifest, training=True, pseudo=True, augmentation=augmentation)
    validation = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_val_manifest.csv',
        training=False)
    test = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_test_manifest.csv',
        training=False)

    pseudo_rows = read_manifest(pseudo_manifest)
    scene_counts = Counter(
        row.get('scene_id') or row['sample_id'] for row in pseudo_rows)
    sampling_weights = [
        1.0 / scene_counts[row.get('scene_id') or row['sample_id']]
        for row in pseudo_rows]
    sampler_generator = torch.Generator().manual_seed(config['seed'] + 101)
    sampler = WeightedRandomSampler(
        sampling_weights,
        num_samples=config['steps_per_epoch'] * config['pseudo_batch_size'],
        replacement=True, generator=sampler_generator)

    supervised_loader = make_loader(
        supervised, config['supervised_batch_size'], True, config['workers'],
        torch.Generator().manual_seed(config['seed']), True)
    pseudo_loader = make_loader(
        pseudo, config['pseudo_batch_size'], False, config['workers'],
        torch.Generator().manual_seed(config['seed'] + 202), True,
        sampler=sampler)
    validation_loader = make_loader(
        validation, config['validation_batch_size'], False,
        config['workers'], torch.Generator().manual_seed(config['seed'] + 303),
        False)
    test_loader = make_loader(
        test, config['validation_batch_size'], False, config['workers'],
        torch.Generator().manual_seed(config['seed'] + 404), False)
    return (supervised_loader, pseudo_loader, validation_loader, test_loader,
            len(scene_counts), max(scene_counts.values()))


@contextlib.contextmanager
def frozen_batch_norm_statistics(model):
    """Use stored BN statistics for pseudo images without freezing gradients."""
    modules = [module for module in model.modules()
               if isinstance(module, _BatchNorm)]
    previous = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        yield
    finally:
        for module, training in zip(modules, previous):
            module.train(training)


def balanced_pseudo_ce(logits, targets, pixel_weights, coefficients):
    per_pixel = F.cross_entropy(
        logits, targets, ignore_index=255, reduction='none')
    losses = []
    active_coefficients = []
    class_values = []
    for class_index, coefficient in enumerate(coefficients):
        mask = targets == class_index
        weights = pixel_weights * mask.float()
        denominator = weights.sum()
        if denominator.item() > 0:
            value = (per_pixel * weights).sum() / (denominator + 1e-6)
            losses.append(coefficient * value)
            active_coefficients.append(coefficient)
            class_values.append(float(value.detach().item()))
        else:
            class_values.append(0.0)
    if not losses:
        return logits.sum() * 0.0, class_values
    return sum(losses) / sum(active_coefficients), class_values


def lambda_for_epoch(epoch, config):
    warmup = config['supervised_warmup_epochs']
    ramp_end = config['unsupervised_ramp_end_epoch']
    maximum = config['maximum_lambda_u']
    if epoch <= warmup:
        return 0.0
    if epoch >= ramp_end:
        return maximum
    return maximum * (epoch - warmup) / (ramp_end - warmup)


def train_epoch(model, supervised_loader, pseudo_loader, optimizer, scaler,
                device, class_weights, pseudo_coefficients, lambda_u, config):
    model.train()
    supervised_iterator = iter(supervised_loader)
    pseudo_iterator = iter(pseudo_loader) if lambda_u > 0 else None
    totals = {
        'supervised_ce': 0.0, 'supervised_dice': 0.0,
        'supervised_loss': 0.0, 'pseudo_loss': 0.0, 'total_loss': 0.0,
        'pseudo_mean_weight': 0.0, 'pseudo_accepted_pixel_percent': 0.0,
        'pseudo_loss_background': 0.0, 'pseudo_loss_healthy': 0.0,
        'pseudo_loss_lesion': 0.0,
    }
    confusion = np.zeros((3, 3), dtype=np.int64)
    for _ in range(config['steps_per_epoch']):
        supervised, supervised_iterator = common.next_or_restart(
            supervised_iterator, supervised_loader)
        supervised_images = supervised['image'].to(device, non_blocking=True)
        supervised_masks = supervised['mask'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            supervised_bn_context = (
                frozen_batch_norm_statistics(model)
                if config.get('freeze_all_batch_norm', False)
                else contextlib.nullcontext())
            with supervised_bn_context:
                supervised_logits = common.forward_logits(
                    model, supervised_images, supervised_masks.shape[-2:])
            supervised_ce = F.cross_entropy(
                supervised_logits, supervised_masks, weight=class_weights)
            supervised_dice = common.soft_dice_loss(
                supervised_logits, supervised_masks)
            supervised_loss = supervised_ce + supervised_dice
            pseudo_loss = supervised_loss.new_zeros(())
            pseudo_class_losses = [0.0, 0.0, 0.0]
            pseudo = None
            if lambda_u > 0:
                pseudo, pseudo_iterator = common.next_or_restart(
                    pseudo_iterator, pseudo_loader)
                pseudo_images = pseudo['image'].to(device, non_blocking=True)
                pseudo_masks = pseudo['mask'].to(device, non_blocking=True)
                pseudo_weights = pseudo['weight'].to(device, non_blocking=True)
                with frozen_batch_norm_statistics(model):
                    pseudo_logits = common.forward_logits(
                        model, pseudo_images, pseudo_masks.shape[-2:])
                pseudo_loss, pseudo_class_losses = balanced_pseudo_ce(
                    pseudo_logits, pseudo_masks, pseudo_weights,
                    pseudo_coefficients)
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
        for name, value in zip(
                ('pseudo_loss_background', 'pseudo_loss_healthy',
                 'pseudo_loss_lesion'), pseudo_class_losses):
            totals[name] += value
        if pseudo is not None:
            valid = pseudo_masks != 255
            totals['pseudo_mean_weight'] += float(
                pseudo_weights[valid].mean().item())
            totals['pseudo_accepted_pixel_percent'] += float(
                100.0 * valid.float().mean().item())
        common.update_confusion(
            confusion, supervised_logits.detach().argmax(dim=1),
            supervised_masks)
    result = {
        key: value / config['steps_per_epoch']
        for key, value in totals.items()}
    result.update({
        f'train_{key}': value
        for key, value in common.metrics_from_confusion(confusion).items()})
    return result


def main():
    args = parse_args()
    config = json.loads(Path(args.training_config).read_text(encoding='utf-8'))
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.workers is not None:
        config['workers'] = args.workers
    if args.maximum_lambda_u is not None:
        config['maximum_lambda_u'] = args.maximum_lambda_u
    if args.seed is not None:
        config['seed'] = args.seed
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, work_dir / 'mmseg_student_config.py')
    (work_dir / 'effective_training_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    common.seed_everything(config['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    (supervised_loader, pseudo_loader, validation_loader, test_loader,
     pseudo_scenes, maximum_scene_multiplicity) = make_loaders(args, config)
    model = common.build_student(args.model_config, device)
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initialization['state_dict'])
    if config.get('train_classifier_only', False):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.decode_head.conv_seg.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_epochs'],
        eta_min=config['minimum_learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    class_weights = torch.tensor(
        config['class_weights'], dtype=torch.float32, device=device)
    pseudo_coefficients = config['pseudo_class_coefficients']

    initial, _ = common.evaluate(
        model, validation_loader, device, class_weights)
    (work_dir / 'initial_validation_metrics.json').write_text(
        json.dumps(initial, ensure_ascii=False, indent=2), encoding='utf-8')
    history = []
    best_miou = initial['mIoU'] if args.init_checkpoint else -1.0
    best_epoch = 0
    best_metrics = initial if args.init_checkpoint else None
    best_path = work_dir / 'best_validation_mIoU.pth'
    if args.init_checkpoint:
        common.save_checkpoint(
            best_path, model, 0, initial,
            'DeepLabV3+ MobileNetV3-Large corrected F initialized from A')
    started = time.time()
    for epoch in range(1, config['max_epochs'] + 1):
        lambda_u = lambda_for_epoch(epoch, config)
        row = {
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'lambda_u': lambda_u,
            **train_epoch(
                model, supervised_loader, pseudo_loader, optimizer, scaler,
                device, class_weights, pseudo_coefficients, lambda_u, config),
        }
        validation, _ = common.evaluate(
            model, validation_loader, device, class_weights)
        row.update({
            f'val_{key}': value for key, value in validation.items()
            if key != 'confusion_matrix'})
        history.append(row)
        common.write_csv(work_dir / 'epoch_metrics.csv', history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation['mIoU'] > best_miou:
            best_miou = validation['mIoU']
            best_epoch = epoch
            best_metrics = validation
            common.save_checkpoint(
                best_path, model, epoch, validation,
                'DeepLabV3+ MobileNetV3-Large corrected F')
        common.save_checkpoint(
            work_dir / 'last_checkpoint.pth', model, epoch, validation,
            'DeepLabV3+ MobileNetV3-Large corrected F')
        scheduler.step()
    common.plot_history(history, work_dir, best_epoch)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test_metrics = None
    if not args.skip_test:
        test_metrics, test_rows = common.evaluate(
            model, test_loader, device, class_weights,
            save_dir=work_dir / 'test_predictions')
        common.write_csv(work_dir / 'test_per_image_metrics.csv', test_rows)
    summary = {
        'method': 'corrected F',
        'initialization_checkpoint': args.init_checkpoint,
        'model': 'DeepLabV3+ MobileNetV3-Large',
        'supervised_train_images': len(supervised_loader.dataset),
        'pseudo_train_images': len(pseudo_loader.dataset),
        'pseudo_unique_scenes': pseudo_scenes,
        'maximum_images_from_one_scene_before_balanced_sampling': (
            maximum_scene_multiplicity),
        'batch_norm_policy': (
            'frozen for all forwards' if config.get(
                'freeze_all_batch_norm', False)
            else 'updated by supervised images only; frozen for pseudo forward'),
        'trainable_scope': (
            'decode_head.conv_seg only' if config.get(
                'train_classifier_only', False) else 'full network'),
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'test_metrics': test_metrics,
        'training_elapsed_seconds': time.time() - started,
        'test_split_loaded_after_validation_selection': not args.skip_test,
        'test_split_used_for_training_or_selection': False,
    }
    (work_dir / 'training_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

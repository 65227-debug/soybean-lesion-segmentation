#!/usr/bin/env python
"""Distil the validation-selected A/corrected-F teacher into one student."""

import argparse
import contextlib
import csv
import json
import random
import sys
import time
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
    parser.add_argument('--checkpoint-a', required=True)
    parser.add_argument('--checkpoint-f', required=True)
    parser.add_argument('--strong-teacher-config', default=None)
    parser.add_argument('--strong-teacher-checkpoint', default=None)
    parser.add_argument('--supervised-root', default='data/dadou_student')
    parser.add_argument('--pseudo-root', default='data/dadou_student_corrected_f')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument(
        '--training-config',
        default='paper_code/corrected_f_training/distillation_config.json')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, workers, seed, drop_last,
                sampler=None):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle if sampler is None else False, sampler=sampler,
        drop_last=drop_last,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0, worker_init_fn=common.seed_worker,
        generator=torch.Generator().manual_seed(seed))


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def disease_code(sample_id):
    return sample_id.rsplit('_', 1)[-1]


def supervised_sampler(manifest_path, config):
    target = config.get('supervised_disease_sampling_probability')
    if not target:
        return None
    rows = read_manifest(manifest_path)
    counts = {}
    for row in rows:
        code = disease_code(row['sample_id'])
        counts[code] = counts.get(code, 0) + 1
    weights = [target.get(disease_code(row['sample_id']), 0.0)
               / counts[disease_code(row['sample_id'])] for row in rows]
    return WeightedRandomSampler(
        weights,
        num_samples=config['steps_per_epoch'] * config['supervised_batch_size'],
        replacement=True,
        generator=torch.Generator().manual_seed(config['seed'] + 303))


def pseudo_scene_class_sampler(manifest_path, config):
    if not config.get('balance_pseudo_by_scene_and_class', False):
        return None
    rows = read_manifest(manifest_path)
    scene_counts = {}
    scenes_by_class = {}
    for row in rows:
        key = (row['class_name'], row['scene_id'])
        scene_counts[key] = scene_counts.get(key, 0) + 1
        scenes_by_class.setdefault(row['class_name'], set()).add(row['scene_id'])
    target = config.get('pseudo_class_sampling_probability', {})
    weights = []
    for row in rows:
        class_name = row['class_name']
        key = (class_name, row['scene_id'])
        class_probability = target.get(
            class_name, 1.0 / max(1, len(scenes_by_class)))
        weights.append(
            class_probability
            / len(scenes_by_class[class_name])
            / scene_counts[key])
    return WeightedRandomSampler(
        weights,
        num_samples=config['steps_per_epoch'] * config['pseudo_batch_size'],
        replacement=True,
        generator=torch.Generator().manual_seed(config['seed'] + 404))


def make_loaders(args, config):
    augmentation = config['augmentation']
    supervised_manifest = (
        Path(args.supervised_root) / 'supervised_train_manifest.csv')
    pseudo_manifest = Path(args.pseudo_root) / 'pseudo_train_manifest.csv'
    supervised = common.StudentSegmentationDataset(
        supervised_manifest,
        training=True, pseudo=False, augmentation=augmentation)
    pseudo = common.StudentSegmentationDataset(
        pseudo_manifest,
        training=True, pseudo=True, augmentation=augmentation)
    validation = common.StudentSegmentationDataset(
        Path(args.supervised_root) / 'supervised_val_manifest.csv',
        training=False)
    return (
        make_loader(
            supervised, config['supervised_batch_size'], True,
            config['workers'], config['seed'], True,
            sampler=supervised_sampler(supervised_manifest, config)),
        make_loader(
            pseudo, config['pseudo_batch_size'], True, config['workers'],
            config['seed'] + 101, True,
            sampler=pseudo_scene_class_sampler(pseudo_manifest, config)),
        make_loader(
            validation, config['validation_batch_size'], False,
            config['workers'], config['seed'] + 202, False),
    )


@contextlib.contextmanager
def frozen_batch_norm_statistics(model):
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


def configure_trainable_parameters(model, scope):
    if scope == 'all':
        return [parameter for parameter in model.parameters()
                if parameter.requires_grad]
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == 'decode_head':
        module = model.decode_head
    elif scope == 'classifier':
        module = model.decode_head.conv_seg
    else:
        raise ValueError(f'Unknown trainable_scope: {scope}')
    for parameter in module.parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters()
            if parameter.requires_grad]


@torch.no_grad()
def ensemble_probabilities(teacher_a, teacher_f, strong_teacher, images,
                           output_size, config):
    strong_weight = config.get('strong_teacher_weight', 0.0)
    if strong_teacher is not None and strong_weight >= 1.0:
        logits = common.forward_logits(strong_teacher, images, output_size)
        bias = logits.new_zeros((1, 3, 1, 1))
    else:
        logits_a = common.forward_logits(teacher_a, images, output_size)
        logits_f = common.forward_logits(teacher_f, images, output_size)
        alpha = config['teacher_alpha_corrected_f']
        logits = logits_a * (1.0 - alpha) + logits_f * alpha
        bias = logits.new_tensor([
            0.0, config['teacher_healthy_logit_bias'],
            config['teacher_lesion_logit_bias']]).view(1, 3, 1, 1)
        if strong_teacher is not None and strong_weight > 0:
            strong_logits = common.forward_logits(
                strong_teacher, images, output_size)
            logits = logits * (1.0 - strong_weight) + strong_logits * strong_weight
            bias = bias * (1.0 - strong_weight)
    temperature = config['temperature']
    return ((logits + bias) / temperature).softmax(dim=1)


def balanced_distillation_loss(student_logits, teacher_probabilities,
                                gate_masks, gate_weights, config):
    temperature = config['temperature']
    teacher_class = teacher_probabilities.argmax(dim=1)
    teacher_confidence = teacher_probabilities.max(dim=1).values
    accepted_gate = ((gate_masks != 255)
                     & (teacher_confidence
                        >= config['minimum_teacher_confidence']))
    weights = gate_weights * teacher_confidence * accepted_gate.float()
    # The hard gate deliberately ignores uncertain boundaries.  Recover only
    # teacher-supported lesion candidates there with a small soft weight.
    outside_lesion = (
        (gate_masks == 255)
        & (teacher_class == 2)
        & (teacher_probabilities[:, 2]
           >= config.get('outside_gate_lesion_minimum_probability', 1.1))
        & (teacher_confidence
           >= config.get('outside_gate_minimum_confidence', 1.1)))
    outside_weight = config.get('outside_gate_lesion_weight', 0.0)
    weights = torch.where(
        outside_lesion,
        outside_weight * teacher_probabilities[:, 2],
        weights)
    valid = weights > 0
    soft_ce = -(teacher_probabilities * F.log_softmax(
        student_logits / temperature, dim=1)).sum(dim=1)
    hard_ce = F.cross_entropy(
        student_logits, teacher_class, reduction='none')
    per_pixel = (config['soft_target_ratio'] * soft_ce * temperature ** 2
                 + (1.0 - config['soft_target_ratio']) * hard_ce)
    losses = []
    class_values = []
    for class_index, coefficient in enumerate(
            config['distillation_class_coefficients']):
        class_weights = weights * (teacher_class == class_index).float()
        denominator = class_weights.sum()
        if denominator.item() > 0:
            value = (per_pixel * class_weights).sum() / (denominator + 1e-6)
            losses.append(coefficient * value)
            class_values.append(float(value.detach().item()))
        else:
            class_values.append(0.0)
    if not losses:
        return student_logits.sum() * 0.0, class_values, 0.0
    denominator = sum(config['distillation_class_coefficients'])
    accepted = float(valid.float().mean().detach().item())
    accepted_by_class = [
        float(((teacher_class == class_index) & valid).float().mean()
              .detach().item())
        for class_index in range(3)]
    outside_percent = float(outside_lesion.float().mean().detach().item())
    return (sum(losses) / denominator, class_values, accepted,
            accepted_by_class, outside_percent)


def lesion_tversky_loss(logits, targets, alpha=0.3, beta=0.7,
                        epsilon=1e-6):
    valid = targets != 255
    probability = logits.softmax(dim=1)[:, 2]
    truth = (targets == 2).float()
    probability = probability * valid.float()
    truth = truth * valid.float()
    true_positive = (probability * truth).sum()
    false_positive = (probability * (1.0 - truth)).sum()
    false_negative = ((1.0 - probability) * truth).sum()
    score = ((true_positive + epsilon)
             / (true_positive + alpha * false_positive
                + beta * false_negative + epsilon))
    return 1.0 - score


def supervised_soft_distillation_loss(student_logits, teacher_probabilities,
                                      targets, temperature):
    per_pixel = -(teacher_probabilities * F.log_softmax(
        student_logits / temperature, dim=1)).sum(dim=1) * temperature ** 2
    valid = targets != 255
    losses = []
    for class_index in range(3):
        class_mask = valid & (targets == class_index)
        if class_mask.any():
            losses.append(per_pixel[class_mask].mean())
    return sum(losses) / len(losses)


def lambda_for_epoch(epoch, config):
    ramp = config['distillation_ramp_epochs']
    if ramp <= 1:
        return config['maximum_lambda_distill']
    return config['maximum_lambda_distill'] * min(1.0, epoch / ramp)


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_epoch(student, teacher_a, teacher_f, strong_teacher,
                supervised_loader,
                pseudo_loader, optimizer, scaler, device, class_weights,
                lambda_distill, config):
    student.train()
    teacher_a.eval()
    teacher_f.eval()
    supervised_iterator = iter(supervised_loader)
    pseudo_iterator = iter(pseudo_loader)
    totals = {
        'supervised_loss': 0.0, 'distillation_loss': 0.0,
        'total_loss': 0.0, 'accepted_distillation_pixel_percent': 0.0,
        'distill_background': 0.0, 'distill_healthy': 0.0,
        'distill_lesion': 0.0,
        'accepted_teacher_background_percent': 0.0,
        'accepted_teacher_healthy_percent': 0.0,
        'accepted_teacher_lesion_percent': 0.0,
        'outside_gate_lesion_percent': 0.0,
        'lesion_tversky_loss': 0.0,
        'supervised_distillation_loss': 0.0,
    }
    for _ in range(config['steps_per_epoch']):
        supervised, supervised_iterator = next_or_restart(
            supervised_iterator, supervised_loader)
        pseudo, pseudo_iterator = next_or_restart(
            pseudo_iterator, pseudo_loader)
        images_s = supervised['image'].to(device, non_blocking=True)
        masks_s = supervised['mask'].to(device, non_blocking=True)
        images_u = pseudo['image'].to(device, non_blocking=True)
        masks_u = pseudo['mask'].to(device, non_blocking=True)
        weights_u = pseudo['weight'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
                device_type=device.type, enabled=device.type == 'cuda'):
            with frozen_batch_norm_statistics(student):
                logits_s = common.forward_logits(
                    student, images_s, masks_s.shape[-2:])
            supervised_loss = (
                F.cross_entropy(logits_s, masks_s, weight=class_weights)
                + common.soft_dice_loss(logits_s, masks_s))
            tversky = lesion_tversky_loss(
                logits_s, masks_s,
                alpha=config.get('tversky_alpha', 0.3),
                beta=config.get('tversky_beta', 0.7))
            supervised_loss = (
                supervised_loss
                + config.get('lesion_tversky_weight', 0.0) * tversky)
            supervised_distillation = supervised_loss.new_zeros(())
            if (strong_teacher is not None
                    and config.get('supervised_strong_teacher_weight', 0) > 0):
                strong_probabilities = ensemble_probabilities(
                    None, None, strong_teacher, images_s,
                    masks_s.shape[-2:], {
                        **config, 'strong_teacher_weight': 1.0})
                supervised_distillation = supervised_soft_distillation_loss(
                    logits_s, strong_probabilities, masks_s,
                    config['temperature'])
                supervised_loss = (
                    supervised_loss
                    + config['supervised_strong_teacher_weight']
                    * supervised_distillation)
            teacher_probabilities = ensemble_probabilities(
                teacher_a, teacher_f, strong_teacher, images_u,
                masks_u.shape[-2:], config)
            with frozen_batch_norm_statistics(student):
                logits_u = common.forward_logits(
                    student, images_u, masks_u.shape[-2:])
            (distill_loss, class_values, accepted, accepted_by_class,
             outside_percent) = balanced_distillation_loss(
                 logits_u, teacher_probabilities, masks_u, weights_u, config)
            total_loss = supervised_loss + lambda_distill * distill_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters()
             if parameter.requires_grad], config['gradient_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        totals['supervised_loss'] += float(supervised_loss.detach().item())
        totals['distillation_loss'] += float(distill_loss.detach().item())
        totals['total_loss'] += float(total_loss.detach().item())
        totals['accepted_distillation_pixel_percent'] += 100.0 * accepted
        totals['lesion_tversky_loss'] += float(tversky.detach().item())
        totals['supervised_distillation_loss'] += float(
            supervised_distillation.detach().item())
        totals['outside_gate_lesion_percent'] += 100.0 * outside_percent
        for key, value in zip(
                ('accepted_teacher_background_percent',
                 'accepted_teacher_healthy_percent',
                 'accepted_teacher_lesion_percent'), accepted_by_class):
            totals[key] += 100.0 * value
        for key, value in zip(
                ('distill_background', 'distill_healthy', 'distill_lesion'),
                class_values):
            totals[key] += value
    return {key: value / config['steps_per_epoch']
            for key, value in totals.items()}


def main():
    args = parse_args()
    config = json.loads(Path(args.training_config).read_text(encoding='utf-8'))
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.seed is not None:
        config['seed'] = args.seed
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    common.seed_everything(config['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    supervised_loader, pseudo_loader, validation_loader = make_loaders(
        args, config)
    checkpoint_a = torch.load(
        args.checkpoint_a, map_location='cpu', weights_only=True)
    checkpoint_f = torch.load(
        args.checkpoint_f, map_location='cpu', weights_only=True)
    student = common.build_student(args.model_config, device)
    teacher_a = common.build_student(args.model_config, device)
    teacher_f = common.build_student(args.model_config, device)
    strong_teacher = None
    if bool(args.strong_teacher_config) != bool(args.strong_teacher_checkpoint):
        raise ValueError(
            'Both --strong-teacher-config and --strong-teacher-checkpoint '
            'must be provided together.')
    if args.strong_teacher_config:
        strong_teacher = common.build_student(
            args.strong_teacher_config, device)
        strong_checkpoint = torch.load(
            args.strong_teacher_checkpoint, map_location='cpu',
            # Local checkpoint produced by this repository's mmengine run.
            weights_only=False)
        strong_teacher.load_state_dict(strong_checkpoint['state_dict'])
    student.load_state_dict(checkpoint_a['state_dict'])
    teacher_a.load_state_dict(checkpoint_a['state_dict'])
    teacher_f.load_state_dict(checkpoint_f['state_dict'])
    for teacher in (teacher_a, teacher_f, strong_teacher):
        if teacher is None:
            continue
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    trainable = configure_trainable_parameters(
        student, config['trainable_scope'])
    optimizer = torch.optim.AdamW(
        trainable, lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    class_weights = torch.tensor(
        config['class_weights'], dtype=torch.float32, device=device)

    initial_metrics, _ = common.evaluate(
        student, validation_loader, device, class_weights)
    best_metrics = initial_metrics
    best_epoch = 0
    best_path = work_dir / 'best_validation_mIoU.pth'
    torch.save({
        **checkpoint_a,
        'distillation_epoch': 0,
        'metrics': initial_metrics,
    }, best_path)
    history = []
    start = time.time()
    for epoch in range(1, config['max_epochs'] + 1):
        lambda_distill = lambda_for_epoch(epoch, config)
        train_metrics = train_epoch(
            student, teacher_a, teacher_f, strong_teacher, supervised_loader,
            pseudo_loader, optimizer, scaler, device, class_weights,
            lambda_distill, config)
        validation_metrics, _ = common.evaluate(
            student, validation_loader, device, class_weights)
        row = {
            'epoch': epoch, 'lambda_distill': lambda_distill,
            'learning_rate': optimizer.param_groups[0]['lr'],
            **{f'train_{key}': value for key, value in train_metrics.items()},
            **{f'val_{key}': value for key, value in validation_metrics.items()},
        }
        history.append(row)
        common.write_csv(work_dir / 'training_history.csv', history)
        torch.save({
            'state_dict': {key: value.detach().cpu()
                           for key, value in student.state_dict().items()},
            'epoch': epoch, 'metrics': validation_metrics,
            'architecture': 'DeepLabV3+ MobileNetV3-Large distilled student',
        }, work_dir / 'last_checkpoint.pth')
        if validation_metrics['mIoU'] > best_metrics['mIoU']:
            best_metrics = validation_metrics
            best_epoch = epoch
            torch.save({
                'state_dict': {key: value.detach().cpu()
                               for key, value in student.state_dict().items()},
                'epoch': epoch, 'metrics': validation_metrics,
                'architecture': 'DeepLabV3+ MobileNetV3-Large distilled student',
            }, best_path)
        print(json.dumps({
            'epoch': epoch,
            'train_total_loss': train_metrics['total_loss'],
            'val_mIoU': validation_metrics['mIoU'],
            'val_IoU_lesion': validation_metrics['IoU_lesion'],
            'best_epoch': best_epoch,
            'best_val_mIoU': best_metrics['mIoU'],
        }, ensure_ascii=False), flush=True)
    summary = {
        'architecture': 'DeepLabV3+ MobileNetV3-Large',
        'method': 'dual-evidence corrected F with soft ensemble distillation',
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'initial_validation_metrics': initial_metrics,
        'test_split_used': False,
        'config': config,
        'source_checkpoints': {
            'A': str(Path(args.checkpoint_a).resolve()),
            'corrected_F': str(Path(args.checkpoint_f).resolve()),
            'strong_teacher': (
                str(Path(args.strong_teacher_checkpoint).resolve())
                if args.strong_teacher_checkpoint else None),
        },
        'elapsed_seconds': time.time() - start,
    }
    (work_dir / 'training_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

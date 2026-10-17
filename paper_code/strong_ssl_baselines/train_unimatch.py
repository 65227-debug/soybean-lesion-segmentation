#!/usr/bin/env python
"""Protocol-aligned UniMatch reproduction for three-class soybean segmentation."""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ssl_common as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('aligned_protocol.json')))
    parser.add_argument('--work-dir', default='work_dirs/dadou_unimatch_aligned_seed20260803')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--steps-per-epoch', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_epoch(model, labeled_loader, unlabeled_loader, optimizer, scaler,
                class_weights, device, cfg, global_step):
    model.train()
    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader)
    unlabeled_mix_iterator = iter(unlabeled_loader)
    totals = {
        'loss_total': 0.0,
        'loss_supervised': 0.0,
        'loss_strong1': 0.0,
        'loss_strong2': 0.0,
        'loss_feature_perturbation': 0.0,
        'accepted_pixel_percent': 0.0,
    }
    confusion = np.zeros((3, 3), dtype=np.int64)
    settings = cfg['unimatch']
    for _ in range(cfg['steps_per_epoch']):
        labeled, labeled_iterator = next_or_restart(labeled_iterator, labeled_loader)
        unlabeled, unlabeled_iterator = next_or_restart(unlabeled_iterator, unlabeled_loader)
        mixed, unlabeled_mix_iterator = next_or_restart(
            unlabeled_mix_iterator, unlabeled_loader)
        images_x = labeled['image'].to(device, non_blocking=True)
        masks_x = labeled['mask'].to(device, non_blocking=True)
        weak = unlabeled['weak'].to(device, non_blocking=True)
        strong1 = unlabeled['strong1'].to(device, non_blocking=True)
        strong2 = unlabeled['strong2'].to(device, non_blocking=True)
        weak_mix = mixed['weak'].to(device, non_blocking=True)
        strong1_mix = mixed['strong1'].to(device, non_blocking=True)
        strong2_mix = mixed['strong2'].to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            model.eval()
            logits_mix = common.forward_logits(model, weak_mix)
            probability_mix = logits_mix.softmax(dim=1)
            confidence_mix, pseudo_mix = probability_mix.max(dim=1)
        boxes1 = [
            common.cutmix_box(
                strong1.shape[-2], strong1.shape[-1], cfg['strong_augmentation'], device)
            for _ in range(strong1.shape[0])]
        boxes2 = [
            common.cutmix_box(
                strong2.shape[-2], strong2.shape[-1], cfg['strong_augmentation'], device)
            for _ in range(strong2.shape[0])]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            combined = torch.cat([images_x, weak], dim=0)
            logits, logits_fp = common.forward_with_feature_perturbation(
                model, combined, settings['feature_dropout_probability'])
            logits_x = logits[:images_x.shape[0]]
            logits_weak = logits[images_x.shape[0]:].detach()
            logits_weak_fp = logits_fp[images_x.shape[0]:]
            probability = logits_weak.softmax(dim=1)
            confidence, pseudo = probability.max(dim=1)
            strong1_cut, pseudo1, confidence1 = common.paired_cutmix(
                strong1, pseudo, confidence, strong1_mix, pseudo_mix,
                confidence_mix, boxes1)
            strong2_cut, pseudo2, confidence2 = common.paired_cutmix(
                strong2, pseudo, confidence, strong2_mix, pseudo_mix,
                confidence_mix, boxes2)
            logits_strong = common.forward_logits(
                model, torch.cat([strong1_cut, strong2_cut], dim=0))
            logits_s1, logits_s2 = logits_strong.chunk(2, dim=0)
            accepted1 = confidence1 >= settings['confidence_threshold']
            accepted2 = confidence2 >= settings['confidence_threshold']
            accepted_fp = confidence >= settings['confidence_threshold']
            loss_x = common.supervised_loss(logits_x, masks_x, class_weights)
            loss_s1 = common.masked_ce(
                logits_s1, pseudo1, accepted1, denominator='all')
            loss_s2 = common.masked_ce(
                logits_s2, pseudo2, accepted2, denominator='all')
            loss_fp = common.masked_ce(
                logits_weak_fp, pseudo, accepted_fp, denominator='all')
            loss = (loss_x + 0.25 * loss_s1 + 0.25 * loss_s2 + 0.5 * loss_fp) / 2.0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        lr = common.cosine_lr(
            optimizer, global_step, cfg['total_optimizer_steps'],
            cfg['learning_rate'], cfg['minimum_learning_rate'])
        totals['loss_total'] += loss.item()
        totals['loss_supervised'] += loss_x.item()
        totals['loss_strong1'] += loss_s1.item()
        totals['loss_strong2'] += loss_s2.item()
        totals['loss_feature_perturbation'] += loss_fp.item()
        totals['accepted_pixel_percent'] += float(
            100.0 * torch.cat([
                accepted1.flatten(), accepted2.flatten(), accepted_fp.flatten()
            ]).float().mean().item())
        common.update_confusion(confusion, logits_x.detach().argmax(dim=1), masks_x)
    result = {key: value / cfg['steps_per_epoch'] for key, value in totals.items()}
    result.update({f'train_{key}': value for key, value in common.metrics_from_confusion(confusion).items()})
    result['learning_rate'] = lr
    return result, global_step


def main():
    args = parse_args()
    cfg = common.load_config(args.config)
    if args.smoke_test:
        cfg['max_epochs'] = 1
        cfg['steps_per_epoch'] = 2
        cfg['total_optimizer_steps'] = 2
        cfg['workers'] = 0
    if args.max_epochs is not None:
        cfg['max_epochs'] = args.max_epochs
        cfg['total_optimizer_steps'] = cfg['max_epochs'] * cfg['steps_per_epoch']
    if args.steps_per_epoch is not None:
        cfg['steps_per_epoch'] = args.steps_per_epoch
        cfg['total_optimizer_steps'] = cfg['max_epochs'] * cfg['steps_per_epoch']
    if args.workers is not None:
        cfg['workers'] = args.workers
    common.set_determinism(cfg['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    work_dir = common.resolve(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(common.resolve(args.config), work_dir / 'aligned_protocol.json')
    labeled = common.base.StudentSegmentationDataset(
        common.resolve(cfg['labeled_train_manifest']), training=True, pseudo=False,
        augmentation=cfg['base_augmentation'])
    unlabeled = common.WeakStrongUnlabeledDataset(
        cfg['unlabeled_manifest'], cfg['base_augmentation'],
        cfg['strong_augmentation'], two_strong=True)
    generator = torch.Generator().manual_seed(cfg['seed'])
    labeled_loader = common.make_loader(
        labeled, cfg['supervised_batch_size'], True, cfg, generator, True)
    unlabeled_loader = common.make_loader(
        unlabeled, cfg['unlabeled_batch_size'], True, cfg, generator, True)
    model = common.build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32, device=device)
    history = []
    best_miou = -1.0
    best_epoch = 0
    best_metrics = None
    global_step = 0
    started = time.time()
    best_path = work_dir / 'best_validation_mIoU.pth'
    for epoch in range(1, cfg['max_epochs'] + 1):
        train_metrics, global_step = train_epoch(
            model, labeled_loader, unlabeled_loader, optimizer, scaler,
            class_weights, device, cfg, global_step)
        validation, _ = common.evaluate_split(
            model, cfg, cfg['validation_manifest'], device)
        row = {'epoch': epoch, 'global_step': global_step, **train_metrics}
        row.update({f'val_{key}': value for key, value in validation.items()
                    if key != 'confusion_matrix'})
        history.append(row)
        common.write_rows(work_dir / 'epoch_metrics.csv', history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation['mIoU'] > best_miou:
            best_miou = validation['mIoU']
            best_epoch = epoch
            best_metrics = copy.deepcopy(validation)
            common.save_checkpoint(
                best_path, model, epoch, validation, 'UniMatch-aligned')
        common.save_checkpoint(
            work_dir / 'last_checkpoint.pth', model, epoch, validation,
            'UniMatch-aligned')
    frozen = common.build_model(cfg, device)
    common.load_checkpoint(frozen, best_path, device)
    test_metrics, test_rows = common.evaluate_split(
        frozen, cfg, cfg['test_manifest'], device,
        save_dir=work_dir / 'test_predictions')
    common.write_rows(work_dir / 'test_per_image.csv', test_rows)
    summary = {
        'method': 'UniMatch-aligned',
        'official_method_core': [
            'two strong image perturbation streams',
            'CutMix pseudo-label propagation',
            'feature-level dropout perturbation stream',
            'confidence-masked weak-to-strong consistency'],
        'protocol': cfg['protocol_name'],
        'seed': cfg['seed'],
        'labeled_train_images': len(labeled),
        'unlabeled_train_images': len(unlabeled),
        'validation_images': len(common.read_rows(cfg['validation_manifest'])),
        'test_images': len(test_rows),
        'epochs_completed': cfg['max_epochs'],
        'steps_per_epoch': cfg['steps_per_epoch'],
        'total_optimizer_steps': global_step,
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'frozen_test_metrics': test_metrics,
        'model_selection': 'validation mIoU only',
        'test_tuning': False,
        'elapsed_seconds': time.time() - started,
        'adaptation_scope': cfg['claim_scope'],
    }
    common.write_json(work_dir / 'training_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

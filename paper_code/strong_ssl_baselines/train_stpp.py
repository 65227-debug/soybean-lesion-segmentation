#!/usr/bin/env python
"""Protocol-aligned ST++ reproduction with stability-based image selection."""

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import ssl_common as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('aligned_protocol.json')))
    parser.add_argument('--work-dir', default='work_dirs/dadou_stpp_aligned_seed20260803')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_stage(stage_name, cfg, device, labeled_loader, pseudo_loader,
                work_dir, epochs, snapshot_epochs=()):
    model = common.build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32, device=device)
    total_steps = epochs * cfg['steps_per_epoch']
    global_step = 0
    history = []
    best_miou = -1.0
    best_epoch = 0
    best_metrics = None
    best_path = work_dir / f'{stage_name}_best_validation_mIoU.pth'
    for epoch in range(1, epochs + 1):
        model.train()
        labeled_iterator = iter(labeled_loader)
        pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None
        totals = {'loss_total': 0.0, 'loss_supervised': 0.0, 'loss_pseudo': 0.0}
        confusion = np.zeros((3, 3), dtype=np.int64)
        for _ in range(cfg['steps_per_epoch']):
            labeled, labeled_iterator = next_or_restart(labeled_iterator, labeled_loader)
            images_x = labeled['image'].to(device, non_blocking=True)
            masks_x = labeled['mask'].to(device, non_blocking=True)
            pseudo = None
            if pseudo_loader is not None:
                pseudo, pseudo_iterator = next_or_restart(pseudo_iterator, pseudo_loader)
                images_u = pseudo['image'].to(device, non_blocking=True)
                masks_u = pseudo['mask'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                if pseudo is None:
                    logits_x = common.forward_logits(model, images_x)
                    loss_x = common.supervised_loss(logits_x, masks_x, class_weights)
                    loss_u = loss_x.new_zeros(())
                    loss = loss_x
                else:
                    logits = common.forward_logits(
                        model, torch.cat([images_x, images_u], dim=0))
                    logits_x = logits[:images_x.shape[0]]
                    logits_u = logits[images_x.shape[0]:]
                    loss_x = common.supervised_loss(logits_x, masks_x, class_weights)
                    loss_u = F.cross_entropy(
                        logits_u, masks_u, weight=class_weights, ignore_index=255)
                    loss = 0.5 * (loss_x + loss_u)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            lr = common.cosine_lr(
                optimizer, global_step, total_steps, cfg['learning_rate'],
                cfg['minimum_learning_rate'])
            totals['loss_total'] += loss.item()
            totals['loss_supervised'] += loss_x.item()
            totals['loss_pseudo'] += loss_u.item()
            common.update_confusion(
                confusion, logits_x.detach().argmax(dim=1), masks_x)
        validation, _ = common.evaluate_split(
            model, cfg, cfg['validation_manifest'], device)
        row = {
            'stage': stage_name,
            'epoch': epoch,
            'stage_step': global_step,
            'learning_rate': lr,
            **{key: value / cfg['steps_per_epoch'] for key, value in totals.items()},
            **{f'train_{key}': value for key, value in common.metrics_from_confusion(confusion).items()},
            **{f'val_{key}': value for key, value in validation.items()
               if key != 'confusion_matrix'},
        }
        history.append(row)
        common.write_rows(work_dir / f'{stage_name}_epoch_metrics.csv', history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation['mIoU'] > best_miou:
            best_miou = validation['mIoU']
            best_epoch = epoch
            best_metrics = copy.deepcopy(validation)
            common.save_checkpoint(best_path, model, epoch, validation, f'ST++-aligned/{stage_name}')
        if epoch in snapshot_epochs:
            common.save_checkpoint(
                work_dir / f'{stage_name}_snapshot_epoch{epoch:02d}.pth',
                model, epoch, validation, f'ST++-aligned/{stage_name}-reliability-snapshot')
    return {
        'best_path': best_path,
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'optimizer_steps': global_step,
        'history': history,
    }


@torch.no_grad()
def select_reliable(cfg, device, work_dir, unlabeled_rows, snapshot_epochs):
    models = []
    for epoch in snapshot_epochs:
        model = common.build_model(cfg, device)
        common.load_checkpoint(
            model, work_dir / f'stage1_snapshot_epoch{epoch:02d}.pth', device)
        model.eval()
        models.append(model)
    dataset = common.ImageOnlyDataset(cfg['unlabeled_manifest'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = common.make_loader(
        dataset, cfg['validation_batch_size'], False, cfg, generator, False)
    score_rows = []
    for images, sample_ids, _ in loader:
        images = images.to(device, non_blocking=True)
        predictions = [
            common.forward_logits(model, images).argmax(dim=1).cpu().numpy()
            for model in models]
        for index, sample_id in enumerate(sample_ids):
            scores = []
            for earlier in predictions[:-1]:
                confusion = np.zeros((3, 3), dtype=np.int64)
                target = torch.from_numpy(predictions[-1][index])
                pred = torch.from_numpy(earlier[index])
                common.update_confusion(confusion, pred, target)
                scores.append(common.metrics_from_confusion(confusion)['mIoU'])
            score_rows.append({
                'sample_id': sample_id,
                'stability_mIoU': float(np.mean(scores)),
            })
    del models
    torch.cuda.empty_cache()
    score_rows.sort(key=lambda row: (-row['stability_mIoU'], row['sample_id']))
    reliable_count = int(len(score_rows) * cfg['stpp']['reliable_fraction'])
    reliable_ids = {row['sample_id'] for row in score_rows[:reliable_count]}
    by_id = {row['sample_id']: row for row in unlabeled_rows}
    reliable_rows = [by_id[row['sample_id']] for row in score_rows[:reliable_count]]
    unreliable_rows = [by_id[row['sample_id']] for row in score_rows[reliable_count:]]
    for rank, row in enumerate(score_rows, start=1):
        row['rank'] = rank
        row['reliable'] = row['sample_id'] in reliable_ids
    common.write_rows(work_dir / 'reliability_ranking.csv', score_rows)
    return reliable_rows, unreliable_rows


@torch.no_grad()
def generate_pseudo(cfg, device, checkpoint_path, rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = common.build_model(cfg, device)
    common.load_checkpoint(model, checkpoint_path, device)
    model.eval()
    row_ids = {row['sample_id'] for row in rows}
    selected = [row for row in common.image_only_unlabeled_rows(
        cfg['unlabeled_manifest']) if row['sample_id'] in row_ids]
    dataset = common.ImageOnlyDataset(cfg['unlabeled_manifest'])
    dataset.rows = selected
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = common.make_loader(
        dataset, cfg['validation_batch_size'], False, cfg, generator, False)
    count = 0
    for images, sample_ids, _ in loader:
        predictions = common.forward_logits(
            model, images.to(device, non_blocking=True)).argmax(dim=1).cpu().numpy()
        for sample_id, prediction in zip(sample_ids, predictions):
            path = output_dir / f'{sample_id}.png'
            if not cv2.imwrite(str(path), prediction.astype(np.uint8)):
                raise RuntimeError(f'Failed to write {path}')
            count += 1
    del model
    torch.cuda.empty_cache()
    return count


def main():
    args = parse_args()
    cfg = common.load_config(args.config)
    if args.workers is not None:
        cfg['workers'] = args.workers
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.smoke_test:
        cfg['workers'] = 0
        cfg['steps_per_epoch'] = 2
        cfg['stpp']['stage_epochs'] = [1, 1, 1]
        cfg['stpp']['stage_steps'] = [2, 2, 2]
        cfg['stpp']['stage1_reliability_snapshots'] = [1, 1]
    common.set_determinism(cfg['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    work_dir = common.resolve(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'aligned_protocol.json').write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    generator = torch.Generator().manual_seed(cfg['seed'])
    labeled = common.base.StudentSegmentationDataset(
        common.resolve(cfg['labeled_train_manifest']), training=True, pseudo=False,
        augmentation=cfg['base_augmentation'])
    labeled_loader = common.make_loader(
        labeled, cfg['supervised_batch_size'], True, cfg, generator, True)
    unlabeled_rows = common.image_only_unlabeled_rows(cfg['unlabeled_manifest'])
    started = time.time()
    epochs1, epochs2, epochs3 = cfg['stpp']['stage_epochs']
    snapshots = cfg['stpp']['stage1_reliability_snapshots']
    stage1 = train_stage(
        'stage1', cfg, device, labeled_loader, None, work_dir, epochs1,
        snapshot_epochs=snapshots)
    reliable_rows, unreliable_rows = select_reliable(
        cfg, device, work_dir, unlabeled_rows, snapshots)
    pseudo_dir = work_dir / 'pseudo_masks'
    reliable_count = generate_pseudo(
        cfg, device, stage1['best_path'], reliable_rows, pseudo_dir)
    pseudo_reliable = common.HardPseudoDataset(
        reliable_rows, pseudo_dir, cfg['base_augmentation'], cfg['strong_augmentation'])
    pseudo_reliable_loader = common.make_loader(
        pseudo_reliable, cfg['unlabeled_batch_size'], True, cfg, generator, True)
    stage2 = train_stage(
        'stage2', cfg, device, labeled_loader, pseudo_reliable_loader,
        work_dir, epochs2)
    unreliable_count = generate_pseudo(
        cfg, device, stage2['best_path'], unreliable_rows, pseudo_dir)
    pseudo_all = common.HardPseudoDataset(
        unlabeled_rows, pseudo_dir, cfg['base_augmentation'], cfg['strong_augmentation'])
    pseudo_all_loader = common.make_loader(
        pseudo_all, cfg['unlabeled_batch_size'], True, cfg, generator, True)
    stage3 = train_stage(
        'stage3', cfg, device, labeled_loader, pseudo_all_loader,
        work_dir, epochs3)
    frozen = common.build_model(cfg, device)
    common.load_checkpoint(frozen, stage3['best_path'], device)
    test_metrics, test_rows = common.evaluate_split(
        frozen, cfg, cfg['test_manifest'], device,
        save_dir=work_dir / 'test_predictions')
    common.write_rows(work_dir / 'test_per_image.csv', test_rows)
    summary = {
        'method': 'ST++-aligned',
        'official_method_core': [
            'three supervised/self-training stages',
            'cross-checkpoint stability ranking',
            'top-half reliable image selection',
            'reliable-first then all-unlabeled pseudo-label retraining'],
        'protocol': cfg['protocol_name'],
        'seed': cfg['seed'],
        'labeled_train_images': len(labeled),
        'unlabeled_train_images': len(unlabeled_rows),
        'reliable_images': len(reliable_rows),
        'unreliable_images': len(unreliable_rows),
        'pseudo_masks_generated': reliable_count + unreliable_count,
        'stage_epochs': cfg['stpp']['stage_epochs'],
        'stage_optimizer_steps': [
            stage1['optimizer_steps'], stage2['optimizer_steps'], stage3['optimizer_steps']],
        'total_optimizer_steps': sum([
            stage1['optimizer_steps'], stage2['optimizer_steps'], stage3['optimizer_steps']]),
        'stage1_best_epoch': stage1['best_epoch'],
        'stage1_best_validation_metrics': stage1['best_validation_metrics'],
        'stage2_best_epoch': stage2['best_epoch'],
        'stage2_best_validation_metrics': stage2['best_validation_metrics'],
        'stage3_best_epoch': stage3['best_epoch'],
        'stage3_best_validation_metrics': stage3['best_validation_metrics'],
        'frozen_test_metrics': test_metrics,
        'model_selection': 'validation mIoU only in each stage',
        'test_tuning': False,
        'elapsed_seconds': time.time() - started,
        'adaptation_scope': cfg['claim_scope'],
    }
    common.write_json(work_dir / 'training_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

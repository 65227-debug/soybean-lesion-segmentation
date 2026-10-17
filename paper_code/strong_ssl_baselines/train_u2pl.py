#!/usr/bin/env python
"""Protocol-aligned U2PL reproduction for the three-class soybean task."""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import ssl_common as common


class U2PLModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.segmentor = common.build_model(cfg, device)
        channels = cfg['u2pl']['representation_channels']
        self.projector = nn.Sequential(
            nn.Conv2d(960, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        ).to(device)

    def forward(self, images):
        features = self.segmentor.extract_feat(images)
        logits = self.segmentor.decode_head.forward(features)
        logits = F.interpolate(
            logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        representation = F.normalize(self.projector(features[-1]), dim=1)
        return logits, representation


class MemoryBank:
    def __init__(self, channels, device):
        self.channels = channels
        self.device = device
        self.queues = [torch.empty((0, channels), dtype=torch.float32)
                       for _ in range(3)]
        self.maximum = [50000, 30000, 30000]

    def enqueue(self, class_index, keys):
        if keys.numel() == 0:
            return 0
        keys = keys.detach().float().cpu()
        self.queues[class_index] = torch.cat(
            [self.queues[class_index], keys], dim=0)[-self.maximum[class_index]:]
        return len(keys)

    def sample(self, class_index, count):
        queue = self.queues[class_index]
        if len(queue) == 0:
            return None
        indices = torch.randint(len(queue), (count,))
        return queue[indices].to(self.device, non_blocking=True)

    def sizes(self):
        return [len(queue) for queue in self.queues]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('aligned_protocol.json')))
    parser.add_argument('--work-dir', default='work_dirs/dadou_u2pl_aligned_seed20260803')
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


@torch.no_grad()
def update_ema(teacher, student, decay):
    for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.mul_(decay).add_(student_buffer, alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


def one_hot(labels, classes=3):
    safe = labels.clone()
    valid = safe != 255
    safe[~valid] = 0
    encoded = F.one_hot(safe, num_classes=classes).permute(0, 3, 1, 2).float()
    return encoded * valid.unsqueeze(1).float()


def percentile(tensor, percent):
    flattened = tensor.detach().float().flatten()
    if flattened.numel() == 0:
        return tensor.new_tensor(0.0)
    return torch.quantile(flattened, percent / 100.0)


def unreliable_contrastive_loss(
        representation, teacher_representation, teacher_probability,
        labels_l, pseudo_u, low_mask_u, high_mask_u, memory, cfg):
    settings = cfg['u2pl']
    batch_l = labels_l.shape[0]
    height, width = representation.shape[-2:]
    representation = representation.permute(0, 2, 3, 1)
    teacher_representation = teacher_representation.permute(0, 2, 3, 1)
    probability = F.interpolate(
        teacher_probability, size=(height, width), mode='bilinear',
        align_corners=False).permute(0, 2, 3, 1)
    labels_l_small = F.interpolate(
        one_hot(labels_l), size=(height, width), mode='nearest')
    pseudo_u_small = F.interpolate(
        one_hot(pseudo_u), size=(height, width), mode='nearest')
    labels_all = torch.cat([labels_l_small, pseudo_u_small], dim=0)
    low_all = torch.cat([
        (labels_l != 255).unsqueeze(1).float(), low_mask_u.unsqueeze(1).float()
    ], dim=0)
    high_all = torch.cat([
        (labels_l != 255).unsqueeze(1).float(), high_mask_u.unsqueeze(1).float()
    ], dim=0)
    low_all = F.interpolate(low_all, size=(height, width), mode='nearest')[:, 0].bool()
    high_all = F.interpolate(high_all, size=(height, width), mode='nearest')[:, 0].bool()
    ranked = probability.argsort(dim=-1, descending=True)
    low_rank, high_rank = settings['rank_interval_adapted_for_three_classes']
    losses = []
    enqueued = [0, 0, 0]
    for class_index in range(3):
        class_pixels = labels_all[:, class_index].bool()
        positive_mask = (
            class_pixels & low_all
            & (probability[..., class_index] > settings['class_positive_threshold']))
        positive_teacher = teacher_representation[positive_mask]
        if positive_teacher.numel() == 0:
            continue
        prototype = positive_teacher.mean(dim=0, keepdim=True)
        labeled_negative_rank = (ranked[:batch_l, ..., :low_rank] == class_index).any(dim=-1)
        unlabeled_negative_rank = (
            ranked[batch_l:, ..., low_rank:high_rank] == class_index).any(dim=-1)
        negative_rank = torch.cat([
            (~labeled_negative_rank) & (~class_pixels[:batch_l]),
            unlabeled_negative_rank,
        ], dim=0)
        negative_mask = (
            high_all & negative_rank
            & (probability[..., class_index] < settings['class_negative_threshold']))
        enqueued[class_index] = memory.enqueue(
            class_index, teacher_representation[negative_mask])
        anchors = representation[positive_mask]
        if anchors.numel() == 0 or memory.sizes()[class_index] == 0:
            continue
        query_indices = torch.randint(
            len(anchors), (settings['num_queries'],), device=anchors.device)
        anchors = anchors[query_indices]
        negatives = memory.sample(
            class_index, settings['num_queries'] * settings['num_negatives'])
        negatives = negatives.reshape(
            settings['num_queries'], settings['num_negatives'], -1)
        positives = prototype.unsqueeze(0).expand(settings['num_queries'], 1, -1)
        candidates = torch.cat([positives, negatives], dim=1)
        logits = F.cosine_similarity(anchors.unsqueeze(1), candidates, dim=2)
        targets = torch.zeros(
            settings['num_queries'], dtype=torch.long, device=anchors.device)
        losses.append(F.cross_entropy(logits / settings['temperature'], targets))
    if not losses:
        loss = representation.sum() * 0.0
    else:
        loss = torch.stack(losses).mean()
    return loss, enqueued


def train_epoch(student, teacher, labeled_loader, unlabeled_loader, optimizer,
                scaler, memory, class_weights, device, cfg, epoch, global_step):
    student.train()
    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader)
    mixed_iterator = iter(unlabeled_loader)
    totals = {
        'loss_total': 0.0,
        'loss_supervised': 0.0,
        'loss_unsupervised': 0.0,
        'loss_contrastive': 0.0,
        'reliable_pixel_percent': 0.0,
    }
    enqueued_total = np.zeros(3, dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    warmup = cfg['u2pl']['supervised_only_epochs']
    for _ in range(cfg['steps_per_epoch']):
        labeled, labeled_iterator = next_or_restart(labeled_iterator, labeled_loader)
        unlabeled, unlabeled_iterator = next_or_restart(unlabeled_iterator, unlabeled_loader)
        mixed, mixed_iterator = next_or_restart(mixed_iterator, unlabeled_loader)
        images_x = labeled['image'].to(device, non_blocking=True)
        masks_x = labeled['mask'].to(device, non_blocking=True)
        weak = unlabeled['weak'].to(device, non_blocking=True)
        strong = unlabeled['strong1'].to(device, non_blocking=True)
        weak_mix = mixed['weak'].to(device, non_blocking=True)
        strong_mix = mixed['strong1'].to(device, non_blocking=True)
        if epoch <= warmup:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits_x, _ = student(images_x)
                loss_x = common.supervised_loss(logits_x, masks_x, class_weights)
                loss_u = loss_x.new_zeros(())
                loss_c = loss_x.new_zeros(())
                loss = loss_x
            reliable_percent = 0.0
            enqueued = [0, 0, 0]
        else:
            teacher.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits_weak, _ = teacher(weak)
                logits_mix, _ = teacher(weak_mix)
                probability_weak = logits_weak.softmax(dim=1)
                probability_mix = logits_mix.softmax(dim=1)
                confidence, pseudo = probability_weak.max(dim=1)
                confidence_mix, pseudo_mix = probability_mix.max(dim=1)
            boxes = [common.cutmix_box(
                strong.shape[-2], strong.shape[-1], cfg['strong_augmentation'], device)
                for _ in range(strong.shape[0])]
            strong_cut, pseudo_cut, _ = common.paired_cutmix(
                strong, pseudo, confidence, strong_mix, pseudo_mix,
                confidence_mix, boxes)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits_all, representation = student(
                    torch.cat([images_x, strong_cut], dim=0))
                logits_x = logits_all[:images_x.shape[0]]
                logits_u = logits_all[images_x.shape[0]:]
                teacher.train()
                with torch.no_grad():
                    teacher_logits, teacher_representation = teacher(
                        torch.cat([images_x, strong_cut], dim=0))
                    teacher_probability = teacher_logits.softmax(dim=1)
                    probability_u = teacher_probability[images_x.shape[0]:]
                    entropy = -torch.sum(
                        probability_u * torch.log(probability_u + 1e-10), dim=1)
                    reliable_percentile = 100.0 - (
                        100.0 - cfg['u2pl']['initial_reliable_percent']) * (
                            1.0 - (epoch - 1) / cfg['max_epochs'])
                    reliable_threshold = percentile(entropy, reliable_percentile)
                    reliable_mask = entropy < reliable_threshold
                    alpha = cfg['u2pl']['low_entropy_percent'] * (
                        1.0 - (epoch - 1) / cfg['max_epochs'])
                    low_threshold = percentile(entropy, alpha)
                    high_threshold = percentile(entropy, 100.0 - alpha)
                    low_mask = entropy <= low_threshold
                    high_mask = entropy >= high_threshold
                loss_x = common.supervised_loss(logits_x, masks_x, class_weights)
                per_pixel = F.cross_entropy(logits_u, pseudo_cut, reduction='none')
                valid_count = reliable_mask.float().sum().clamp_min(1.0)
                scaling = reliable_mask.numel() / valid_count
                loss_u = scaling * per_pixel[reliable_mask].mean() if reliable_mask.any() else per_pixel.sum() * 0.0
                loss_c, enqueued = unreliable_contrastive_loss(
                    representation, teacher_representation, teacher_probability,
                    masks_x, pseudo_cut, low_mask, high_mask, memory, cfg)
                loss = loss_x + loss_u + cfg['u2pl']['contrastive_loss_weight'] * loss_c
            reliable_percent = float(100.0 * reliable_mask.float().mean().item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['gradient_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        if epoch == warmup:
            teacher.load_state_dict(student.state_dict())
        elif epoch > warmup:
            ema_step = global_step - warmup * cfg['steps_per_epoch'] + 1
            decay = min(1.0 - 1.0 / max(ema_step, 1), cfg['u2pl']['ema_decay'])
            update_ema(teacher, student, decay)
        global_step += 1
        lr = common.cosine_lr(
            optimizer, global_step, cfg['total_optimizer_steps'],
            cfg['learning_rate'], cfg['minimum_learning_rate'])
        totals['loss_total'] += loss.item()
        totals['loss_supervised'] += loss_x.item()
        totals['loss_unsupervised'] += loss_u.item()
        totals['loss_contrastive'] += loss_c.item()
        totals['reliable_pixel_percent'] += reliable_percent
        enqueued_total += np.asarray(enqueued, dtype=np.int64)
        common.update_confusion(
            confusion, logits_x.detach().argmax(dim=1), masks_x)
    result = {key: value / cfg['steps_per_epoch'] for key, value in totals.items()}
    result.update({f'train_{key}': value for key, value in common.metrics_from_confusion(confusion).items()})
    result['learning_rate'] = lr
    result['memory_bank_background'] = memory.sizes()[0]
    result['memory_bank_healthy'] = memory.sizes()[1]
    result['memory_bank_lesion'] = memory.sizes()[2]
    result['enqueued_background'] = int(enqueued_total[0])
    result['enqueued_healthy'] = int(enqueued_total[1])
    result['enqueued_lesion'] = int(enqueued_total[2])
    return result, global_step


def validation_model(student, teacher, epoch, cfg):
    return student if epoch <= cfg['u2pl']['supervised_only_epochs'] else teacher


def main():
    args = parse_args()
    cfg = common.load_config(args.config)
    if args.smoke_test:
        cfg['max_epochs'] = 2
        cfg['steps_per_epoch'] = 2
        cfg['total_optimizer_steps'] = 4
        cfg['workers'] = 0
        cfg['u2pl']['num_queries'] = 16
        cfg['u2pl']['num_negatives'] = 8
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
    generator = torch.Generator().manual_seed(cfg['seed'])
    labeled = common.base.StudentSegmentationDataset(
        common.resolve(cfg['labeled_train_manifest']), training=True, pseudo=False,
        augmentation=cfg['base_augmentation'])
    unlabeled = common.WeakStrongUnlabeledDataset(
        cfg['unlabeled_manifest'], cfg['base_augmentation'],
        cfg['strong_augmentation'], two_strong=False)
    labeled_loader = common.make_loader(
        labeled, cfg['supervised_batch_size'], True, cfg, generator, True)
    unlabeled_loader = common.make_loader(
        unlabeled, cfg['unlabeled_batch_size'], True, cfg, generator, True)
    student = U2PLModel(cfg, device).to(device)
    teacher = copy.deepcopy(student).to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    memory = MemoryBank(cfg['u2pl']['representation_channels'], device)
    class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32, device=device)
    history = []
    best_miou = -1.0
    best_epoch = 0
    best_metrics = None
    best_path = work_dir / 'best_validation_mIoU.pth'
    global_step = 0
    started = time.time()
    for epoch in range(1, cfg['max_epochs'] + 1):
        train_metrics, global_step = train_epoch(
            student, teacher, labeled_loader, unlabeled_loader, optimizer,
            scaler, memory, class_weights, device, cfg, epoch, global_step)
        selected = validation_model(student, teacher, epoch, cfg)
        validation, _ = common.evaluate_split(
            selected.segmentor, cfg, cfg['validation_manifest'], device)
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
                best_path, selected, epoch, validation, 'U2PL-aligned')
        common.save_checkpoint(
            work_dir / 'last_checkpoint.pth', selected, epoch, validation,
            'U2PL-aligned')
    frozen = U2PLModel(cfg, device).to(device)
    common.load_checkpoint(frozen, best_path, device)
    test_metrics, test_rows = common.evaluate_split(
        frozen.segmentor, cfg, cfg['test_manifest'], device,
        save_dir=work_dir / 'test_predictions')
    common.write_rows(work_dir / 'test_per_image.csv', test_rows)
    summary = {
        'method': 'U2PL-aligned',
        'official_method_core': [
            'EMA teacher pseudo labels',
            'adaptive entropy split for reliable and unreliable pixels',
            'reliable-pixel unsupervised segmentation loss',
            'class-wise memory-bank contrastive learning from unreliable pixels'],
        'three_class_rank_adaptation': (
            'Official low/high rank 3:20 targets 21/19-class benchmarks; '
            'for three classes it is deterministically mapped to ranks 1:3, '
            'i.e. classes outside Top-1.'),
        'protocol': cfg['protocol_name'],
        'seed': cfg['seed'],
        'labeled_train_images': len(labeled),
        'unlabeled_train_images': len(unlabeled),
        'epochs_completed': cfg['max_epochs'],
        'steps_per_epoch': cfg['steps_per_epoch'],
        'total_optimizer_steps': global_step,
        'best_epoch': best_epoch,
        'best_validation_metrics': best_metrics,
        'frozen_test_metrics': test_metrics,
        'final_memory_bank_sizes': memory.sizes(),
        'model_selection': 'validation mIoU only',
        'test_tuning': False,
        'elapsed_seconds': time.time() - started,
        'adaptation_scope': cfg['claim_scope'],
    }
    common.write_json(work_dir / 'training_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

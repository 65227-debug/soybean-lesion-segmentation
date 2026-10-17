#!/usr/bin/env python
"""Train an independent EfficientNet-B0 disease semantic classifier."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFile
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, self.samples[index][0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/dadou_semantic')
    parser.add_argument(
        '--work-dir', default='work_dirs/dadou_efficientnet_b0_semantic')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--freeze-epochs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=20260802)
    parser.add_argument('--no-pretrained', action='store_true')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(input_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            input_size, scale=(0.65, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, validation_transform


def set_backbone_trainable(model, trainable):
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []
    amp_enabled = device.type == 'cuda'

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_probabilities.extend(probabilities.detach().cpu().tolist())
        all_paths.extend(paths)

    macro_f1 = f1_score(
        all_labels, all_predictions, average='macro', zero_division=0)
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'macro_f1': macro_f1,
        'labels': all_labels,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'paths': all_paths,
    }


def save_epoch_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training(rows, output_dir, best_epoch):
    epochs = [row['epoch'] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=180)
    axes[0].plot(epochs, [row['train_loss'] for row in rows], label='Train loss')
    axes[0].plot(epochs, [row['val_loss'] for row in rows], label='Validation loss')
    axes[0].set(title='Disease classifier loss', xlabel='Epoch', ylabel='Loss')
    axes[1].plot(
        epochs, [100 * row['val_accuracy'] for row in rows], label='Accuracy')
    axes[1].plot(
        epochs, [100 * row['val_macro_f1'] for row in rows], label='Macro F1')
    axes[1].set(
        title='Validation classification performance',
        xlabel='Epoch', ylabel='Score (%)')
    for axis in axes:
        axis.axvline(
            best_epoch, color='black', linestyle='--', linewidth=1,
            label=f'Best epoch ({best_epoch})')
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(output_dir / f'training_curves{suffix}', bbox_inches='tight')
    plt.close(fig)


def plot_confusion(matrix, class_names, output_dir):
    fig, axis = plt.subplots(figsize=(5.2, 4.6), dpi=180)
    image = axis.imshow(matrix, cmap='Blues')
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted class', ylabel='True class',
        title='Validation confusion matrix')
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]),
                      ha='center', va='center', color='black')
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(output_dir / f'confusion_matrix{suffix}', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_transform, validation_transform = build_transforms(args.input_size)
    train_dataset = ImageFolderWithPaths(data_root / 'train', train_transform)
    validation_dataset = ImageFolderWithPaths(
        data_root / 'val', validation_transform)
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError('Train/validation class mappings differ')
    class_names = train_dataset.classes
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0)

    weights = None if args.no_pretrained else EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.to(device)
    if args.freeze_epochs:
        set_backbone_trainable(model, False)

    counts = np.bincount(train_dataset.targets, minlength=len(class_names))
    class_weights = counts.sum() / (len(class_names) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    rows = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = work_dir / 'best_macro_f1.pth'
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
        train_result = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler)
        with torch.no_grad():
            validation_result = run_epoch(
                model, validation_loader, criterion, device)
        row = {
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'backbone_trainable': epoch > args.freeze_epochs,
            'train_loss': train_result['loss'],
            'train_accuracy': train_result['accuracy'],
            'train_macro_f1': train_result['macro_f1'],
            'val_loss': validation_result['loss'],
            'val_accuracy': validation_result['accuracy'],
            'val_macro_f1': validation_result['macro_f1'],
        }
        rows.append(row)
        save_epoch_csv(work_dir / 'epoch_metrics.csv', rows)
        print(json.dumps(row, ensure_ascii=False))

        if validation_result['macro_f1'] > best_f1 + 1e-4:
            best_f1 = validation_result['macro_f1']
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'arch': 'efficientnet_b0',
                'state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'input_size': args.input_size,
                'normalization': {
                    'mean': IMAGENET_MEAN,
                    'std': IMAGENET_STD,
                },
                'layercam_feature_indices': (3, 5, 8),
                'imagenet_pretrained': not args.no_pretrained,
                'best_epoch': best_epoch,
                'best_val_macro_f1': best_f1,
            }, best_path)
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        final_validation = run_epoch(model, validation_loader, criterion, device)
    report = classification_report(
        final_validation['labels'], final_validation['predictions'],
        target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(
        final_validation['labels'], final_validation['predictions'],
        labels=list(range(len(class_names))))

    prediction_rows = []
    for path, label, prediction, probabilities in zip(
            final_validation['paths'], final_validation['labels'],
            final_validation['predictions'], final_validation['probabilities']):
        row = {
            'image_path': str(Path(path).resolve()),
            'true_class': class_names[label],
            'predicted_class': class_names[prediction],
            'correct': label == prediction,
        }
        for index, name in enumerate(class_names):
            row[f'probability_{name}'] = probabilities[index]
        prediction_rows.append(row)
    with (work_dir / 'val_predictions.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    plot_training(rows, work_dir, best_epoch)
    plot_confusion(matrix, class_names, work_dir)
    summary = {
        'model': 'EfficientNet-B0',
        'independent_from_segmentation_models': True,
        'imagenet_pretrained': not args.no_pretrained,
        'classes': class_names,
        'train_images': len(train_dataset),
        'train_class_counts': {
            name: int(counts[index]) for index, name in enumerate(class_names)
        },
        'validation_images': len(validation_dataset),
        'epochs_completed': len(rows),
        'best_epoch': best_epoch,
        'best_val_accuracy_percent': 100 * final_validation['accuracy'],
        'best_val_macro_f1_percent': 100 * final_validation['macro_f1'],
        'best_val_loss': final_validation['loss'],
        'classification_report': report,
        'confusion_matrix': matrix.tolist(),
        'class_weights': class_weights.tolist(),
        'input_size': args.input_size,
        'test_split_used': False,
    }
    with (work_dir / 'training_summary.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Create publication-ready loss and validation metric curves."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='epoch_metrics.csv produced during training')
    parser.add_argument('--output', default=None)
    parser.add_argument('--target-class', default='lesion')
    parser.add_argument('--selection-metric', default='val_mIoU')
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output) if args.output else csv_path.with_name(
        'training_curves.png')
    frame = pd.read_csv(csv_path).sort_values('epoch')
    target_title = args.target_class.replace('_', ' ').title()
    best_index = frame[args.selection_metric].idxmax()
    best_epoch = int(frame.loc[best_index, 'epoch'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    loss_columns = [
        ('train_loss', 'Total loss'),
        ('train_decode.loss_ce', 'Main CE'),
        ('train_decode.loss_dice', 'Main Dice'),
        ('train_aux.loss_ce', 'Auxiliary CE'),
    ]
    for column, label in loss_columns:
        if column in frame and frame[column].notna().any():
            axes[0].plot(frame['epoch'], frame[column], label=label, linewidth=1.8)
    axes[0].set(title='Training loss convergence', xlabel='Epoch', ylabel='Loss')
    axes[0].grid(alpha=0.25)

    metric_columns = [
        ('val_mIoU', 'mIoU'),
        (f'val_IoU_{args.target_class}', f'{target_title} IoU'),
        (f'val_Dice_{args.target_class}', f'{target_title} Dice'),
        ('val_mDice', 'mDice'),
    ]
    for column, label in metric_columns:
        if column in frame and frame[column].notna().any():
            axes[1].plot(frame['epoch'], frame[column], label=label, linewidth=1.8)
    axes[1].set(
        title='Validation performance', xlabel='Epoch', ylabel='Score (%)')
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.axvline(
            best_epoch,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.6,
            label=f'Best epoch ({best_epoch})')
        axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')

    best_row = frame.loc[[best_index]]
    best_csv_path = csv_path.with_name('best_epoch_metrics.csv')
    best_row.to_csv(best_csv_path, index=False, encoding='utf-8-sig')
    summary = {
        'epochs_completed': int(frame['epoch'].max()),
        'selection_metric': args.selection_metric,
        'best_epoch': best_epoch,
        'best_val_mIoU_percent': float(frame.loc[best_index, 'val_mIoU']),
        f'best_val_{args.target_class}_IoU_percent': float(
            frame.loc[best_index, f'val_IoU_{args.target_class}']),
        f'best_val_{args.target_class}_Dice_percent': float(
            frame.loc[best_index, f'val_Dice_{args.target_class}']),
        'first_epoch_train_loss': float(frame.iloc[0]['train_loss']),
        'best_epoch_train_loss': float(frame.loc[best_index, 'train_loss']),
        'final_epoch_train_loss': float(frame.iloc[-1]['train_loss']),
    }
    summary_path = csv_path.with_name('training_summary.json')
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(output_path.resolve())
    print(pdf_path.resolve())
    print(best_csv_path.resolve())
    print(summary_path.resolve())


if __name__ == '__main__':
    main()

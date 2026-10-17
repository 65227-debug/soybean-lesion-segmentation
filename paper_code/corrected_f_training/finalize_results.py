#!/usr/bin/env python
"""Create paper-ready tables, plots and paired bootstrap results for final G."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OLD_DIR = ROOT / 'work_dirs/dadou_ablation_comparison'
FINAL_DIR = ROOT / 'work_dirs/dadou_corrected_F_final_test'
TRAIN_DIR = ROOT / 'work_dirs/dadou_corrected_F_distilled'
CALIBRATION_DIR = ROOT / 'work_dirs/dadou_corrected_F_distilled_calibrated_fine'
OUTPUT_DIR = ROOT / 'work_dirs/dadou_final_G_comparison'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(reference_rows, group_rows, metric, rng,
                     resamples=20000):
    reference = {row['sample_id']: row for row in reference_rows}
    group = {row['sample_id']: row for row in group_rows}
    differences = []
    for sample_id in sorted(set(reference) & set(group)):
        value_reference = float(reference[sample_id][metric])
        value_group = float(group[sample_id][metric])
        if np.isfinite(value_reference) and np.isfinite(value_group):
            differences.append(value_reference - value_group)
    differences = np.asarray(differences, dtype=np.float64)
    indices = rng.integers(
        0, len(differences), size=(resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        'paired_images': len(differences),
        'mean_difference_percentage_points': 100 * differences.mean(),
        'ci95_lower_percentage_points': 100 * lower,
        'ci95_upper_percentage_points': 100 * upper,
        'ci_excludes_zero': bool(lower > 0 or upper < 0),
        'resamples': resamples,
    }


def comparison_rows():
    old = read_csv(OLD_DIR / 'ABCDEF_统一测试对比.csv')
    rows = []
    for row in old:
        rows.append({
            'group': row['group'],
            'method': {
                'A': 'Supervised only',
                'B': 'Teacher confidence',
                'C': 'EMA Mean Teacher',
                'D': 'Confidence + leaf gate',
                'E': 'Confidence + semantic gate',
                'F': 'Original dual-evidence gate',
            }[row['group']],
            'test_mIoU': float(row['test_mIoU']),
            'test_IoU_lesion': float(row['test_IoU_lesion']),
            'test_Dice_lesion': float(row['test_Dice_lesion']),
            'test_Precision_lesion': float(row['test_Precision_lesion']),
            'test_Recall_lesion': float(row['test_Recall_lesion']),
        })
    final = json.loads(
        (FINAL_DIR / 'test_summary.json').read_text(encoding='utf-8'))['metrics']
    rows.append({
        'group': 'G',
        'method': 'Corrected F + soft distillation + validation calibration',
        'test_mIoU': final['mIoU'],
        'test_IoU_lesion': final['IoU_lesion'],
        'test_Dice_lesion': final['Dice_lesion'],
        'test_Precision_lesion': final['Precision_lesion'],
        'test_Recall_lesion': final['Recall_lesion'],
    })
    return rows


def plot_comparison(rows):
    labels = [row['group'] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    bars_miou = axis.bar(
        x - width / 2, [100 * row['test_mIoU'] for row in rows], width,
        label='mIoU')
    bars_lesion = axis.bar(
        x + width / 2, [100 * row['test_IoU_lesion'] for row in rows], width,
        label='Lesion IoU')
    for bar in (bars_miou[-1], bars_lesion[-1]):
        bar.set_hatch('///')
        bar.set_edgecolor('black')
        bar.set_linewidth(1.0)
    axis.set_xticks(x, labels)
    axis.set_xlabel('Ablation group')
    axis.set_ylabel('Test score (%)')
    axis.set_ylim(50, 88)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        figure.savefig(
            OUTPUT_DIR / f'ABCDEFG_test_comparison.{suffix}',
            bbox_inches='tight')
    plt.close(figure)


def plot_training():
    rows = read_csv(TRAIN_DIR / 'training_history.csv')
    epoch = np.asarray([int(row['epoch']) for row in rows])
    supervised = np.asarray([float(row['train_supervised_loss']) for row in rows])
    distillation = np.asarray([
        float(row['train_distillation_loss']) for row in rows])
    total = np.asarray([float(row['train_total_loss']) for row in rows])
    val_miou = np.asarray([100 * float(row['val_mIoU']) for row in rows])
    val_lesion = np.asarray([
        100 * float(row['val_IoU_lesion']) for row in rows])
    lambda_d = np.asarray([float(row['lambda_distill']) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), dpi=200)
    axes[0].plot(epoch, supervised, label='Supervised loss', linewidth=1.8)
    axes[0].plot(epoch, total, label='Total loss', linewidth=1.8)
    axes[0].plot(epoch, distillation, label='Distillation loss', linewidth=1.4)
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Loss')
    axes[1].plot(epoch, val_miou, label='Validation mIoU', linewidth=1.8)
    axes[1].plot(epoch, val_lesion, label='Validation lesion IoU', linewidth=1.8)
    axes[1].axvline(15, color='black', linestyle='--', linewidth=1,
                    label='Selected epoch (15)')
    axes[1].set(title='Validation metrics', xlabel='Epoch', ylabel='Score (%)')
    axes[2].plot(epoch, lambda_d, label='Distillation weight', linewidth=1.8)
    axes[2].set(title='Distillation schedule', xlabel='Epoch', ylabel='Weight')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        figure.savefig(
            OUTPUT_DIR / f'G_training_curves.{suffix}', bbox_inches='tight')
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows()
    write_csv(OUTPUT_DIR / 'ABCDEFG_统一测试对比.csv', rows)
    plot_comparison(rows)
    plot_training()

    reference_rows = read_csv(FINAL_DIR / 'test_per_image.csv')
    rng = np.random.default_rng(20260803)
    bootstrap_rows = []
    for group in 'ABCDEF':
        group_rows = read_csv(OLD_DIR / f'test_per_image_{group}.csv')
        for metric in ('mIoU', 'IoU_lesion'):
            bootstrap_rows.append({
                'comparison': f'G-{group}', 'metric': metric,
                **paired_bootstrap(reference_rows, group_rows, metric, rng),
                'seed': 20260803,
            })
    write_csv(OUTPUT_DIR / 'G_vs_ABCDEF_paired_bootstrap.csv', bootstrap_rows)

    best_old_miou = max(rows[:-1], key=lambda item: item['test_mIoU'])
    best_old_lesion = max(rows[:-1], key=lambda item: item['test_IoU_lesion'])
    final = rows[-1]
    calibration = json.loads(
        (CALIBRATION_DIR / 'calibration_summary.json').read_text(
            encoding='utf-8'))
    lines = [
        '# 修正版G最终结果（单学生模型）', '',
        '|组别|方法|测试mIoU|病斑IoU|病斑Dice|病斑Precision|病斑Recall|',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"|{row['group']}|{row['method']}|{100*row['test_mIoU']:.2f}%|"
            f"{100*row['test_IoU_lesion']:.2f}%|"
            f"{100*row['test_Dice_lesion']:.2f}%|"
            f"{100*row['test_Precision_lesion']:.2f}%|"
            f"{100*row['test_Recall_lesion']:.2f}%|")
    lines.extend([
        '', '## 指标比较', '',
        f"G组测试mIoU为{100*final['test_mIoU']:.2f}%，与原mIoU最高的"
        f"{best_old_miou['group']}组（{100*best_old_miou['test_mIoU']:.2f}%）"
        f"的差值（G减该组）为{100*(final['test_mIoU']-best_old_miou['test_mIoU']):.2f}个百分点。",
        f"G组病斑IoU为{100*final['test_IoU_lesion']:.2f}%，与原F组的差值（G减F）为"
        f"{100*(final['test_IoU_lesion']-rows[5]['test_IoU_lesion']):.2f}个百分点；"
        f"与原病斑IoU最高的{best_old_lesion['group']}组的差值（G减该组）为"
        f"{100*(final['test_IoU_lesion']-best_old_lesion['test_IoU_lesion']):.2f}个百分点。",
        f"G组在验证集选择的mIoU为"
        f"{100*calibration['selected']['metrics']['mIoU']:.2f}%，"
        '测试集只在权重与类别偏置冻结后评估一次。',
        '', '## 方法与报告边界', '',
        'G组最终仍为单个DeepLabV3+ + MobileNetV3-Large，类别偏置已折叠进分类头，'
        '不增加参数量和推理次数。训练中使用A与修正版F的软预测作为蒸馏教师，'
        '推理时不需要双模型。',
        'G相对A–F包含修正门控、软蒸馏和验证集校准；各组件贡献应通过'
        '“修正门控→蒸馏→校准”的逐项消融和相同预设种子的比较计算。',
    ])
    (OUTPUT_DIR / '论文结果说明.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

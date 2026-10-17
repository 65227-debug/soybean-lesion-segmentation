#!/usr/bin/env python
"""Evaluate frozen A/B/C/D/E/F checkpoints on the test split once."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


STEP9_DIR = Path(__file__).resolve().parents[1] / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


GROUPS = {
    'A': 'work_dirs/dadou_ablation_A',
    'B': 'work_dirs/dadou_ablation_B',
    'C': 'work_dirs/dadou_ablation_C',
    'D': 'work_dirs/dadou_ablation_D',
    'E': 'work_dirs/dadou_ablation_E',
    'F': 'work_dirs/dadou_student_deeplabv3plus_mobilenetv3_full60',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/dadou_student')
    parser.add_argument(
        '--model-config',
        default='paper_code/step9_student_training/mmseg_student_config.py')
    parser.add_argument(
        '--training-config',
        default='paper_code/ablation_abdec/ablation_config.json')
    parser.add_argument('--output-dir', default='work_dirs/dadou_ablation_comparison')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    missing = []
    for group, work_dir in GROUPS.items():
        for name in ('best_validation_mIoU.pth', 'training_summary.json'):
            path = Path(work_dir) / name
            if not path.is_file():
                missing.append(f'{group}: {path}')
    if missing:
        raise FileNotFoundError(
            'All groups must be frozen before test evaluation:\n'
            + '\n'.join(missing))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.training_config).read_text(encoding='utf-8'))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = common.StudentSegmentationDataset(
        Path(args.data_root) / 'supervised_test_manifest.csv', training=False)
    loader = DataLoader(
        dataset, batch_size=config['validation_batch_size'], shuffle=False,
        num_workers=config['workers'], pin_memory=device.type == 'cuda',
        persistent_workers=config['workers'] > 0,
        worker_init_fn=common.seed_worker)
    class_weights = torch.tensor(
        config['class_weights'], dtype=torch.float32, device=device)
    rows = []
    details = {}
    for group, work_dir in GROUPS.items():
        summary = json.loads(
            (Path(work_dir) / 'training_summary.json').read_text(
                encoding='utf-8'))
        checkpoint = torch.load(
            Path(work_dir) / 'best_validation_mIoU.pth', map_location=device,
            weights_only=True)
        model = common.build_student(args.model_config, device)
        model.load_state_dict(checkpoint['state_dict'])
        metrics, per_image = common.evaluate(
            model, loader, device, class_weights,
            save_dir=output_dir / 'predictions' / group)
        common.write_csv(output_dir / f'test_per_image_{group}.csv', per_image)
        validation = summary['best_validation_metrics']
        row = {
            'group': group,
            'best_epoch': summary['best_epoch'],
            'val_mIoU': validation['mIoU'],
            'val_IoU_lesion': validation['IoU_lesion'],
            'test_mIoU': metrics['mIoU'],
            'test_mDice': metrics['mDice'],
            'test_pixel_accuracy': metrics['pixel_accuracy'],
            'test_IoU_background': metrics['IoU_background'],
            'test_IoU_healthy_leaf': metrics['IoU_healthy_leaf'],
            'test_IoU_lesion': metrics['IoU_lesion'],
            'test_Dice_lesion': metrics['Dice_lesion'],
            'test_Precision_lesion': metrics['Precision_lesion'],
            'test_Recall_lesion': metrics['Recall_lesion'],
        }
        rows.append(row)
        details[group] = {'validation': validation, 'test': metrics}
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    common.write_csv(output_dir / 'ABCDEF_统一测试对比.csv', rows)
    (output_dir / 'comparison_summary.json').write_text(
        json.dumps({
            'groups': details,
            'test_images': len(dataset),
            'test_loaded_after_all_checkpoints_frozen': True,
        }, ensure_ascii=False, indent=2), encoding='utf-8')

    labels = [row['group'] for row in rows]
    miou = [100 * row['test_mIoU'] for row in rows]
    lesion = [100 * row['test_IoU_lesion'] for row in rows]
    x = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    width = 0.36
    axis.bar([item - width / 2 for item in x], miou, width, label='mIoU')
    axis.bar([item + width / 2 for item in x], lesion, width,
             label='Lesion IoU')
    axis.set_xticks(x, labels)
    axis.set_xlabel('Ablation group')
    axis.set_ylabel('Test score (%)')
    axis.set_ylim(0, 100)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / 'ablation_comparison.png', bbox_inches='tight')
    figure.savefig(output_dir / 'ablation_comparison.pdf', bbox_inches='tight')
    plt.close(figure)

    descriptions = {
        'A': 'supervised only',
        'B': 'teacher confidence pseudo labels',
        'C': 'EMA Mean Teacher',
        'D': 'confidence + leaf spatial gate',
        'E': 'confidence + disease semantic gate',
        'F': 'complete dual-evidence gate',
    }
    lines = ['# A/B/C/D/E/F统一测试结果', '']
    lines.append('|组别|方法|mIoU|病斑IoU|病斑Dice|病斑Precision|病斑Recall|')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for row in rows:
        lines.append(
            f"|{row['group']}|{descriptions[row['group']]}|"
            f"{100 * row['test_mIoU']:.2f}%|"
            f"{100 * row['test_IoU_lesion']:.2f}%|"
            f"{100 * row['test_Dice_lesion']:.2f}%|"
            f"{100 * row['test_Precision_lesion']:.2f}%|"
            f"{100 * row['test_Recall_lesion']:.2f}%|")
    best_miou_row = max(rows, key=lambda item: item['test_mIoU'])
    best_lesion_row = max(rows, key=lambda item: item['test_IoU_lesion'])
    lines.extend([
        '',
        '所有组使用相同DeepLabV3+ + MobileNetV3-Large、相同精标划分、',
        '相同训练轮数和每轮优化步数。测试集在全部最佳验证权重冻结后统一加载。',
        '',
        '## 结果解释',
        '',
        f"测试mIoU最高的是{best_miou_row['group']}组"
        f"（{100 * best_miou_row['test_mIoU']:.2f}%），病斑IoU最高的是"
        f"{best_lesion_row['group']}组"
        f"（{100 * best_lesion_row['test_IoU_lesion']:.2f}%）。",
        '当前单次划分不能支持“完整门控F优于全部消融组”的结论。',
        '验证集与测试集排序差异较大，建议结合成对bootstrap结果，并在按数据来源、',
        '病种和采集场景分层后运行至少3个随机种子，报告均值±标准差。',
        '已经查看本测试集后，不应再据此调阈值并在同一测试集上作确认性结论。',
    ])
    (output_dir / '论文消融结果.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

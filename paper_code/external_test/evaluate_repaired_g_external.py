#!/usr/bin/env python
"""Evaluate the repaired G variants and document the post-hoc repair.

This is an exploratory repair run because the frozen external test result had
already been inspected before the repair was designed.  The script preserves
that distinction in every generated summary.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STEP9_DIR = PROJECT_ROOT / 'paper_code' / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


STUDENT_CONFIG = (
    PROJECT_ROOT / 'paper_code/step9_student_training/mmseg_student_config.py')
VARIANTS = {
    'G_old_calibrated': PROJECT_ROOT / (
        'work_dirs/dadou_corrected_F_distilled_calibrated_fine/'
        'best_calibrated_student.pth'),
    'G_old_uncalibrated': PROJECT_ROOT / (
        'work_dirs/dadou_corrected_F_distilled/best_validation_mIoU.pth'),
    'G4_source_uncalibrated': PROJECT_ROOT / (
        'work_dirs/dadou_corrected_G4_weight_soup/'
        'best_validation_weight_soup.pth'),
    'G_UDA_epoch12': PROJECT_ROOT / (
        'work_dirs/dadou_corrected_G_external_UDA_20260826/'
        'last_checkpoint.pth'),
    'G_repaired': PROJECT_ROOT / (
        'work_dirs/dadou_corrected_G_external_UDA_calibrated_floor785_20260826/'
        'best_unlabeled_teacher_calibrated.pth'),
}
REFERENCE_ROOT = PROJECT_ROOT / (
    'paper_materials/'
    'MH-SoyaHealthVision_External_Test_80_冻结评估_20260819')
CATEGORY_ORDER = (
    'Healthy', 'Frogeye_leaf_spot', 'Rust', 'Septoria_brown_spot')
PALETTE_BGR = np.asarray(
    ((0, 0, 0), (0, 180, 0), (30, 30, 220)), dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        default=str(PROJECT_ROOT / (
            'data/MH-SoyaHealthVision_External_Test_80_3class')))
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_ROOT / (
            'paper_materials/'
            'MH-SoyaHealthVision_G修复与重跑_探索性_20260826')))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260826)
    parser.add_argument(
        '--reuse-predictions', action='store_true',
        help='Reuse complete variant prediction folders in a non-empty output.')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_confusion(confusion, prediction, target):
    valid = target < 3
    encoded = target[valid].astype(np.int64) * 3 \
        + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=9).reshape(3, 3)


def metrics_from_confusion(confusion):
    result = common.metrics_from_confusion(confusion)
    result['confusion_matrix'] = np.asarray(confusion).tolist()
    return result


def evaluate_saved(rows, prediction_dir):
    overall = np.zeros((3, 3), dtype=np.int64)
    disease_only = np.zeros((3, 3), dtype=np.int64)
    categories = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    per_image = []
    per_image_confusions = []
    for row in rows:
        target = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        prediction = cv2.imread(
            str(prediction_dir / 'masks' / f"{row['sample_id']}.png"),
            cv2.IMREAD_UNCHANGED)
        if target is None or prediction is None or target.shape != prediction.shape:
            raise RuntimeError(f'Alignment failure: {row["sample_id"]}')
        confusion = np.zeros((3, 3), dtype=np.int64)
        update_confusion(confusion, prediction, target)
        overall += confusion
        categories[row['class_name']] += confusion
        if row['class_name'] != 'Healthy':
            disease_only += confusion
        metrics = metrics_from_confusion(confusion)
        per_image.append({
            'sample_id': row['sample_id'],
            'class_name': row['class_name'],
            'scene_id': row['scene_id'],
            **{key: value for key, value in metrics.items()
               if key != 'confusion_matrix'},
        })
        per_image_confusions.append(confusion)
    return {
        'overall': metrics_from_confusion(overall),
        'disease_only_60': metrics_from_confusion(disease_only),
        'by_category': {
            category: metrics_from_confusion(categories[category])
            for category in CATEGORY_ORDER},
        'per_image': per_image,
        'per_image_confusions': np.stack(per_image_confusions),
    }


def bootstrap_difference(reference, comparison, metric, resamples, rng):
    samples = len(reference)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, samples, size=samples)
        reference_metrics = metrics_from_confusion(
            reference[selected].sum(axis=0))
        comparison_metrics = metrics_from_confusion(
            comparison[selected].sum(axis=0))
        differences[index] = (
            reference_metrics[metric] - comparison_metrics[metric])
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {
        'metric': metric,
        'mean_difference_percentage_points': 100 * differences.mean(),
        'ci95_lower_percentage_points': 100 * lower,
        'ci95_upper_percentage_points': 100 * upper,
        'ci_excludes_zero': bool(lower > 0 or upper < 0),
        'resamples': resamples,
    }


def prediction_percent(metrics):
    confusion = np.asarray(metrics['confusion_matrix'], dtype=np.float64)
    return 100.0 * confusion.sum(axis=0) / confusion.sum()


def plot_progression(rows, output_dir):
    labels = [row['model_id'] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.8, 5.2), dpi=220)
    axis.bar(x - width / 2, [100 * row['mIoU'] for row in rows], width,
             label='mIoU')
    axis.bar(x + width / 2,
             [100 * row['IoU_lesion'] for row in rows], width,
             label='Lesion IoU')
    axis.set_xticks(x, labels, rotation=12, ha='right')
    axis.set_ylim(0, 100)
    axis.set_ylabel('External score (%)')
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(output_dir / f'01_G修复过程对比.{extension}',
                       bbox_inches='tight')
    plt.close(figure)


def plot_uda_training(history_path, output_dir):
    rows = read_csv(history_path)
    epochs = np.asarray([float(row['epoch']) for row in rows])
    supervised = np.asarray(
        [float(row['train_supervised_loss']) for row in rows])
    distillation = np.asarray(
        [float(row['train_distillation_loss']) for row in rows])
    validation = np.asarray([float(row['val_mIoU']) for row in rows])
    figure, (loss_axis, score_axis) = plt.subplots(
        2, 1, figsize=(8.8, 7.0), dpi=220, sharex=True,
        gridspec_kw={'height_ratios': (1.15, 1.0)})
    supervised_axis = loss_axis.twinx()
    loss_axis.plot(epochs, distillation, marker='o', linewidth=1.8,
                   color='#D55E00', label='Target distillation loss')
    supervised_axis.plot(
        epochs, supervised, marker='s', linewidth=1.8,
        color='#009E73', label='Source supervised loss')
    loss_axis.set_ylabel('Distillation loss', color='#D55E00')
    supervised_axis.set_ylabel('Supervised loss', color='#009E73')
    handles = loss_axis.get_lines() + supervised_axis.get_lines()
    loss_axis.legend(handles, [line.get_label() for line in handles],
                     frameon=False, loc='best')
    loss_axis.grid(alpha=0.22)
    score_axis.plot(epochs, 100 * validation, marker='D', linewidth=1.8,
                    color='#0072B2')
    score_axis.set_xlabel('Epoch')
    score_axis.set_ylabel('Source validation mIoU (%)', color='#0072B2')
    score_axis.grid(alpha=0.22)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(output_dir / f'02_无标签域适配训练曲线.{extension}',
                       bbox_inches='tight')
    plt.close(figure)


def qualitative(rows, output_dir):
    target_dir = output_dir / 'qualitative_all_80'
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = {
        'Teacher': REFERENCE_ROOT / 'predictions/Teacher_R50/masks',
        'B': REFERENCE_ROOT / 'predictions/B/masks',
        'Old G': output_dir / 'predictions/G_old_calibrated/masks',
        'Repaired G': output_dir / 'predictions/G_repaired/masks',
    }
    for row in rows:
        image = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        target = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        panels = [('Image', image), ('Ground truth', PALETTE_BGR[target])]
        for title, directory in source_dirs.items():
            prediction = cv2.imread(
                str(directory / f"{row['sample_id']}.png"),
                cv2.IMREAD_UNCHANGED)
            panels.append((title, PALETTE_BGR[prediction]))
        views = []
        for title, panel in panels:
            view = cv2.resize(panel, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.rectangle(view, (0, 0), (256, 30), (255, 255, 255), -1)
            cv2.putText(view, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (20, 20, 20), 1, cv2.LINE_AA)
            views.append(view)
        overview = np.concatenate(views, axis=1)
        cv2.imwrite(
            str(target_dir / f"{row['sample_id']}.jpg"), overview,
            [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if (output_dir.exists() and any(output_dir.iterdir())
            and not args.reuse_predictions):
        raise FileExistsError(f'Output is non-empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(data_root / 'external_test_manifest.csv')
    if len(rows) != 80:
        raise RuntimeError(f'Expected 80 test images, got {len(rows)}')
    required = [STUDENT_CONFIG, *VARIANTS.values()]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = common.StudentSegmentationDataset(
        data_root / 'external_test_manifest.csv', training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
        worker_init_fn=common.seed_worker)
    class_weights = torch.tensor(
        [0.4914, 0.6319, 1.8768], dtype=torch.float32, device=device)
    model = None
    results = {}
    for model_id, checkpoint_path in VARIANTS.items():
        prediction_dir = output_dir / 'predictions' / model_id
        masks = list((prediction_dir / 'masks').glob('*.png'))
        overlays = list((prediction_dir / 'overlays').glob('*.jpg'))
        complete = len(masks) == len(rows) and len(overlays) == len(rows)
        if args.reuse_predictions and complete:
            started = None
        else:
            if model is None:
                model = common.build_student(STUDENT_CONFIG, device)
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            started = time.time()
            common.evaluate(
                model, loader, device, class_weights, save_dir=prediction_dir)
        result = evaluate_saved(rows, prediction_dir)
        result['checkpoint'] = str(checkpoint_path)
        result['checkpoint_sha256'] = sha256(checkpoint_path)
        result['elapsed_seconds'] = (
            None if started is None else time.time() - started)
        results[model_id] = result
        print(json.dumps({
            'completed': model_id,
            'mIoU': result['overall']['mIoU'],
            'IoU_lesion': result['overall']['IoU_lesion'],
        }), flush=True)
    if model is not None:
        del model

    # Reuse the already frozen unchanged reference predictions.
    for model_id in ('B', 'Teacher_R50'):
        result = evaluate_saved(
            rows, REFERENCE_ROOT / 'predictions' / model_id)
        result['checkpoint'] = 'unchanged frozen reference; see 20260819 run'
        result['checkpoint_sha256'] = None
        result['elapsed_seconds'] = None
        results[model_id] = result

    comparison_order = [
        'Teacher_R50', 'B', 'G_old_calibrated', 'G_old_uncalibrated',
        'G4_source_uncalibrated', 'G_UDA_epoch12', 'G_repaired']
    comparison_rows = []
    category_rows = []
    per_image_rows = []
    for model_id in comparison_order:
        result = results[model_id]
        metrics = result['overall']
        proportions = prediction_percent(metrics)
        comparison_rows.append({
            'model_id': model_id,
            **{key: value for key, value in metrics.items()
               if key != 'confusion_matrix'},
            'predicted_background_percent': proportions[0],
            'predicted_healthy_percent': proportions[1],
            'predicted_lesion_percent': proportions[2],
        })
        for category in CATEGORY_ORDER:
            category_rows.append({
                'model_id': model_id,
                'class_name': category,
                **{key: value
                   for key, value in result['by_category'][category].items()
                   if key != 'confusion_matrix'},
            })
        for row in result['per_image']:
            per_image_rows.append({'model_id': model_id, **row})
    write_csv(output_dir / '01_修复前后总体指标.csv', comparison_rows)
    write_csv(output_dir / '02_修复后分病种指标.csv', category_rows)
    write_csv(output_dir / '03_修复前后逐图指标.csv', per_image_rows)

    disease_rows = []
    for model_id in comparison_order:
        metrics = results[model_id]['disease_only_60']
        disease_rows.append({
            'model_id': model_id,
            **{key: value for key, value in metrics.items()
               if key != 'confusion_matrix'},
        })
    write_csv(output_dir / '04_病害子集60张指标.csv', disease_rows)

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = []
    repaired_confusions = results['G_repaired']['per_image_confusions']
    for comparison in ('G_old_calibrated', 'B', 'Teacher_R50'):
        for metric in ('mIoU', 'IoU_lesion'):
            bootstrap_rows.append({
                'comparison': f'G_repaired-{comparison}',
                **bootstrap_difference(
                    repaired_confusions,
                    results[comparison]['per_image_confusions'],
                    metric, args.bootstrap_resamples, rng),
                'seed': args.seed,
            })
    write_csv(output_dir / '05_修复G配对Bootstrap.csv', bootstrap_rows)
    plot_progression(comparison_rows, output_dir)
    uda_work_dir = PROJECT_ROOT / (
        'work_dirs/dadou_corrected_G_external_UDA_20260826')
    plot_uda_training(uda_work_dir / 'training_history.csv', output_dir)
    qualitative(rows, output_dir)

    repaired = results['G_repaired']['overall']
    old = results['G_old_calibrated']['overall']
    summary = {
        'status': 'post-hoc exploratory repair; not an untouched confirmatory test',
        'frozen_test_images': 80,
        'test_manifest_sha256': sha256(
            data_root / 'external_test_manifest.csv'),
        'external_test_used_for_gradient_updates': False,
        'external_test_used_for_checkpoint_or_bias_selection': False,
        'unlabeled_adaptation_images': 80,
        'adaptation_test_image_sha256_overlap': 0,
        'adaptation_test_scene_overlap': 0,
        'old_G': old,
        'repaired_G': repaired,
        'absolute_improvement_percentage_points': {
            'mIoU': 100 * (repaired['mIoU'] - old['mIoU']),
            'IoU_lesion': 100 * (
                repaired['IoU_lesion'] - old['IoU_lesion']),
            'Precision_lesion': 100 * (
                repaired['Precision_lesion'] - old['Precision_lesion']),
        },
        'repaired_G_disease_only_60':
            results['G_repaired']['disease_only_60'],
        'reference_B': results['B']['overall'],
        'reference_teacher': results['Teacher_R50']['overall'],
    }
    (output_dir / '06_修复实验汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    provenance = {
        model_id: {
            'checkpoint': result['checkpoint'],
            'checkpoint_sha256': result['checkpoint_sha256'],
        }
        for model_id, result in results.items()
    }
    (output_dir / '00_模型与数据来源.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding='utf-8')
    code_dir = output_dir / 'code_used'
    code_dir.mkdir(parents=True, exist_ok=True)
    for source in (
            Path(__file__), SCRIPT_DIR / 'prepare_external_unlabeled_80.py',
            PROJECT_ROOT / (
                'paper_code/corrected_f_training/'
                'calibrate_student_to_unlabeled_teacher.py'),
            PROJECT_ROOT / (
                'paper_code/corrected_f_training/'
                'external_uda_config.json')):
        shutil.copy2(source, code_dir / source.name)
    records_dir = output_dir / 'training_and_calibration_records'
    records_dir.mkdir(parents=True, exist_ok=True)
    record_sources = {
        'UDA_training_history.csv': uda_work_dir / 'training_history.csv',
        'UDA_training_summary.json': uda_work_dir / 'training_summary.json',
        'UDA_config.json': PROJECT_ROOT / (
            'paper_code/corrected_f_training/external_uda_config.json'),
        'unlabeled_calibration_summary.json': PROJECT_ROOT / (
            'work_dirs/'
            'dadou_corrected_G_external_UDA_calibrated_floor785_20260826/'
            'calibration_summary.json'),
        'unlabeled_calibration_grid.csv': PROJECT_ROOT / (
            'work_dirs/'
            'dadou_corrected_G_external_UDA_calibrated_floor785_20260826/'
            'target_teacher_bias_grid.csv'),
        'adaptation_dataset_summary.json': PROJECT_ROOT / (
            'data/MH-SoyaHealthVision_External_Unlabeled_80_UDA/'
            'dataset_summary.json'),
        'adaptation_manifest.csv': PROJECT_ROOT / (
            'data/MH-SoyaHealthVision_External_Unlabeled_80_UDA/'
            'pseudo_train_manifest.csv'),
    }
    for target_name, source in record_sources.items():
        shutil.copy2(source, records_dir / target_name)

    old_predicted = prediction_percent(old)
    repaired_predicted = prediction_percent(repaired)
    lines = [
        '# G外部评估与无标签适配比较', '',
        '## 指标比较', '',
        f"旧G的外部mIoU为{100*old['mIoU']:.2f}%，病斑IoU为"
        f"{100*old['IoU_lesion']:.2f}%；修复G分别为"
        f"{100*repaired['mIoU']:.2f}%和{100*repaired['IoU_lesion']:.2f}%。",
        f"病斑Precision：旧G为{100*old['Precision_lesion']:.2f}%，适配G为"
        f"{100*repaired['Precision_lesion']:.2f}%，变化（适配G减旧G）为"
        f"{100*(repaired['Precision_lesion']-old['Precision_lesion']):.2f}个百分点。"
        f"预测病斑像素占比：旧G为{old_predicted[2]:.2f}%，适配G为"
        f"{repaired_predicted[2]:.2f}%，变化（适配G减旧G）为"
        f"{repaired_predicted[2]-old_predicted[2]:.2f}个百分点。",
        '', '## 适配步骤', '',
        '1. 旧G使用了在内部验证域选出的+1.1病斑logit偏置。',
        '2. 先移除域相关偏置，再使用与测试图像及scene均不重叠的80张公开图像做无标签教师蒸馏。',
        '3. 最后仅依据未标注适配图上的教师一致性选择类别偏置，并要求内部验证mIoU不低于78.5%。',
        '', '## 论文统计边界', '',
        '原20260819结果仍是一次性冻结零样本外部测试。修复是在看到该结果后设计的，因此必须标为“探索性无标签域适配”，不能替换成新的独立确认性外部测试。若需确认性结论，应另取未查看标签的新测试集。',
    ]
    (output_dir / 'README_诊断修复与论文使用说明.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

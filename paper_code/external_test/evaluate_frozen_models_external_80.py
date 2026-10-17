#!/usr/bin/env python
"""One-time frozen external evaluation on the MH-SoyaHealthVision 80 subset.

No external image or mask is used for training, calibration, threshold tuning,
checkpoint selection, or model selection. The seven student checkpoints A--G,
the three-class segmentation teacher, and the independent binary leaf model are
loaded exactly as frozen by the internal validation experiments.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmseg.apis import inference_model, init_model
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STEP9_DIR = PROJECT_ROOT / 'paper_code' / 'step9_student_training'
sys.path.insert(0, str(STEP9_DIR))
import train_student as common  # noqa: E402


STUDENT_CONFIG = PROJECT_ROOT / 'paper_code/step9_student_training/mmseg_student_config.py'
STUDENTS = {
    'A': {
        'method': 'Supervised only',
        'checkpoint': PROJECT_ROOT / 'work_dirs/dadou_ablation_A/best_validation_mIoU.pth'},
    'B': {
        'method': 'Teacher confidence',
        'checkpoint': PROJECT_ROOT / 'work_dirs/dadou_ablation_B/best_validation_mIoU.pth'},
    'C': {
        'method': 'EMA Mean Teacher',
        'checkpoint': PROJECT_ROOT / 'work_dirs/dadou_ablation_C/best_validation_mIoU.pth'},
    'D': {
        'method': 'Confidence + leaf gate',
        'checkpoint': PROJECT_ROOT / 'work_dirs/dadou_ablation_D/best_validation_mIoU.pth'},
    'E': {
        'method': 'Confidence + semantic gate',
        'checkpoint': PROJECT_ROOT / 'work_dirs/dadou_ablation_E/best_validation_mIoU.pth'},
    'F': {
        'method': 'Original dual-evidence gate',
        'checkpoint': PROJECT_ROOT / (
            'work_dirs/dadou_student_deeplabv3plus_mobilenetv3_full60/'
            'best_validation_mIoU.pth')},
    'G': {
        'method': 'Corrected gate + soft distillation + calibration',
        'checkpoint': PROJECT_ROOT / (
            'work_dirs/dadou_corrected_F_distilled_calibrated_fine/'
            'best_calibrated_student.pth')},
}
TEACHER = {
    'model_id': 'Teacher_R50',
    'method': 'DeepLabV3+ ResNet-50 segmentation teacher',
    'config': PROJECT_ROOT / (
        'work_dirs/dadou_deeplabv3plus_r50_teacher/'
        'deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py'),
    'checkpoint': PROJECT_ROOT / (
        'work_dirs/dadou_deeplabv3plus_r50_teacher/'
        'best_mIoU_epoch_23_inference.pth'),
}
LEAF = {
    'model_id': 'Leaf_UNet',
    'method': 'Independent U-Net leaf segmentation model',
    'config': PROJECT_ROOT / (
        'work_dirs/dadou_unet_leaf/'
        'unet-s5-d16_fcn_1xb4-60e_dadou-leaf-512x512.py'),
    'checkpoint': PROJECT_ROOT / (
        'work_dirs/dadou_unet_leaf/best_IoU_leaf_epoch_46_inference.pth'),
}
PALETTE_BGR = np.asarray(((0, 0, 0), (0, 180, 0), (30, 30, 220)),
                         dtype=np.uint8)
CATEGORY_ORDER = ('Healthy', 'Frogeye_leaf_spot', 'Rust',
                  'Septoria_brown_spot')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        default=str(PROJECT_ROOT / 'data/MH-SoyaHealthVision_External_Test_80_3class'))
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_ROOT / (
            'paper_materials/'
            'MH-SoyaHealthVision_External_Test_80_冻结评估_20260819')))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260819)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_confusion_numpy(confusion, prediction, target, classes=3):
    valid = (target >= 0) & (target < classes)
    encoded = target[valid].astype(np.int64) * classes \
        + prediction[valid].astype(np.int64)
    confusion += np.bincount(
        encoded, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(confusion, names=('background', 'healthy_leaf', 'lesion')):
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union,
                    out=np.full_like(true_positive, np.nan), where=union > 0)
    dice = np.divide(2 * true_positive, ground_truth + predicted,
                     out=np.full_like(true_positive, np.nan),
                     where=(ground_truth + predicted) > 0)
    precision = np.divide(true_positive, predicted,
                          out=np.full_like(true_positive, np.nan),
                          where=predicted > 0)
    recall = np.divide(true_positive, ground_truth,
                       out=np.full_like(true_positive, np.nan),
                       where=ground_truth > 0)
    result = {
        'mIoU': float(np.nanmean(iou)),
        'mDice': float(np.nanmean(dice)),
        'pixel_accuracy': float(true_positive.sum() / confusion.sum()),
    }
    for index, name in enumerate(names):
        result[f'IoU_{name}'] = float(iou[index])
        result[f'Dice_{name}'] = float(dice[index])
        result[f'Precision_{name}'] = float(precision[index])
        result[f'Recall_{name}'] = float(recall[index])
    result['confusion_matrix'] = confusion.tolist()
    return result


def binary_metrics(confusion):
    tn, fp, fn, tp = confusion.ravel().astype(np.float64)
    union = tp + fp + fn
    return {
        'pixel_accuracy': float((tp + tn) / confusion.sum()),
        'IoU_leaf': float(tp / union) if union else math.nan,
        'Dice_leaf': float(2 * tp / (2 * tp + fp + fn))
        if 2 * tp + fp + fn else math.nan,
        'Precision_leaf': float(tp / (tp + fp)) if tp + fp else math.nan,
        'Recall_leaf': float(tp / (tp + fn)) if tp + fn else math.nan,
        'confusion_matrix': confusion.tolist(),
    }


def prediction_paths(output_dir, model_id):
    root = output_dir / 'predictions' / model_id
    for name in ('masks', 'overlays'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def save_prediction(root, row, prediction):
    sample_id = row['sample_id']
    mask_path = root / 'masks' / f'{sample_id}.png'
    cv2.imwrite(str(mask_path), prediction.astype(np.uint8))
    image = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
    colour = PALETTE_BGR[prediction]
    overlay = cv2.addWeighted(image, 0.65, colour, 0.35, 0.0)
    cv2.imwrite(str(root / 'overlays' / f'{sample_id}.jpg'), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def evaluate_saved_predictions(model_id, method, rows, prediction_root):
    overall_confusion = np.zeros((3, 3), dtype=np.int64)
    category_confusions = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    per_image = []
    per_image_confusions = []
    for row in rows:
        target = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        prediction = cv2.imread(
            str(prediction_root / 'masks' / f"{row['sample_id']}.png"),
            cv2.IMREAD_UNCHANGED)
        if target is None or prediction is None or target.shape != prediction.shape:
            raise RuntimeError(f'Prediction alignment failure: {row["sample_id"]}')
        confusion = np.zeros((3, 3), dtype=np.int64)
        update_confusion_numpy(confusion, prediction, target)
        overall_confusion += confusion
        category_confusions[row['class_name']] += confusion
        metrics = metrics_from_confusion(confusion)
        per_image.append({
            'model_id': model_id,
            'method': method,
            'sample_id': row['sample_id'],
            'class_name': row['class_name'],
            'scene_id': row['scene_id'],
            **{key: value for key, value in metrics.items()
               if key != 'confusion_matrix'},
        })
        per_image_confusions.append(confusion)
    overall = metrics_from_confusion(overall_confusion)
    by_category = {
        category: metrics_from_confusion(category_confusions[category])
        for category in CATEGORY_ORDER}
    return overall, by_category, per_image, np.stack(per_image_confusions)


def run_students(rows, data_root, output_dir, device, batch_size, workers):
    dataset = common.StudentSegmentationDataset(
        data_root / 'external_test_manifest.csv', training=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == 'cuda', persistent_workers=workers > 0,
        worker_init_fn=common.seed_worker)
    class_weights = torch.tensor(
        [0.4914, 0.6319, 1.8768], dtype=torch.float32, device=device)
    results = {}
    total_parameters = None
    for model_id, specification in STUDENTS.items():
        started = time.time()
        checkpoint = torch.load(
            specification['checkpoint'], map_location='cpu', weights_only=True)
        model = common.build_student(STUDENT_CONFIG, device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if total_parameters is None:
            total_parameters = parameters
        elif total_parameters != parameters:
            raise RuntimeError('A--G student parameter counts differ')
        root = prediction_paths(output_dir, model_id)
        metrics, _ = common.evaluate(
            model, loader, device, class_weights, save_dir=root)
        overall, by_category, per_image, confusions = \
            evaluate_saved_predictions(
                model_id, specification['method'], rows, root)
        for key, value in overall.items():
            if key != 'confusion_matrix' and abs(value - metrics[key]) > 1e-10:
                raise RuntimeError(
                    f'Independent metric mismatch for {model_id}, {key}')
        results[model_id] = {
            'method': specification['method'],
            'checkpoint': str(specification['checkpoint']),
            'parameters': parameters,
            'elapsed_seconds': time.time() - started,
            'overall': overall,
            'by_category': by_category,
            'per_image': per_image,
            'per_image_confusions': confusions,
        }
        del model, checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(json.dumps({
            'completed': model_id,
            'mIoU': overall['mIoU'],
            'IoU_lesion': overall['IoU_lesion'],
        }, ensure_ascii=False), flush=True)
    return results


def run_teacher(rows, output_dir, device):
    started = time.time()
    model = init_model(
        str(TEACHER['config']), str(TEACHER['checkpoint']), device=str(device))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    root = prediction_paths(output_dir, TEACHER['model_id'])
    for index, row in enumerate(rows, start=1):
        result = inference_model(model, row['image_path'])
        prediction = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        save_prediction(root, row, prediction)
        if index % 20 == 0:
            print(f'Teacher [{index}/{len(rows)}]', flush=True)
    overall, by_category, per_image, confusions = evaluate_saved_predictions(
        TEACHER['model_id'], TEACHER['method'], rows, root)
    result = {
        'method': TEACHER['method'],
        'checkpoint': str(TEACHER['checkpoint']),
        'parameters': parameters,
        'elapsed_seconds': time.time() - started,
        'overall': overall,
        'by_category': by_category,
        'per_image': per_image,
        'per_image_confusions': confusions,
    }
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(json.dumps({
        'completed': TEACHER['model_id'],
        'mIoU': overall['mIoU'],
        'IoU_lesion': overall['IoU_lesion'],
    }, ensure_ascii=False), flush=True)
    return result


def run_leaf(rows, output_dir, device):
    started = time.time()
    model = init_model(
        str(LEAF['config']), str(LEAF['checkpoint']), device=str(device))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    root = output_dir / 'predictions' / LEAF['model_id']
    (root / 'masks').mkdir(parents=True, exist_ok=True)
    (root / 'overlays').mkdir(parents=True, exist_ok=True)
    overall = np.zeros((2, 2), dtype=np.int64)
    category_confusions = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    per_image = []
    for index, row in enumerate(rows, start=1):
        result = inference_model(model, row['image_path'])
        prediction = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        target3 = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        target = (target3 > 0).astype(np.uint8)
        confusion = np.zeros((2, 2), dtype=np.int64)
        update_confusion_numpy(confusion, prediction, target, classes=2)
        overall += confusion
        category_confusions[row['class_name']] += confusion
        metrics = binary_metrics(confusion)
        per_image.append({
            'model_id': LEAF['model_id'],
            'sample_id': row['sample_id'],
            'class_name': row['class_name'],
            **{key: value for key, value in metrics.items()
               if key != 'confusion_matrix'},
        })
        cv2.imwrite(str(root / 'masks' / f"{row['sample_id']}.png"), prediction)
        image = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        colour = np.zeros_like(image)
        colour[prediction > 0] = (0, 180, 0)
        overlay = cv2.addWeighted(image, 0.65, colour, 0.35, 0.0)
        cv2.imwrite(str(root / 'overlays' / f"{row['sample_id']}.jpg"), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if index % 20 == 0:
            print(f'Leaf [{index}/{len(rows)}]', flush=True)
    metrics = binary_metrics(overall)
    result = {
        'method': LEAF['method'],
        'checkpoint': str(LEAF['checkpoint']),
        'parameters': parameters,
        'elapsed_seconds': time.time() - started,
        'overall': metrics,
        'by_category': {
            category: binary_metrics(category_confusions[category])
            for category in CATEGORY_ORDER},
        'per_image': per_image,
    }
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(json.dumps({
        'completed': LEAF['model_id'],
        'IoU_leaf': metrics['IoU_leaf'],
    }, ensure_ascii=False), flush=True)
    return result


def bootstrap_metrics(confusions, resamples, rng):
    samples = len(confusions)
    miou = np.empty(resamples, dtype=np.float64)
    lesion = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, samples, size=samples)
        metrics = metrics_from_confusion(confusions[selected].sum(axis=0))
        miou[index] = metrics['mIoU']
        lesion[index] = metrics['IoU_lesion']
    return {
        'mIoU_ci95_lower': float(np.quantile(miou, 0.025)),
        'mIoU_ci95_upper': float(np.quantile(miou, 0.975)),
        'IoU_lesion_ci95_lower': float(np.quantile(lesion, 0.025)),
        'IoU_lesion_ci95_upper': float(np.quantile(lesion, 0.975)),
    }


def paired_bootstrap(reference, comparison, metric, resamples, rng):
    metric_index = {'mIoU': None, 'IoU_lesion': 2}[metric]
    samples = len(reference)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, samples, size=samples)
        ref_metrics = metrics_from_confusion(reference[selected].sum(axis=0))
        cmp_metrics = metrics_from_confusion(comparison[selected].sum(axis=0))
        differences[index] = ref_metrics[metric] - cmp_metrics[metric]
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {
        'metric': metric,
        'mean_difference_percentage_points': float(100 * differences.mean()),
        'ci95_lower_percentage_points': float(100 * lower),
        'ci95_upper_percentage_points': float(100 * upper),
        'ci_excludes_zero': bool(lower > 0 or upper < 0),
        'resamples': resamples,
    }


def plot_comparison(rows, output_dir):
    ids = [row['model_id'] for row in rows]
    x = np.arange(len(ids))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.2, 5.2), dpi=220)
    axis.bar(x - width / 2, [100 * row['mIoU'] for row in rows], width,
             label='mIoU')
    axis.bar(x + width / 2, [100 * row['IoU_lesion'] for row in rows], width,
             label='Lesion IoU')
    axis.set_xticks(x, ids)
    axis.set_ylabel('External test score (%)')
    axis.set_ylim(0, 100)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(output_dir / f'01_外部测试模型对比.{extension}',
                       bbox_inches='tight')
    plt.close(figure)


def plot_g_by_category(by_category, output_dir):
    x = np.arange(len(CATEGORY_ORDER))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.5, 5.2), dpi=220)
    axis.bar(x - width / 2,
             [100 * by_category[c]['mIoU'] for c in CATEGORY_ORDER], width,
             label='mIoU')
    axis.bar(x + width / 2,
             [100 * by_category[c]['IoU_lesion'] for c in CATEGORY_ORDER],
             width, label='Lesion IoU')
    axis.set_xticks(x, ('Healthy', 'Frogeye', 'Rust', 'Septoria'))
    axis.set_ylabel('Final G score (%)')
    axis.set_ylim(0, 100)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(output_dir / f'02_最终G分病种结果.{extension}',
                       bbox_inches='tight')
    plt.close(figure)


def plot_confusion(confusion, output_dir):
    matrix = np.asarray(confusion, dtype=np.float64)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(6.2, 5.4), dpi=220)
    image = axis.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    labels = ('Background', 'Healthy', 'Lesion')
    axis.set_xticks(range(3), labels)
    axis.set_yticks(range(3), labels)
    axis.set_xlabel('Predicted class')
    axis.set_ylabel('Ground-truth class')
    for row in range(3):
        for column in range(3):
            value = normalized[row, column]
            axis.text(column, row, f'{100 * value:.1f}%', ha='center',
                      va='center', color='white' if value > 0.5 else 'black')
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(output_dir / f'03_最终G混淆矩阵.{extension}',
                       bbox_inches='tight')
    plt.close(figure)


def qualitative_overviews(rows, output_dir, model_ids):
    target_dir = output_dir / 'qualitative_all_models'
    target_dir.mkdir(parents=True, exist_ok=True)
    labels = ['Image', 'Ground truth'] + model_ids
    for row in rows:
        image = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
        target = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        panels = [image, PALETTE_BGR[target]]
        for model_id in model_ids:
            prediction = cv2.imread(
                str(output_dir / 'predictions' / model_id / 'masks'
                    / f"{row['sample_id']}.png"), cv2.IMREAD_UNCHANGED)
            panels.append(PALETTE_BGR[prediction])
        labelled = []
        for panel, label in zip(panels, labels):
            view = cv2.resize(panel, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.rectangle(view, (0, 0), (256, 30), (255, 255, 255), -1)
            cv2.putText(view, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (20, 20, 20), 1, cv2.LINE_AA)
            labelled.append(view)
        top = np.concatenate(labelled[:5], axis=1)
        bottom = np.concatenate(labelled[5:], axis=1)
        if bottom.shape[1] < top.shape[1]:
            padding = np.full(
                (bottom.shape[0], top.shape[1] - bottom.shape[1], 3),
                255, dtype=np.uint8)
            bottom = np.concatenate((bottom, padding), axis=1)
        overview = np.concatenate((top, bottom), axis=0)
        cv2.imwrite(str(target_dir / f"{row['sample_id']}.jpg"), overview,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output is non-empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(data_root / 'external_test_manifest.csv')
    if len(rows) != 80:
        raise RuntimeError(f'Expected 80 frozen external images, got {len(rows)}')
    counts = defaultdict(int)
    for row in rows:
        counts[row['class_name']] += 1
    if any(counts[name] != 20 for name in CATEGORY_ORDER):
        raise RuntimeError(f'External categories are not balanced: {dict(counts)}')

    required = [STUDENT_CONFIG, TEACHER['config'], TEACHER['checkpoint'],
                LEAF['config'], LEAF['checkpoint']]
    required += [specification['checkpoint'] for specification in STUDENTS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Frozen files missing:\n' + '\n'.join(missing))
    frozen_manifest = {
        'external_test_images': len(rows),
        'external_test_manifest_sha256': sha256(
            data_root / 'external_test_manifest.csv'),
        'model_selection_or_tuning_on_external_test': False,
        'student_config': {
            'path': str(STUDENT_CONFIG), 'sha256': sha256(STUDENT_CONFIG)},
        'models': {},
    }
    for model_id, specification in STUDENTS.items():
        frozen_manifest['models'][model_id] = {
            'method': specification['method'],
            'checkpoint': str(specification['checkpoint']),
            'sha256': sha256(specification['checkpoint']),
        }
    for specification in (TEACHER, LEAF):
        frozen_manifest['models'][specification['model_id']] = {
            'method': specification['method'],
            'config': str(specification['config']),
            'config_sha256': sha256(specification['config']),
            'checkpoint': str(specification['checkpoint']),
            'sha256': sha256(specification['checkpoint']),
        }
    (output_dir / '00_冻结模型与数据清单.json').write_text(
        json.dumps(frozen_manifest, ensure_ascii=False, indent=2),
        encoding='utf-8')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    started = time.time()
    results = run_students(
        rows, data_root, output_dir, device, args.batch_size, args.workers)
    results[TEACHER['model_id']] = run_teacher(rows, output_dir, device)
    leaf_result = run_leaf(rows, output_dir, device)

    comparison_order = [TEACHER['model_id']] + list(STUDENTS)
    rng = np.random.default_rng(args.seed)
    comparison_rows = []
    category_rows = []
    all_per_image = []
    for model_id in comparison_order:
        result = results[model_id]
        ci = bootstrap_metrics(
            result['per_image_confusions'], args.bootstrap_resamples, rng)
        metrics = result['overall']
        comparison_rows.append({
            'model_id': model_id,
            'method': result['method'],
            'mIoU': metrics['mIoU'],
            'mIoU_ci95_lower': ci['mIoU_ci95_lower'],
            'mIoU_ci95_upper': ci['mIoU_ci95_upper'],
            'mDice': metrics['mDice'],
            'pixel_accuracy': metrics['pixel_accuracy'],
            'IoU_background': metrics['IoU_background'],
            'IoU_healthy_leaf': metrics['IoU_healthy_leaf'],
            'IoU_lesion': metrics['IoU_lesion'],
            'IoU_lesion_ci95_lower': ci['IoU_lesion_ci95_lower'],
            'IoU_lesion_ci95_upper': ci['IoU_lesion_ci95_upper'],
            'Dice_lesion': metrics['Dice_lesion'],
            'Precision_lesion': metrics['Precision_lesion'],
            'Recall_lesion': metrics['Recall_lesion'],
            'parameters': result['parameters'],
            'elapsed_seconds_80_images': result['elapsed_seconds'],
        })
        for category in CATEGORY_ORDER:
            category_metrics = result['by_category'][category]
            category_rows.append({
                'model_id': model_id,
                'method': result['method'],
                'class_name': category,
                'images': 20,
                **{key: value for key, value in category_metrics.items()
                   if key != 'confusion_matrix'},
            })
        all_per_image.extend(result['per_image'])
        (output_dir / f'{model_id}_metrics.json').write_text(
            json.dumps({
                key: value for key, value in result.items()
                if key not in {'per_image', 'per_image_confusions'}},
                ensure_ascii=False, indent=2), encoding='utf-8')

    leaf_rows = []
    for category in CATEGORY_ORDER:
        leaf_rows.append({
            'class_name': category,
            'images': 20,
            **{key: value for key, value in leaf_result['by_category'][category].items()
               if key != 'confusion_matrix'},
        })
    write_csv(output_dir / '01_外部测试总体指标.csv', comparison_rows)
    write_csv(output_dir / '02_外部测试分病种指标.csv', category_rows)
    write_csv(output_dir / '03_外部测试逐图指标.csv', all_per_image)
    write_csv(output_dir / '04_叶片U-Net分病种指标.csv', leaf_rows)
    (output_dir / 'Leaf_UNet_metrics.json').write_text(
        json.dumps({key: value for key, value in leaf_result.items()
                    if key != 'per_image'}, ensure_ascii=False, indent=2),
        encoding='utf-8')

    bootstrap_rows = []
    reference = results['G']['per_image_confusions']
    for model_id in ('A', 'B', 'C', 'D', 'E', 'F'):
        for metric in ('mIoU', 'IoU_lesion'):
            bootstrap_rows.append({
                'comparison': f'G-{model_id}',
                **paired_bootstrap(
                    reference, results[model_id]['per_image_confusions'],
                    metric, args.bootstrap_resamples, rng),
                'seed': args.seed,
            })
    write_csv(output_dir / '05_G对A-F成对bootstrap.csv', bootstrap_rows)

    plot_comparison(comparison_rows, output_dir)
    plot_g_by_category(results['G']['by_category'], output_dir)
    plot_confusion(results['G']['overall']['confusion_matrix'], output_dir)
    qualitative_overviews(
        rows, output_dir, [TEACHER['model_id']] + list(STUDENTS))

    best_miou = max(comparison_rows, key=lambda row: row['mIoU'])
    best_lesion = max(comparison_rows, key=lambda row: row['IoU_lesion'])
    summary = {
        'dataset': 'MH-SoyaHealthVision frozen balanced external test subset',
        'candidate_pool_images': 160,
        'evaluated_images': 80,
        'images_per_category': 20,
        'external_test_used_for_training_tuning_or_selection': False,
        'student_groups_evaluated': list(STUDENTS),
        'additional_models_evaluated': [TEACHER['model_id'], LEAF['model_id']],
        'best_three_class_mIoU_model': best_miou['model_id'],
        'best_three_class_mIoU': best_miou['mIoU'],
        'best_lesion_IoU_model': best_lesion['model_id'],
        'best_lesion_IoU': best_lesion['IoU_lesion'],
        'final_G_metrics': results['G']['overall'],
        'leaf_UNet_metrics': leaf_result['overall'],
        'bootstrap_resamples': args.bootstrap_resamples,
        'bootstrap_seed': args.seed,
        'device': str(device),
        'total_elapsed_seconds': time.time() - started,
        'semantic_model_external_metric_status': (
            'not evaluated: EfficientNet-B0 was a training-time two-disease '
            'semantic evidence model; its label space does not cover Healthy, '
            'Rust, or Septoria and it is not used in final-G inference'),
    }
    (output_dir / '06_外部测试汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# MH-SoyaHealthVision 冻结外部测试结果', '',
        '本次使用从160张候选图中预先挑选并人工标注的平衡80张子集；每类20张。',
        '所有模型均使用内部验证集已冻结的权重，外部数据没有参与训练、校准、阈值选择或模型选择。',
        '',
        '|模型|方法|mIoU|病斑IoU|病斑Dice|病斑Precision|病斑Recall|',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in comparison_rows:
        lines.append(
            f"|{row['model_id']}|{row['method']}|{100*row['mIoU']:.2f}%|"
            f"{100*row['IoU_lesion']:.2f}%|{100*row['Dice_lesion']:.2f}%|"
            f"{100*row['Precision_lesion']:.2f}%|"
            f"{100*row['Recall_lesion']:.2f}%|")
    lines.extend([
        '', '## 报告边界', '',
        f"三分类mIoU最高模型为{best_miou['model_id']}"
        f"（{100*best_miou['mIoU']:.2f}%）；病斑IoU最高模型为"
        f"{best_lesion['model_id']}（{100*best_lesion['IoU_lesion']:.2f}%）。",
        f"最终G外部mIoU为{100*results['G']['overall']['mIoU']:.2f}%，"
        f"病斑IoU为{100*results['G']['overall']['IoU_lesion']:.2f}%。",
        f"独立叶片U-Net的外部叶片IoU为"
        f"{100*leaf_result['overall']['IoU_leaf']:.2f}%。",
        'EfficientNet-B0语义模型只有Frogeye/target-spot两个训练标签，不能在包含'
        'Healthy/Rust/Septoria的四类外部集上计算可比分类准确率；它是训练期伪标签证据，'
        '最终G推理不调用该模型。',
        '本结果是一次性外部评估，不应据此重新调阈值或选择新模型后再在同一80张上报告确认性结果。',
    ])
    (output_dir / 'README_论文使用说明.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

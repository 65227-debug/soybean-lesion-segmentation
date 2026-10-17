#!/usr/bin/env python
"""Export the complete, auditable final-G pseudo-label gating process.

This script is visualization-only. It reads the frozen selected configuration
and the formal fifth-step float32 NPZ maps, then invokes the exact gating
implementation used by the corrected F/final G experiment. It does not train,
tune, or infer any model.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CORRECTED_CODE_DIR = SCRIPT_PATH.parent
if str(CORRECTED_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CORRECTED_CODE_DIR))

from corrected_gating import (  # noqa: E402
    BACKGROUND,
    HEALTHY,
    IGNORE,
    LESION,
    gate_pixels,
    leaf_interior,
    load_config,
)


# These values are used only as a fail-fast audit of the user-specified frozen
# experiment. All actual gating and weighting calculations read `config`.
REQUIRED_FROZEN_CONFIG = {
    'lesion_teacher_threshold': 0.85,
    'lesion_semantic_threshold': 0.0,
    'healthy_teacher_threshold': 0.60,
    'lesion_negative_threshold': 0.20,
    'background_teacher_threshold': 0.60,
    'background_leaf_threshold': 0.15,
    'leaf_threshold': 0.50,
    'erosion_radius': 5,
    'semantic_weight_floor': 0.75,
    'min_leaf_inner_area_percent': 5.0,
    'min_positive_pixels': 16,
    'min_semantic_confidence': 0.60,
}

LABEL_COLOURS_BGR = {
    BACKGROUND: (20, 20, 20),
    HEALTHY: (45, 180, 55),
    LESION: (40, 40, 220),
    IGNORE: (225, 225, 225),
}

EXPECTED_SUMMARY = {
    'input_images': 1390,
    'accepted_images': 1354,
    'rejected_images': 36,
    'accepted_by_class': {'frogeye': 917, 'target_spot': 437},
    'accepted_unique_scenes': 293,
    'pixel_distribution_percent': {
        '0': 53.78866914103971,
        '1': 25.81881096303199,
        '2': 1.4526524959321916,
        '255': 18.939867399996107,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-root',
        default='.')
    parser.add_argument(
        '--base-maps', default='data/dadou/05_三张基础图_修正版')
    parser.add_argument(
        '--config',
        default=('paper_code/corrected_f_training/'
                 'selected_gating_config.json'))
    parser.add_argument(
        '--existing-result', default='data/dadou_student_corrected_f')
    parser.add_argument(
        '--output-dir',
        default='paper_materials/最终G_伪标签与权重流程_1390张')
    parser.add_argument('--audit-seed', type=int, default=20260811)
    return parser.parse_args()


def resolve_from_root(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        if not rows:
            raise ValueError(f'Cannot infer fields for empty CSV: {path}')
        fieldnames = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_config(config_path):
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    actual = raw.get('gating', {})
    differences = []
    for key, expected in REQUIRED_FROZEN_CONFIG.items():
        if key not in actual:
            differences.append(f'{key}: missing (expected {expected!r})')
        elif actual[key] != expected:
            differences.append(
                f'{key}: actual {actual[key]!r}, expected {expected!r}')
    extras = sorted(set(actual) - set(REQUIRED_FROZEN_CONFIG))
    if extras:
        differences.append(f'unexpected gating keys: {extras}')
    if differences:
        raise RuntimeError(
            'Frozen selected_gating_config.json differs from the required '
            'final-G specification:\n' + '\n'.join(differences))
    return raw


def uint16_probability(array):
    return np.clip(np.rint(np.clip(array, 0.0, 1.0) * 65535.0),
                   0, 65535).astype(np.uint16)


def heatmap(array):
    values = np.clip(np.rint(np.clip(array, 0.0, 1.0) * 255.0),
                     0, 255).astype(np.uint8)
    return cv2.applyColorMap(values, cv2.COLORMAP_JET)


def binary_view(mask, colour=(255, 255, 255)):
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask.astype(bool)] = colour
    return result


def label_colour(labels):
    result = np.empty((*labels.shape, 3), dtype=np.uint8)
    for value, colour in LABEL_COLOURS_BGR.items():
        result[labels == value] = colour
    return result


def overlay_labels(image, labels, alpha=0.45):
    result = image.copy()
    colour = label_colour(labels)
    valid = labels != IGNORE
    blended = cv2.addWeighted(image, 1.0 - alpha, colour, alpha, 0.0)
    result[valid] = blended[valid]
    return result


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Failed to write {path}')


def export_continuous(folder, filename, array):
    """Save the requested heatmap name plus its lossless uint16 companion."""
    base = folder / filename
    write_png(base, heatmap(array))
    write_png(base.with_name(f'{base.stem}_16位.png'), uint16_probability(array))


def labelled_panel(image, title, size=256):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    result = resized.copy()
    cv2.rectangle(result, (0, 0), (size, 31), (255, 255, 255), -1)
    cv2.putText(result, title, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return result


def make_overview(original, maps, accepted, reasons):
    panels = [
        (original, 'Original'),
        (heatmap(maps['p_background']), 'Pbackground'),
        (heatmap(maps['p_healthy']), 'Phealthy'),
        (heatmap(maps['p_lesion']), 'Plesion'),
        (heatmap(maps['p_leaf']), 'Pleaf'),
        (binary_view(maps['m_leaf'], (0, 190, 0)), 'Mleaf'),
        (binary_view(maps['m_inner'], (0, 190, 0)), 'Minner'),
        (heatmap(maps['a_sem']), 'Asem'),
        (binary_view(maps['background_candidate']), 'BG candidate'),
        (binary_view(maps['healthy_candidate']), 'Healthy candidate'),
        (binary_view(maps['lesion_candidate']), 'Lesion candidate'),
        (binary_view(maps['ignore_candidate']), 'Ignore'),
        (label_colour(maps['candidate_label']), 'Candidate label'),
        (heatmap(maps['w_background']), 'w_background'),
        (heatmap(maps['w_healthy']), 'w_healthy'),
        (heatmap(maps['w_lesion']), 'w_lesion'),
        (heatmap(maps['candidate_weight']), 'Candidate weight'),
        (label_colour(maps['final_label']), 'Final label'),
        (heatmap(maps['final_weight']), 'Final weight'),
        (overlay_labels(original, maps['final_label']), 'Final overlay'),
    ]
    rows = []
    for start in range(0, len(panels), 4):
        rows.append(np.concatenate([
            labelled_panel(image, title) for image, title in panels[start:start + 4]
        ], axis=1))
    grid = np.concatenate(rows, axis=0)
    width = grid.shape[1]
    header = np.full((55, width, 3), 255, dtype=np.uint8)
    status = 'ACCEPTED' if accepted else 'REJECTED: ' + '|'.join(reasons)
    cv2.putText(header, f'Final G corrected gating - {status}', (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2,
                cv2.LINE_AA)
    legend = np.full((52, width, 3), 255, dtype=np.uint8)
    x = 12
    for value, name in ((BACKGROUND, 'Background'), (HEALTHY, 'Healthy'),
                        (LESION, 'Lesion'), (IGNORE, 'Ignore')):
        cv2.rectangle(legend, (x, 12), (x + 28, 40),
                      LABEL_COLOURS_BGR[value], -1)
        cv2.putText(legend, name, (x + 36, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 1, cv2.LINE_AA)
        x += 215
    return np.concatenate([header, grid, legend], axis=0)


def create_colour_scale(path):
    width, height = 600, 92
    values = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (45, 1))
    bar = cv2.applyColorMap(values, cv2.COLORMAP_JET)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[:45] = bar
    cv2.putText(canvas, '0.0', (0, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, '0.5', (width // 2 - 15, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, '1.0', (width - 35, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                cv2.LINE_AA)
    write_png(path, canvas)


def rejection_reasons(row, masks, config):
    reasons = []
    if 100.0 * masks['inner'].mean() < config.min_leaf_inner_area_percent:
        reasons.append('leaf_area_too_small')
    if int(masks['lesion'].sum()) < config.min_positive_pixels:
        reasons.append('too_few_lesion_pixels')
    confidence = float(row.get('semantic_label_probability') or 1.0)
    if confidence < config.min_semantic_confidence:
        reasons.append('semantic_image_confidence_too_low')
    return reasons, confidence


def proportions(labels):
    return {
        'background_percent': 100.0 * float(np.mean(labels == BACKGROUND)),
        'healthy_percent': 100.0 * float(np.mean(labels == HEALTHY)),
        'lesion_percent': 100.0 * float(np.mean(labels == LESION)),
        'ignore_percent': 100.0 * float(np.mean(labels == IGNORE)),
    }


def expected_existing_sets(existing_dir):
    accepted_rows = read_csv(existing_dir / 'pseudo_train_manifest.csv')
    rejected_rows = read_csv(existing_dir / 'rejected_manifest.csv')
    accepted = {row['sample_id'] for row in accepted_rows}
    rejected = {
        f"{row['class_name']}__{Path(row['filename']).stem}": row['reasons']
        for row in rejected_rows
    }
    return accepted, rejected


def verify_existing_result(existing_dir, sample_id, labels, weights):
    old_label_path = existing_dir / 'masks' / f'{sample_id}.png'
    old_weight_path = existing_dir / 'weights' / f'{sample_id}.png'
    old_labels = cv2.imread(str(old_label_path), cv2.IMREAD_UNCHANGED)
    old_weights = cv2.imread(str(old_weight_path), cv2.IMREAD_UNCHANGED)
    if old_labels is None or old_weights is None:
        raise FileNotFoundError(
            f'Existing final-G output missing for {sample_id}')
    if not np.array_equal(old_labels, labels):
        raise RuntimeError(f'Label mismatch with existing G: {sample_id}')
    if not np.array_equal(old_weights, uint16_probability(weights)):
        raise RuntimeError(f'Weight mismatch with existing G: {sample_id}')


def markdown_readme(config_raw, summary, source_hashes, discrepancies):
    g = config_raw['gating']
    discrepancy_text = ('无；本次导出与原实验汇总一致。' if not discrepancies
                        else '\n'.join(f'- {item}' for item in discrepancies))
    return f"""# 最终 G 修正版三态伪标签与连续像素权重生成过程

本目录由冻结的最终 G 代码和配置逐张导出，仅进行可视化与一致性核验；没有修改阈值、重新调参或训练模型。

## 输入与版本

- 正式基础图：`data/dadou/05_三张基础图_修正版/`
- 门控实现：`paper_code/corrected_f_training/corrected_gating.py`
- 生成实现：`paper_code/corrected_f_training/generate_corrected_pseudo_labels.py`
- 冻结配置：`paper_code/corrected_f_training/selected_gating_config.json`
- 输入图像：{summary['input_images']} 张；保留 {summary['accepted_images']} 张；排除 {summary['rejected_images']} 张。
- 门控代码 SHA256：`{source_hashes['corrected_gating.py']}`
- 原生成代码 SHA256：`{source_hashes['generate_corrected_pseudo_labels.py']}`
- 冻结配置 SHA256：`{source_hashes['selected_gating_config.json']}`
- 本次导出脚本 SHA256：`{source_hashes['export_final_g_process.py']}`

## 冻结规则

- `M_leaf = p_leaf >= {g['leaf_threshold']}`
- `M_inner = erosion(M_leaf, radius={g['erosion_radius']})`
- lesion：`M_inner AND p_lesion >= {g['lesion_teacher_threshold']} AND A_sem >= {g['lesion_semantic_threshold']}`
- healthy：`M_inner AND p_healthy >= {g['healthy_teacher_threshold']} AND p_lesion <= {g['lesion_negative_threshold']}`
- background：`p_leaf <= {g['background_leaf_threshold']} AND p_background >= {g['background_teacher_threshold']}`
- 其余像素：`Ignore = 255`

连续权重只在对应候选像素上赋值：

- `w_bg = p_background * (1 - p_leaf)`
- `w_healthy = p_healthy * p_leaf`
- `w_lesion = p_lesion * ({g['semantic_weight_floor']} + {1.0 - g['semantic_weight_floor']:.2f} * A_sem)`
- Ignore 像素权重为 0。

整图排除条件：`M_inner` 面积小于 {g['min_leaf_inner_area_percent']}%，或病斑候选少于 {g['min_positive_pixels']} 像素，或图像级语义标签置信度低于 {g['min_semantic_confidence']}。排除样本仍保存候选结果，但最终标签全部为 255，最终权重全部为 0。

## 文件编码

- 概率图和权重图的无后缀版本是固定 0--1 映射的 JET 彩色热力图；根目录提供 `固定0-1_JET色标.png`。
- 同名 `_16位.png` 文件为无损数值图，编码为 `round(value * 65535)`。
- `08` 和 `12` 原始标签 PNG 保留真实值：背景 0、健康 1、病斑 2、Ignore 255。
- 彩色标签：背景深灰、健康绿色、病斑红色、Ignore 浅灰。
- `15_完整数值数组.npz` 保留 float32 概率/权重及原始布尔或 uint8 掩码。
- 为保证所有像素图严格对齐，单图均为 512×512；色标单独保存在根目录，没有嵌入单图。

## 与原实验汇总比较

{discrepancy_text}

本次保留样本像素比例：背景 {summary['pixel_distribution_percent']['0']:.6f}%，健康 {summary['pixel_distribution_percent']['1']:.6f}%，病斑 {summary['pixel_distribution_percent']['2']:.6f}%，Ignore {summary['pixel_distribution_percent']['255']:.6f}%。

`随机抽查核验.csv` 为固定随机种子 {summary['audit_seed']} 的分层随机抽查；生成期间同时对全部 {summary['input_images']} 张执行形状、标签集合、权重范围、Ignore 权重和整图排除规则核验，并对全部保留样本逐像素比对现有最终 G 标签与 16 位权重。
"""


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    base_dir = resolve_from_root(project_root, args.base_maps).resolve()
    config_path = resolve_from_root(project_root, args.config).resolve()
    existing_dir = resolve_from_root(project_root, args.existing_result).resolve()
    output_dir = resolve_from_root(project_root, args.output_dir).resolve()

    required_paths = [
        project_root,
        base_dir / '三张基础图生成清单.csv',
        config_path,
        existing_dir / 'pseudo_label_summary.json',
        existing_dir / 'pseudo_train_manifest.csv',
        existing_dir / 'rejected_manifest.csv',
        CORRECTED_CODE_DIR / 'corrected_gating.py',
        CORRECTED_CODE_DIR / 'generate_corrected_pseudo_labels.py',
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Required inputs missing:\n' + '\n'.join(missing))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f'Output directory already exists and is non-empty: {output_dir}')

    config_raw = validate_frozen_config(config_path)
    config = load_config(config_path)
    rows = read_csv(base_dir / '三张基础图生成清单.csv')
    if len(rows) != EXPECTED_SUMMARY['input_images']:
        raise RuntimeError(
            f'Expected 1390 inputs, found {len(rows)}; export stopped')
    existing_summary = json.loads(
        (existing_dir / 'pseudo_label_summary.json').read_text(encoding='utf-8'))
    existing_accepted, existing_rejected = expected_existing_sets(existing_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, output_dir / '本次实际执行的导出脚本.py')
    shutil.copy2(config_path, output_dir / 'selected_gating_config.json')
    create_colour_scale(output_dir / '固定0-1_JET色标.png')

    all_rows = []
    accepted_rows = []
    rejected_rows = []
    audit_records = []
    accepted_ids = set()
    rejected_reason_map = {}
    accepted_pixels = Counter()
    accepted_class_counts = Counter()
    accepted_scenes = Counter()
    failures = []

    rng = random.Random(args.audit_seed)
    row_ids = [
        f"{row['class_name']}__{Path(row['filename']).stem}" for row in rows
    ]
    planned_general_audit = set(rng.sample(row_ids, 10))
    planned_rejected_audit = set(rng.sample(sorted(existing_rejected), 4))
    planned_audit = planned_general_audit | planned_rejected_audit

    for index, row in enumerate(rows, start=1):
        class_name = row['class_name']
        stem = Path(row['filename']).stem
        sample_id = f'{class_name}__{stem}'
        source_path = Path(row['source_path'])
        npz_path = base_dir / row['numeric_npz']
        try:
            with np.load(npz_path) as source:
                p_teacher = source['p_teacher'].astype(np.float32)
                p_leaf = source['p_leaf'].astype(np.float32)
                a_sem = source['a_sem'].astype(np.float32)
                stored_inner = source['m_inner'].astype(bool)
            if p_teacher.shape[0] != 3:
                raise ValueError(f'Invalid teacher shape {p_teacher.shape}')
            shape = p_leaf.shape
            if shape != (512, 512) or p_teacher.shape[1:] != shape \
                    or a_sem.shape != shape:
                raise ValueError(
                    f'Unaligned input maps: teacher={p_teacher.shape}, '
                    f'leaf={p_leaf.shape}, semantic={a_sem.shape}')

            p_background, p_healthy, p_lesion = p_teacher
            m_leaf = p_leaf >= config.leaf_threshold
            m_inner = leaf_interior(p_leaf, config)
            if not np.array_equal(m_inner, stored_inner):
                raise RuntimeError(
                    f'Recomputed M_inner differs from base NPZ: {sample_id}')

            candidate_label, candidate_weight, masks = gate_pixels(
                p_teacher, p_leaf, a_sem, config)
            if not np.array_equal(masks['inner'], m_inner):
                raise RuntimeError(f'gate_pixels M_inner mismatch: {sample_id}')

            background_candidate = masks['background']
            healthy_candidate = masks['healthy']
            lesion_candidate = masks['lesion']
            ignore_candidate = candidate_label == IGNORE

            w_background = np.zeros(shape, dtype=np.float32)
            w_healthy = np.zeros(shape, dtype=np.float32)
            w_lesion = np.zeros(shape, dtype=np.float32)
            w_background[background_candidate] = (
                p_background[background_candidate]
                * (1.0 - p_leaf[background_candidate]))
            w_healthy[healthy_candidate] = (
                p_healthy[healthy_candidate] * p_leaf[healthy_candidate])
            w_lesion[lesion_candidate] = (
                p_lesion[lesion_candidate]
                * (config.semantic_weight_floor
                   + (1.0 - config.semantic_weight_floor)
                   * a_sem[lesion_candidate]))
            reconstructed_weight = np.clip(
                w_background + w_healthy + w_lesion, 0.0, 1.0)
            if not np.allclose(
                    reconstructed_weight, candidate_weight,
                    rtol=0.0, atol=1e-7):
                raise RuntimeError(f'Weight reconstruction mismatch: {sample_id}')

            reasons, semantic_confidence = rejection_reasons(row, masks, config)
            image_accepted = not reasons
            if image_accepted:
                final_label = candidate_label.copy()
                final_weight = candidate_weight.copy()
                category_root = output_dir / '01_保留样本'
                accepted_ids.add(sample_id)
                accepted_class_counts[class_name] += 1
                scene_id = row.get('scene_id', '') or sample_id
                accepted_scenes[scene_id] += 1
                for value in (BACKGROUND, HEALTHY, LESION, IGNORE):
                    accepted_pixels[value] += int(np.sum(final_label == value))
                verify_existing_result(
                    existing_dir, sample_id, final_label, final_weight)
            else:
                final_label = np.full(shape, IGNORE, dtype=np.uint8)
                final_weight = np.zeros(shape, dtype=np.float32)
                category_root = output_dir / '02_整图排除样本'
                rejected_reason_map[sample_id] = '|'.join(reasons)

            if sample_id in existing_accepted and not image_accepted:
                raise RuntimeError(
                    f'Current rule rejects existing accepted sample: {sample_id}')
            if sample_id in existing_rejected:
                old_reasons = existing_rejected[sample_id]
                if image_accepted or '|'.join(reasons) != old_reasons:
                    raise RuntimeError(
                        f'Rejection mismatch for {sample_id}: '
                        f'current={reasons}, existing={old_reasons}')

            allowed = {BACKGROUND, HEALTHY, LESION, IGNORE}
            actual_values = set(np.unique(final_label).tolist())
            if not actual_values <= allowed:
                raise RuntimeError(
                    f'Invalid final label values {actual_values}: {sample_id}')
            if not np.all(m_inner <= m_leaf):
                raise RuntimeError(f'M_inner is not a subset of M_leaf: {sample_id}')
            probability_error = float(np.max(
                np.abs(p_teacher.sum(axis=0) - 1.0)))
            if probability_error > 1e-5:
                raise RuntimeError(
                    f'Teacher probabilities do not sum to one: '
                    f'{sample_id}, max error={probability_error}')
            if float(final_weight.min()) < 0.0 or float(final_weight.max()) > 1.0:
                raise RuntimeError(f'Final weight outside [0,1]: {sample_id}')
            if np.any(final_weight[final_label == IGNORE] != 0.0):
                raise RuntimeError(f'Ignore pixel has nonzero weight: {sample_id}')
            if not image_accepted and (
                    np.any(final_label != IGNORE) or np.any(final_weight != 0.0)):
                raise RuntimeError(f'Rejected final output is nonempty: {sample_id}')

            sample_dir = category_root / class_name / stem
            sample_dir.mkdir(parents=True, exist_ok=False)
            original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError(f'Unreadable source image: {source_path}')
            original = cv2.resize(
                original, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

            maps = {
                'p_teacher': p_teacher,
                'p_background': p_background,
                'p_healthy': p_healthy,
                'p_lesion': p_lesion,
                'p_leaf': p_leaf,
                'a_sem': a_sem,
                'm_leaf': m_leaf,
                'm_inner': m_inner,
                'background_candidate': background_candidate,
                'healthy_candidate': healthy_candidate,
                'lesion_candidate': lesion_candidate,
                'ignore_candidate': ignore_candidate,
                'candidate_label': candidate_label,
                'candidate_weight': candidate_weight,
                'w_background': w_background,
                'w_healthy': w_healthy,
                'w_lesion': w_lesion,
                'final_label': final_label,
                'final_weight': final_weight,
            }

            write_png(sample_dir / '00_完整流程总览.png',
                      make_overview(original, maps, image_accepted, reasons))
            write_png(sample_dir / '01_原图.png', original)
            export_continuous(
                sample_dir, '02a_教师背景概率_Pbackground.png', p_background)
            export_continuous(
                sample_dir, '02b_教师健康概率_Phealthy.png', p_healthy)
            export_continuous(
                sample_dir, '02c_教师病斑概率_Plesion.png', p_lesion)
            export_continuous(sample_dir, '03_叶片概率_Pleaf.png', p_leaf)
            write_png(sample_dir / '04_腐蚀前叶片掩码_Mleaf.png',
                      m_leaf.astype(np.uint8) * 255)
            write_png(sample_dir / '05_叶片可信内部区域_Minner.png',
                      m_inner.astype(np.uint8) * 255)
            export_continuous(sample_dir, '06_病种语义响应_Asem.png', a_sem)
            write_png(sample_dir / '07a_背景候选区域.png',
                      background_candidate.astype(np.uint8) * 255)
            write_png(sample_dir / '07b_健康叶片候选区域.png',
                      healthy_candidate.astype(np.uint8) * 255)
            write_png(sample_dir / '07c_病斑候选区域.png',
                      lesion_candidate.astype(np.uint8) * 255)
            write_png(sample_dir / '07d_忽略区域.png',
                      ignore_candidate.astype(np.uint8) * 255)
            write_png(sample_dir / '08_候选三态伪标签_原始数值.png',
                      candidate_label)
            write_png(sample_dir / '09_候选三态伪标签_彩色.png',
                      label_colour(candidate_label))
            export_continuous(
                sample_dir, '10a_背景像素权重_wbg.png', w_background)
            export_continuous(
                sample_dir, '10b_健康像素权重_whealthy.png', w_healthy)
            export_continuous(
                sample_dir, '10c_病斑像素权重_wlesion.png', w_lesion)
            export_continuous(
                sample_dir, '11a_门控候选连续像素权重图.png', candidate_weight)
            export_continuous(
                sample_dir, '11_最终连续像素权重图.png', final_weight)
            write_png(sample_dir / '12_整图筛选后伪标签_原始数值.png',
                      final_label)
            write_png(sample_dir / '13_整图筛选后伪标签_彩色.png',
                      label_colour(final_label))
            write_png(sample_dir / '14_伪标签叠加原图.png',
                      overlay_labels(original, final_label))

            np.savez_compressed(
                sample_dir / '15_完整数值数组.npz',
                p_teacher=p_teacher.astype(np.float32),
                p_background=p_background.astype(np.float32),
                p_healthy=p_healthy.astype(np.float32),
                p_lesion=p_lesion.astype(np.float32),
                p_leaf=p_leaf.astype(np.float32),
                a_sem=a_sem.astype(np.float32),
                m_leaf=m_leaf.astype(bool),
                m_inner=m_inner.astype(bool),
                background_candidate=background_candidate.astype(bool),
                healthy_candidate=healthy_candidate.astype(bool),
                lesion_candidate=lesion_candidate.astype(bool),
                ignore_candidate=ignore_candidate.astype(bool),
                candidate_label=candidate_label.astype(np.uint8),
                candidate_weight=candidate_weight.astype(np.float32),
                w_background=w_background.astype(np.float32),
                w_healthy=w_healthy.astype(np.float32),
                w_lesion=w_lesion.astype(np.float32),
                final_label=final_label.astype(np.uint8),
                final_weight=final_weight.astype(np.float32),
                image_accepted=np.asarray(image_accepted, dtype=bool))

            candidate_stats = proportions(candidate_label)
            final_stats = proportions(final_label)
            lesion_values = final_weight[final_label == LESION]
            sample_stats = {
                'sample_id': sample_id,
                'class_name': class_name,
                'scene_id': row.get('scene_id', ''),
                'source_path': str(source_path),
                'numeric_source_npz': str(npz_path),
                'image_accepted': image_accepted,
                'rejection_reasons': reasons,
                'semantic_label_probability': semantic_confidence,
                'leaf_inner_percent': 100.0 * float(m_inner.mean()),
                'lesion_candidate_pixels': int(lesion_candidate.sum()),
                'candidate_pixel_distribution_percent': candidate_stats,
                'final_pixel_distribution_percent': final_stats,
                'candidate_mean_weight': float(candidate_weight.mean()),
                'final_mean_weight': float(final_weight.mean()),
                'mean_final_lesion_weight': (
                    float(lesion_values.mean()) if lesion_values.size else 0.0),
                'teacher_probability_sum_max_abs_error': probability_error,
                'final_weight_min': float(final_weight.min()),
                'final_weight_max': float(final_weight.max()),
                'array_shape': list(shape),
                'frozen_gating_config': config_raw['gating'],
            }
            (sample_dir / '16_样本统计.json').write_text(
                json.dumps(sample_stats, ensure_ascii=False, indent=2),
                encoding='utf-8')

            csv_row = {
                'sample_id': sample_id,
                'class_name': class_name,
                'scene_id': row.get('scene_id', ''),
                'source_path': str(source_path),
                'image_accepted': image_accepted,
                'rejection_reasons': '|'.join(reasons),
                **final_stats,
                'mean_weight': float(final_weight.mean()),
                'mean_lesion_weight': (
                    float(lesion_values.mean()) if lesion_values.size else 0.0),
                'semantic_label_probability': semantic_confidence,
                'sample_output_dir': str(sample_dir),
                'candidate_background_percent': candidate_stats['background_percent'],
                'candidate_healthy_percent': candidate_stats['healthy_percent'],
                'candidate_lesion_percent': candidate_stats['lesion_percent'],
                'candidate_ignore_percent': candidate_stats['ignore_percent'],
                'candidate_mean_weight': float(candidate_weight.mean()),
                'leaf_inner_percent': 100.0 * float(m_inner.mean()),
                'lesion_candidate_pixels': int(lesion_candidate.sum()),
            }
            all_rows.append(csv_row)
            (accepted_rows if image_accepted else rejected_rows).append(csv_row)

            if sample_id in planned_audit:
                written_label = cv2.imread(
                    str(sample_dir / '12_整图筛选后伪标签_原始数值.png'),
                    cv2.IMREAD_UNCHANGED)
                written_weight = cv2.imread(
                    str(sample_dir / '11_最终连续像素权重图_16位.png'),
                    cv2.IMREAD_UNCHANGED)
                audit_records.append({
                    'sample_id': sample_id,
                    'image_accepted': image_accepted,
                    'rejection_reasons': '|'.join(reasons),
                    'teacher_sum_max_abs_error': probability_error,
                    'minner_subset_mleaf': bool(np.all(m_inner <= m_leaf)),
                    'final_label_values': '|'.join(
                        str(value) for value in sorted(actual_values)),
                    'weight_min': float(final_weight.min()),
                    'weight_max': float(final_weight.max()),
                    'ignore_weight_all_zero': bool(np.all(
                        final_weight[final_label == IGNORE] == 0.0)),
                    'rejected_final_all_ignore': (
                        bool(np.all(final_label == IGNORE))
                        if not image_accepted else 'not_applicable'),
                    'rejected_final_weight_all_zero': (
                        bool(np.all(final_weight == 0.0))
                        if not image_accepted else 'not_applicable'),
                    'all_array_shapes_aligned_512x512': True,
                    'written_label_exact': bool(np.array_equal(
                        written_label, final_label)),
                    'written_weight_uint16_exact': bool(np.array_equal(
                        written_weight, uint16_probability(final_weight))),
                    'audit_passed': True,
                })
        except Exception as error:
            failures.append({
                'sample_index': index,
                'sample_id': sample_id,
                'source_path': str(source_path),
                'numeric_npz': str(npz_path),
                'error': repr(error),
            })
            raise

        if index % 25 == 0 or index == len(rows):
            print(
                f'[{index}/{len(rows)}] accepted={len(accepted_rows)} '
                f'rejected={len(rejected_rows)} failures={len(failures)}',
                flush=True)

    if accepted_ids != existing_accepted:
        raise RuntimeError(
            'Accepted sample set differs from existing corrected-F result')
    if rejected_reason_map != existing_rejected:
        raise RuntimeError(
            'Rejected sample set/reasons differ from existing corrected-F result')

    total_accepted_pixels = sum(accepted_pixels.values())
    distribution = {
        str(value): 100.0 * accepted_pixels[value] / total_accepted_pixels
        for value in (BACKGROUND, HEALTHY, LESION, IGNORE)
    }
    summary = {
        'task': '最终G修正版三态伪标签与连续像素权重完整过程导出',
        'input_images': len(rows),
        'generated_samples': len(all_rows),
        'accepted_images': len(accepted_rows),
        'rejected_images': len(rejected_rows),
        'accepted_by_class': dict(sorted(accepted_class_counts.items())),
        'accepted_unique_scenes': len(accepted_scenes),
        'maximum_images_from_one_scene': max(accepted_scenes.values()),
        'pixel_distribution_percent': distribution,
        'source_base_maps': str(base_dir),
        'source_existing_result': str(existing_dir),
        'output_directory': str(output_dir),
        'audit_seed': args.audit_seed,
        'random_audit_samples': len(audit_records),
        'all_samples_invariants_checked': True,
        'all_accepted_outputs_match_existing_g_pixelwise': True,
        'missing_or_abnormal_samples': len(failures),
        'configuration': config_raw,
    }

    discrepancies = []
    scalar_keys = (
        'input_images', 'accepted_images', 'rejected_images',
        'accepted_unique_scenes')
    for key in scalar_keys:
        if summary[key] != existing_summary[key]:
            discrepancies.append(
                f'{key}: export={summary[key]}, existing={existing_summary[key]}')
    if summary['accepted_by_class'] != existing_summary['accepted_by_class']:
        discrepancies.append(
            f"accepted_by_class: export={summary['accepted_by_class']}, "
            f"existing={existing_summary['accepted_by_class']}")
    for key, value in distribution.items():
        old = existing_summary['pixel_distribution_percent'][key]
        if abs(value - old) > 1e-10:
            discrepancies.append(
                f'pixel {key}: export={value}, existing={old}')
    summary['matches_existing_experiment_summary'] = not discrepancies
    summary['summary_discrepancies'] = discrepancies

    write_csv(output_dir / '全部1390张生成清单.csv', all_rows)
    write_csv(output_dir / '保留样本清单.csv', accepted_rows,
              fieldnames=list(all_rows[0]))
    write_csv(output_dir / '整图排除样本清单.csv', rejected_rows,
              fieldnames=list(all_rows[0]))
    write_csv(output_dir / '随机抽查核验.csv', audit_records)
    (output_dir / '生成汇总.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    source_hashes = {
        'corrected_gating.py': sha256(CORRECTED_CODE_DIR / 'corrected_gating.py'),
        'generate_corrected_pseudo_labels.py': sha256(
            CORRECTED_CODE_DIR / 'generate_corrected_pseudo_labels.py'),
        'selected_gating_config.json': sha256(config_path),
        'export_final_g_process.py': sha256(SCRIPT_PATH),
    }
    (output_dir / 'README_版本配置与公式说明.md').write_text(
        markdown_readme(config_raw, summary, source_hashes, discrepancies),
        encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if discrepancies:
        raise RuntimeError(
            'Export completed but differs from existing experiment summary; '
            'see README and 生成汇总.json')


if __name__ == '__main__':
    main()

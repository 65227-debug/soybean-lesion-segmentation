"""Core implementation for dual-evidence pseudo-label gating.

Label convention used by the three-class segmentation student:
    0: background
    1: healthy leaf tissue
    2: lesion
  255: ignored pixel
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np


BACKGROUND = 0
HEALTHY = 1
LESION = 2
IGNORE = 255


@dataclass
class GatingConfig:
    tau_p: float = 0.90
    tau_n: float = 0.10
    tau_s: float = 0.45
    tau_sn: float = 0.20
    leaf_threshold: float = 0.50
    background_leaf_threshold: float = 0.10
    erosion_radius: int = 5
    min_leaf_inner_area_percent: float = 5.0
    max_lesion_candidate_leaf_percent: float = 60.0
    min_overlap_ratio: float = 0.10
    min_semantic_confidence: float = 0.60
    min_blur_score: float = 20.0
    max_brightness_mean: float = 245.0
    min_positive_pixels: int = 16

    def validate(self):
        probability_fields = (
            'tau_p', 'tau_n', 'tau_s', 'tau_sn', 'leaf_threshold',
            'background_leaf_threshold', 'min_overlap_ratio',
            'min_semantic_confidence')
        for name in probability_fields:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be within [0, 1], got {value}')
        if self.tau_n >= self.tau_p:
            raise ValueError('tau_n must be lower than tau_p')
        if self.tau_sn >= self.tau_s:
            raise ValueError('tau_sn must be lower than tau_s')
        if self.background_leaf_threshold >= self.leaf_threshold:
            raise ValueError(
                'background_leaf_threshold must be lower than leaf_threshold')
        if self.erosion_radius < 0:
            raise ValueError('erosion_radius cannot be negative')
        if self.min_positive_pixels < 0:
            raise ValueError('min_positive_pixels cannot be negative')


def load_config(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    values = raw.get('gating', raw)
    allowed = {item.name for item in fields(GatingConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    config = GatingConfig(**values)
    config.validate()
    return config


def save_config(path, config, extra=None):
    output = {'gating': asdict(config)}
    if extra:
        output.update(extra)
    Path(path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')


def trustworthy_leaf_interior(p_leaf, config):
    leaf = (p_leaf >= config.leaf_threshold).astype(np.uint8)
    if config.erosion_radius == 0:
        return leaf
    width = 2 * config.erosion_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.erode(leaf, kernel)


def gate_pixels(p_seg, p_leaf, a_sem, config, healthy_image=False):
    """Return candidate labels, pixel weights, and the evidence masks."""
    if not (p_seg.shape == p_leaf.shape == a_sem.shape):
        raise ValueError(
            f'Base map shapes differ: {p_seg.shape}, {p_leaf.shape}, '
            f'{a_sem.shape}')
    inner = trustworthy_leaf_interior(p_leaf, config).astype(bool)
    trusted_background = p_leaf <= config.background_leaf_threshold
    if healthy_image:
        positive = np.zeros_like(inner)
        negative = (p_seg <= config.tau_n) & inner
    else:
        positive = (
            (p_seg >= config.tau_p)
            & (a_sem >= config.tau_s)
            & inner)
        negative = (
            (p_seg <= config.tau_n)
            & (a_sem <= config.tau_sn)
            & inner)

    labels = np.full(p_seg.shape, IGNORE, dtype=np.uint8)
    labels[trusted_background] = BACKGROUND
    labels[negative] = HEALTHY
    labels[positive] = LESION

    weights = np.zeros(p_seg.shape, dtype=np.float32)
    weights[trusted_background] = (
        (1.0 - p_leaf[trusted_background])
        * (1.0 - p_seg[trusted_background]))
    if healthy_image:
        weights[negative] = 1.0 - p_seg[negative]
    else:
        weights[negative] = (
            (1.0 - p_seg[negative]) * (1.0 - a_sem[negative]))
    weights[positive] = p_seg[positive] * a_sem[positive]
    weights = np.clip(weights, 0, 1).astype(np.float32)

    masks = {
        'inner': inner,
        'trusted_background': trusted_background,
        'positive': positive,
        'negative': negative,
        'segmentation_candidate': p_seg >= config.tau_p,
        'semantic_candidate': a_sem >= config.tau_s,
    }
    return labels, weights, masks


def calculate_image_statistics(labels, masks):
    total = labels.size
    inner_pixels = int(masks['inner'].sum())
    positive_pixels = int(masks['positive'].sum())
    segmentation_candidate = masks['segmentation_candidate']
    semantic_candidate = masks['semantic_candidate']
    overlap_intersection = int((
        segmentation_candidate & semantic_candidate & masks['inner']).sum())
    overlap_union = int((segmentation_candidate | semantic_candidate).sum())
    overlap_ratio = (
        overlap_intersection / overlap_union if overlap_union else 0.0)
    return {
        'leaf_inner_area_percent': 100.0 * inner_pixels / total,
        'positive_pixels': positive_pixels,
        'positive_area_percent': 100.0 * positive_pixels / total,
        'healthy_area_percent': 100.0 * np.mean(labels == HEALTHY),
        'background_area_percent': 100.0 * np.mean(labels == BACKGROUND),
        'ignored_area_percent_before_image_filter': 100.0 * np.mean(
            labels == IGNORE),
        'lesion_candidate_leaf_percent': (
            100.0 * int((segmentation_candidate & masks['inner']).sum())
            / inner_pixels if inner_pixels else 0.0),
        'semantic_segmentation_intersection_pixels': overlap_intersection,
        'semantic_segmentation_union_pixels': overlap_union,
        'semantic_segmentation_overlap_ratio': overlap_ratio,
    }


def _optional_float(metadata, key):
    value = metadata.get(key, '')
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assess_image_quality(statistics, metadata, config, healthy_image=False):
    """Apply the reproducible whole-image quality rules from step eight."""
    reasons = []
    if statistics['leaf_inner_area_percent'] < config.min_leaf_inner_area_percent:
        reasons.append('leaf_area_too_small')
    if (statistics['lesion_candidate_leaf_percent']
            > config.max_lesion_candidate_leaf_percent):
        reasons.append('lesion_area_too_large')
    if not healthy_image:
        if (statistics['semantic_segmentation_overlap_ratio']
                < config.min_overlap_ratio):
            reasons.append('semantic_segmentation_overlap_too_low')
        if statistics['positive_pixels'] < config.min_positive_pixels:
            reasons.append('too_few_positive_pixels')
        semantic_confidence = _optional_float(
            metadata, 'semantic_label_probability')
        if (semantic_confidence is not None
                and semantic_confidence < config.min_semantic_confidence):
            reasons.append('semantic_confidence_too_low')
    blur_score = _optional_float(metadata, 'blur_score')
    if blur_score is not None and blur_score < config.min_blur_score:
        reasons.append('image_too_blurry')
    brightness = _optional_float(metadata, 'brightness_mean')
    if brightness is not None and brightness > config.max_brightness_mean:
        reasons.append('image_overexposed')
    return len(reasons) == 0, reasons


def apply_image_filter(candidate_labels, candidate_weights, accepted):
    if accepted:
        return candidate_labels.copy(), candidate_weights.copy()
    return (
        np.full(candidate_labels.shape, IGNORE, dtype=np.uint8),
        np.zeros(candidate_weights.shape, dtype=np.float32))


def colorize_labels(labels):
    color = np.full((*labels.shape, 3), (128, 128, 128), dtype=np.uint8)
    color[labels == BACKGROUND] = (0, 0, 0)
    color[labels == HEALTHY] = (0, 180, 0)
    color[labels == LESION] = (30, 30, 220)
    return color


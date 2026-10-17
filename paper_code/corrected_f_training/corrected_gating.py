"""Corrected three-class pseudo-label gating for the F experiment."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


BACKGROUND = 0
HEALTHY = 1
LESION = 2
IGNORE = 255


@dataclass
class CorrectedGatingConfig:
    lesion_teacher_threshold: float = 0.80
    lesion_semantic_threshold: float = 0.10
    healthy_teacher_threshold: float = 0.80
    lesion_negative_threshold: float = 0.10
    background_teacher_threshold: float = 0.80
    background_leaf_threshold: float = 0.10
    leaf_threshold: float = 0.50
    erosion_radius: int = 5
    semantic_weight_floor: float = 0.75
    min_leaf_inner_area_percent: float = 5.0
    min_positive_pixels: int = 16
    min_semantic_confidence: float = 0.60

    def validate(self):
        for name, value in asdict(self).items():
            if name in {'erosion_radius', 'min_positive_pixels'}:
                continue
            if name == 'min_leaf_inner_area_percent':
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {value}')
        if self.erosion_radius < 0 or self.min_positive_pixels < 0:
            raise ValueError('Radius and minimum pixel count must be nonnegative')


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    config = CorrectedGatingConfig(**raw['gating'])
    config.validate()
    return config


def save_config(path, config, selection):
    Path(path).write_text(
        json.dumps(
            {'gating': asdict(config), 'selection': selection},
            ensure_ascii=False, indent=2),
        encoding='utf-8')


def leaf_interior(p_leaf, config):
    mask = (p_leaf >= config.leaf_threshold).astype(np.uint8)
    if config.erosion_radius == 0:
        return mask.astype(bool)
    width = 2 * config.erosion_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.erode(mask, kernel).astype(bool)


def gate_pixels(p_teacher, p_leaf, a_sem, config):
    """Require teacher/leaf agreement; use CAM conservatively for lesions."""
    if p_teacher.shape[0] != 3:
        raise ValueError(f'Expected p_teacher [3,H,W], got {p_teacher.shape}')
    if p_teacher.shape[1:] != p_leaf.shape or p_leaf.shape != a_sem.shape:
        raise ValueError('Teacher, leaf, and semantic maps are not aligned')
    p_background, p_healthy, p_lesion = p_teacher
    inner = leaf_interior(p_leaf, config)
    lesion = (
        inner
        & (p_lesion >= config.lesion_teacher_threshold)
        & (a_sem >= config.lesion_semantic_threshold))
    healthy = (
        inner
        & (p_healthy >= config.healthy_teacher_threshold)
        & (p_lesion <= config.lesion_negative_threshold))
    background = (
        (p_leaf <= config.background_leaf_threshold)
        & (p_background >= config.background_teacher_threshold))

    labels = np.full(p_leaf.shape, IGNORE, dtype=np.uint8)
    labels[background] = BACKGROUND
    labels[healthy] = HEALTHY
    labels[lesion] = LESION

    weights = np.zeros(p_leaf.shape, dtype=np.float32)
    weights[background] = (
        p_background[background] * (1.0 - p_leaf[background]))
    weights[healthy] = p_healthy[healthy] * p_leaf[healthy]
    semantic_factor = (
        config.semantic_weight_floor
        + (1.0 - config.semantic_weight_floor) * a_sem[lesion])
    weights[lesion] = p_lesion[lesion] * semantic_factor
    return labels, np.clip(weights, 0.0, 1.0), {
        'background': background,
        'healthy': healthy,
        'lesion': lesion,
        'inner': inner,
    }


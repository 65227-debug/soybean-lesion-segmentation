"""Dataset and paper-friendly metrics for the soybean lesion task."""

from collections import OrderedDict

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.datasets import BaseSegDataset
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import DATASETS, METRICS, TRANSFORMS


@DATASETS.register_module()
class DadouLesionDataset(BaseSegDataset):
    """Three-class soybean leaf lesion dataset.

    Label IDs are intentionally fixed because they are also consumed by the
    downstream dual-evidence gating pipeline.
    """

    METAINFO = dict(
        classes=('background', 'healthy_leaf', 'lesion'),
        palette=[[0, 0, 0], [0, 180, 0], [220, 30, 30]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class DadouLeafDataset(BaseSegDataset):
    """Independent binary leaf/background dataset for the U-Net gate."""

    METAINFO = dict(
        classes=('background', 'leaf'),
        palette=[[0, 0, 0], [0, 180, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@TRANSFORMS.register_module()
class MergeLeafAndLesion(BaseTransform):
    """Map healthy tissue (1) and lesion (2) to the binary leaf class (1)."""

    def transform(self, results: dict) -> dict:
        for key in results.get('seg_fields', []):
            source = results[key]
            binary = np.zeros_like(source, dtype=np.uint8)
            binary[(source == 1) | (source == 2)] = 1
            binary[source == 255] = 255
            results[key] = binary
        return results


@METRICS.register_module()
class DadouIoUMetric(IoUMetric):
    """IoU metric that also returns per-class values to the JSON/CSV logs."""

    def compute_metrics(self, results: list) -> dict:
        summary = super().compute_metrics(results)

        transposed = tuple(zip(*results))
        total_area_intersect = sum(transposed[0])
        total_area_union = sum(transposed[1])
        total_area_pred_label = sum(transposed[2])
        total_area_label = sum(transposed[3])
        detailed = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta)

        output = OrderedDict(summary)
        class_names = self.dataset_meta['classes']
        for metric_name, values in detailed.items():
            if metric_name == 'aAcc':
                continue
            for class_name, value in zip(class_names, values):
                scalar = float(value)
                output[f'{metric_name}_{class_name}'] = (
                    round(scalar * 100, 4) if np.isfinite(scalar) else float('nan'))

        for class_name, pixels in zip(class_names, total_area_label):
            output[f'Pixels_{class_name}'] = int(pixels.item())
        return output

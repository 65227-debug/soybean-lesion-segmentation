#!/usr/bin/env python
"""Supplementary severity, boundary, and lesion-size analysis for A--G.

This script only reads the frozen internal-test masks and previously exported
predictions.  It never loads images from the training or validation splits and
does not modify any model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


SEED = 20260826
BOOTSTRAP_REPEATS = 10000
BOUNDARY_TOLERANCE_PX = 2
BOUNDARY_BAND_WIDTH_PX = 3
COMPONENT_OVERLAP_THRESHOLD = 0.10
SIZE_BINS = {
    "small": (0, 64),
    "medium": (64, 256),
    "large": (256, math.inf),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="paper_materials/一区提升实验补充_20260826/01_严重度与小病斑边界",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unreadable mask: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel: {path}, shape={mask.shape}")
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape mismatch: {path}, got={mask.shape}, expected={expected_shape}"
        )
    labels = set(np.unique(mask).tolist())
    if not labels.issubset({0, 1, 2, 255}):
        raise ValueError(f"Unexpected labels in {path}: {sorted(labels)}")
    return mask.astype(np.uint8)


def severity_percent(mask: np.ndarray) -> float:
    leaf_pixels = int(np.count_nonzero((mask == 1) | (mask == 2)))
    lesion_pixels = int(np.count_nonzero(mask == 2))
    return 100.0 * lesion_pixels / leaf_pixels if leaf_pixels else 0.0


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mean_x, mean_y = float(x.mean()), float(y.mean())
    var_x = float(np.mean((x - mean_x) ** 2))
    var_y = float(np.mean((y - mean_y) ** 2))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def severity_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    difference = prediction - truth
    rho = spearmanr(truth, prediction).statistic
    return {
        "MAE_percentage_points": float(np.mean(np.abs(difference))),
        "RMSE_percentage_points": float(np.sqrt(np.mean(difference**2))),
        "R2": float(r2_score(truth, prediction)),
        "Spearman_rho": float(rho),
        "CCC": float(concordance_correlation(truth, prediction)),
        "mean_bias_percentage_points": float(np.mean(difference)),
        "loa_lower_percentage_points": float(np.mean(difference) - 1.96 * np.std(difference, ddof=1)),
        "loa_upper_percentage_points": float(np.mean(difference) + 1.96 * np.std(difference, ddof=1)),
    }


def one_pixel_boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return mask & ~binary_erosion(mask, structure=structure, border_value=0)


def boundary_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = one_pixel_boundary(prediction)
    truth_boundary = one_pixel_boundary(truth)
    if not truth_boundary.any():
        return float("nan"), float("nan")

    dilated_truth = binary_dilation(
        truth_boundary, structure=structure, iterations=BOUNDARY_TOLERANCE_PX
    )
    dilated_prediction = binary_dilation(
        pred_boundary, structure=structure, iterations=BOUNDARY_TOLERANCE_PX
    )
    precision = (
        float(np.count_nonzero(pred_boundary & dilated_truth))
        / float(np.count_nonzero(pred_boundary))
        if pred_boundary.any()
        else 0.0
    )
    recall = float(np.count_nonzero(truth_boundary & dilated_prediction)) / float(
        np.count_nonzero(truth_boundary)
    )
    boundary_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    pred_band = prediction & ~binary_erosion(
        prediction,
        structure=structure,
        iterations=BOUNDARY_BAND_WIDTH_PX,
        border_value=0,
    )
    truth_band = truth & ~binary_erosion(
        truth,
        structure=structure,
        iterations=BOUNDARY_BAND_WIDTH_PX,
        border_value=0,
    )
    union = int(np.count_nonzero(pred_band | truth_band))
    boundary_iou = (
        float(np.count_nonzero(pred_band & truth_band)) / union if union else 0.0
    )
    return boundary_f1, boundary_iou


def component_counts(prediction: np.ndarray, truth: np.ndarray) -> dict[str, tuple[int, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        truth.astype(np.uint8), connectivity=8
    )
    result = {name: [0, 0] for name in SIZE_BINS}
    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        name = next(
            key for key, (lower, upper) in SIZE_BINS.items() if lower <= area < upper
        )
        component = labels == component_index
        overlap = int(np.count_nonzero(prediction & component)) / area
        result[name][1] += 1
        result[name][0] += int(overlap >= COMPONENT_OVERLAP_THRESHOLD)
    return {name: tuple(values) for name, values in result.items()}


def percentile_interval(values: list[float]) -> tuple[float, float]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return tuple(np.percentile(valid, [2.5, 97.5]).tolist())


def bootstrap_severity(
    truth: np.ndarray, prediction: np.ndarray, repeats: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        indices = rng.integers(0, truth.size, truth.size)
        for key, value in severity_metrics(truth[indices], prediction[indices]).items():
            samples.setdefault(key, []).append(value)
    return {key: percentile_interval(values) for key, values in samples.items()}


def bootstrap_boundary(
    per_image: list[dict], repeats: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    samples = {"Boundary_F1": [], "Boundary_IoU": []}
    for name in SIZE_BINS:
        samples[f"{name}_component_recall"] = []
    n = len(per_image)
    for _ in range(repeats):
        indices = rng.integers(0, n, n)
        selected = [per_image[index] for index in indices]
        for metric in ("Boundary_F1", "Boundary_IoU"):
            values = [row[metric] for row in selected if np.isfinite(row[metric])]
            samples[metric].append(float(np.mean(values)))
        for name in SIZE_BINS:
            detected = sum(row[f"{name}_detected"] for row in selected)
            total = sum(row[f"{name}_total"] for row in selected)
            samples[f"{name}_component_recall"].append(detected / total if total else np.nan)
    return {key: percentile_interval(values) for key, values in samples.items()}


def paired_bootstrap(
    rows_by_model: dict[str, list[dict]], repeats: int
) -> list[dict]:
    rng = np.random.default_rng(SEED + 300)
    comparisons = []
    n = len(rows_by_model["G"])
    for baseline in "ABCDEF":
        metrics = [
            ("severity_absolute_error", "lower"),
            ("Boundary_F1", "higher"),
            ("Boundary_IoU", "higher"),
        ]
        for size in SIZE_BINS:
            metrics.append((f"{size}_component_recall", "higher"))
        for metric, direction in metrics:
            differences = []
            for _ in range(repeats):
                indices = rng.integers(0, n, n)
                g_rows = [rows_by_model["G"][index] for index in indices]
                b_rows = [rows_by_model[baseline][index] for index in indices]
                if metric.endswith("component_recall"):
                    size = metric.split("_")[0]
                    g_total = sum(row[f"{size}_total"] for row in g_rows)
                    b_total = sum(row[f"{size}_total"] for row in b_rows)
                    if not g_total or not b_total:
                        differences.append(np.nan)
                    else:
                        g_value = sum(row[f"{size}_detected"] for row in g_rows) / g_total
                        b_value = sum(row[f"{size}_detected"] for row in b_rows) / b_total
                        differences.append(g_value - b_value)
                else:
                    pairs = [
                        (g[metric], b[metric])
                        for g, b in zip(g_rows, b_rows)
                        if np.isfinite(g[metric]) and np.isfinite(b[metric])
                    ]
                    differences.append(
                        float(np.mean([g - b for g, b in pairs])) if pairs else np.nan
                    )
            low, high = percentile_interval(differences)
            if metric == "severity_absolute_error":
                point = float(
                    np.mean(
                        [
                            b[metric] - g[metric]
                            for g, b in zip(rows_by_model["G"], rows_by_model[baseline])
                        ]
                    )
                )
                low, high = -high, -low
                interpretation = "positive means G has lower absolute severity error"
            elif metric.endswith("component_recall"):
                size = metric.split("_")[0]
                g_detected = sum(row[f"{size}_detected"] for row in rows_by_model["G"])
                g_total = sum(row[f"{size}_total"] for row in rows_by_model["G"])
                b_detected = sum(row[f"{size}_detected"] for row in rows_by_model[baseline])
                b_total = sum(row[f"{size}_total"] for row in rows_by_model[baseline])
                point = g_detected / g_total - b_detected / b_total
                interpretation = "positive means G is higher"
            else:
                pairs = [
                    (g[metric], b[metric])
                    for g, b in zip(rows_by_model["G"], rows_by_model[baseline])
                    if np.isfinite(g[metric]) and np.isfinite(b[metric])
                ]
                point = float(np.mean([g - b for g, b in pairs]))
                interpretation = "positive means G is higher"
            comparisons.append(
                {
                    "comparison": f"G-{baseline}",
                    "metric": metric,
                    "direction": direction,
                    "difference": point,
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "interpretation": interpretation,
                }
            )
    return comparisons


def plot_severity(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    severity_summary: list[dict],
    output_dir: Path,
) -> None:
    g = predictions["G"]
    limit = max(float(truth.max()), float(g.max())) * 1.06
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=220)
    axes[0].scatter(truth, g, s=27, alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[0].plot([0, limit], [0, limit], "--", color="black", linewidth=1)
    axes[0].set(
        xlabel="Reference severity (%)",
        ylabel="Predicted severity (%)",
        title="G: severity agreement",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    difference = g - truth
    mean_pair = (g + truth) / 2.0
    bias = float(difference.mean())
    lower = bias - 1.96 * float(difference.std(ddof=1))
    upper = bias + 1.96 * float(difference.std(ddof=1))
    axes[1].scatter(mean_pair, difference, s=27, alpha=0.82, edgecolor="white", linewidth=0.4)
    for value, style, label in (
        (bias, "-", f"Bias {bias:.2f} pp"),
        (lower, "--", f"Lower LoA {lower:.2f}"),
        (upper, "--", f"Upper LoA {upper:.2f}"),
    ):
        axes[1].axhline(value, color="black", linestyle=style, linewidth=1, label=label)
    axes[1].set(
        xlabel="Mean of reference and prediction (%)",
        ylabel="Prediction - reference (percentage points)",
        title="G: Bland-Altman analysis",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_dir / "图_最终G严重度一致性与Bland_Altman.png", bbox_inches="tight")
    figure.savefig(output_dir / "图_最终G严重度一致性与Bland_Altman.pdf", bbox_inches="tight")
    plt.close(figure)

    labels = [row["model"] for row in severity_summary]
    mae = [row["MAE_percentage_points"] for row in severity_summary]
    rmse = [row["RMSE_percentage_points"] for row in severity_summary]
    r2 = [row["R2"] for row in severity_summary]
    ccc = [row["CCC"] for row in severity_summary]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), dpi=220)
    width = 0.36
    axes[0].bar(x - width / 2, mae, width, label="MAE")
    axes[0].bar(x + width / 2, rmse, width, label="RMSE")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Percentage points", title="Severity error (lower is better)")
    axes[1].bar(x - width / 2, r2, width, label=r"$R^2$")
    axes[1].bar(x + width / 2, ccc, width, label="CCC")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Score", title="Severity agreement (higher is better)", ylim=(0, 1.0))
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "图_A至G严重度定量比较.png", bbox_inches="tight")
    figure.savefig(output_dir / "图_A至G严重度定量比较.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_boundary(summary: list[dict], output_dir: Path) -> None:
    labels = [row["model"] for row in summary]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=220)
    width = 0.36
    axes[0].bar(x - width / 2, [100 * row["Boundary_F1"] for row in summary], width, label="Boundary F1")
    axes[0].bar(x + width / 2, [100 * row["Boundary_IoU"] for row in summary], width, label="Boundary IoU")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Score (%)", title="Lesion boundary quality", ylim=(0, 100))
    size_width = 0.24
    for offset, name in zip((-size_width, 0.0, size_width), SIZE_BINS):
        axes[1].bar(
            x + offset,
            [100 * row[f"{name}_component_recall"] for row in summary],
            size_width,
            label=name.capitalize(),
        )
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Component recall (%)", title="Recall by lesion-component size", ylim=(0, 100))
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "图_A至G边界与分尺度病斑召回.png", bbox_inches="tight")
    figure.savefig(output_dir / "图_A至G边界与分尺度病斑召回.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(root / "data/dadou_student/supervised_test_manifest.csv")
    if len(manifest) != 36:
        raise ValueError(f"Expected 36 frozen test images, found {len(manifest)}")
    prediction_dirs = {
        **{
            name: root / f"work_dirs/dadou_ablation_comparison/predictions/{name}/masks"
            for name in "ABCDEF"
        },
        "G": root / "work_dirs/dadou_corrected_F_final_test/predictions/masks",
    }

    truth_severity: list[float] = []
    model_severity: dict[str, list[float]] = {name: [] for name in prediction_dirs}
    rows_by_model: dict[str, list[dict]] = {name: [] for name in prediction_dirs}
    per_image_wide: list[dict] = []
    expected_shape: tuple[int, int] | None = None
    for manifest_row in manifest:
        sample_id = manifest_row["sample_id"]
        truth_mask = load_mask(Path(manifest_row["mask_path"]), expected_shape)
        expected_shape = truth_mask.shape
        if expected_shape != (512, 512):
            raise ValueError(f"Expected 512x512 masks, found {expected_shape}")
        truth_lesion = truth_mask == 2
        truth_value = severity_percent(truth_mask)
        truth_severity.append(truth_value)
        wide_row = {
            "sample_id": sample_id,
            "class_name": sample_id.split("_")[-1],
            "reference_severity_percent": truth_value,
            "reference_lesion_pixels": int(np.count_nonzero(truth_lesion)),
            "reference_leaf_pixels": int(np.count_nonzero((truth_mask == 1) | truth_lesion)),
        }
        for model_name, prediction_dir in prediction_dirs.items():
            prediction_mask = load_mask(prediction_dir / f"{sample_id}.png", expected_shape)
            prediction_lesion = prediction_mask == 2
            prediction_value = severity_percent(prediction_mask)
            model_severity[model_name].append(prediction_value)
            boundary_f1, boundary_iou = boundary_metrics(prediction_lesion, truth_lesion)
            counts = component_counts(prediction_lesion, truth_lesion)
            model_row = {
                "sample_id": sample_id,
                "truth_severity": truth_value,
                "predicted_severity": prediction_value,
                "severity_absolute_error": abs(prediction_value - truth_value),
                "Boundary_F1": boundary_f1,
                "Boundary_IoU": boundary_iou,
            }
            for size_name, (detected, total) in counts.items():
                model_row[f"{size_name}_detected"] = detected
                model_row[f"{size_name}_total"] = total
                model_row[f"{size_name}_component_recall"] = detected / total if total else np.nan
            rows_by_model[model_name].append(model_row)
            wide_row[f"{model_name}_predicted_severity_percent"] = prediction_value
            wide_row[f"{model_name}_severity_absolute_error_pp"] = abs(prediction_value - truth_value)
            wide_row[f"{model_name}_Boundary_F1"] = boundary_f1
            wide_row[f"{model_name}_Boundary_IoU"] = boundary_iou
            for size_name, (detected, total) in counts.items():
                wide_row[f"{model_name}_{size_name}_detected"] = detected
                wide_row[f"{model_name}_{size_name}_total"] = total
        per_image_wide.append(wide_row)

    truth_array = np.asarray(truth_severity, dtype=np.float64)
    severity_summary = []
    severity_ci_rows = []
    boundary_summary = []
    boundary_ci_rows = []
    for model_index, model_name in enumerate(prediction_dirs):
        prediction_array = np.asarray(model_severity[model_name], dtype=np.float64)
        metrics = severity_metrics(truth_array, prediction_array)
        severity_summary.append({"model": model_name, "n_images": len(manifest), **metrics})
        intervals = bootstrap_severity(
            truth_array,
            prediction_array,
            args.bootstrap_repeats,
            np.random.default_rng(SEED + model_index),
        )
        for metric, (lower, upper) in intervals.items():
            severity_ci_rows.append(
                {"model": model_name, "metric": metric, "ci95_lower": lower, "ci95_upper": upper}
            )

        model_rows = rows_by_model[model_name]
        boundary_values = [row["Boundary_F1"] for row in model_rows if np.isfinite(row["Boundary_F1"])]
        iou_values = [row["Boundary_IoU"] for row in model_rows if np.isfinite(row["Boundary_IoU"])]
        boundary_row = {
            "model": model_name,
            "n_lesion_positive_images": len(boundary_values),
            "Boundary_F1": float(np.mean(boundary_values)),
            "Boundary_IoU": float(np.mean(iou_values)),
        }
        for size_name in SIZE_BINS:
            detected = sum(row[f"{size_name}_detected"] for row in model_rows)
            total = sum(row[f"{size_name}_total"] for row in model_rows)
            boundary_row[f"{size_name}_detected"] = detected
            boundary_row[f"{size_name}_total"] = total
            boundary_row[f"{size_name}_component_recall"] = detected / total if total else np.nan
        boundary_summary.append(boundary_row)
        intervals = bootstrap_boundary(
            model_rows,
            args.bootstrap_repeats,
            np.random.default_rng(SEED + 100 + model_index),
        )
        for metric, (lower, upper) in intervals.items():
            boundary_ci_rows.append(
                {"model": model_name, "metric": metric, "ci95_lower": lower, "ci95_upper": upper}
            )

    paired_rows = paired_bootstrap(rows_by_model, args.bootstrap_repeats)
    write_csv(output_dir / "表_逐图严重度与边界原始数据.csv", per_image_wide)
    write_csv(output_dir / "表_A至G严重度指标.csv", severity_summary)
    write_csv(output_dir / "表_A至G严重度指标Bootstrap95CI.csv", severity_ci_rows)
    write_csv(output_dir / "表_A至G边界与分尺度病斑召回.csv", boundary_summary)
    write_csv(output_dir / "表_A至G边界指标Bootstrap95CI.csv", boundary_ci_rows)
    write_csv(output_dir / "表_G对各基线配对Bootstrap_严重度边界小病斑.csv", paired_rows)
    plot_severity(
        truth_array,
        {name: np.asarray(values) for name, values in model_severity.items()},
        severity_summary,
        output_dir,
    )
    plot_boundary(boundary_summary, output_dir)

    summary = {
        "protocol": {
            "split": "frozen internal test only",
            "test_images": len(manifest),
            "input_size": list(expected_shape),
            "models": list(prediction_dirs),
            "model_training_or_selection_performed": False,
            "severity_formula": "100 * lesion_pixels / (healthy_pixels + lesion_pixels)",
            "boundary_f1_tolerance_pixels": BOUNDARY_TOLERANCE_PX,
            "boundary_iou_inner_band_width_pixels": BOUNDARY_BAND_WIDTH_PX,
            "component_connectivity": 8,
            "component_detection_rule": f"predicted lesion overlap / component area >= {COMPONENT_OVERLAP_THRESHOLD}",
            "component_size_bins_pixels": {
                key: [lower, None if math.isinf(upper) else upper]
                for key, (lower, upper) in SIZE_BINS.items()
            },
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": SEED,
            "bootstrap_unit": "image (cluster bootstrap for component metrics)",
        },
        "severity": severity_summary,
        "boundary_and_size": boundary_summary,
        "notes": [
            "Severity is derived from the same pixel masks; it is not an independently hand-scored field severity measurement.",
            "Boundary metrics exclude images without reference lesion pixels.",
            "Component-size analysis is supplementary and should be reported together with its fixed pixel thresholds.",
        ],
    }
    (output_dir / "实验汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

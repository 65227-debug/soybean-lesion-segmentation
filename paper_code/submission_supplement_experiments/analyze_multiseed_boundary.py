#!/usr/bin/env python3
"""Read-only five-seed Boundary F1 and lesion-component recall analysis."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_code/q1_upgrade_experiments"))
from analyze_severity_boundary import (  # noqa: E402
    SIZE_BINS,
    boundary_metrics,
    component_counts,
    load_mask,
)

SEEDS = (20260803, 20260804, 20260805, 20260806, 20260807)
OUT = ROOT / "paper_materials/投稿前补充实验_20260829"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prediction_dir(seed, method):
    if seed == 20260803:
        if method == "G":
            return ROOT / "work_dirs/dadou_corrected_F_final_test/predictions/masks"
        return ROOT / "work_dirs/dadou_stpp_aligned_seed20260803/test_predictions/masks"
    base = ROOT / "work_dirs" / f"submission_seed_{seed}"
    if method == "G":
        return base / "G_final_test/predictions/masks"
    return base / "STPP/test_predictions/masks"


def paired_ci(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    half = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(ROOT / "data/dadou_student/supervised_test_manifest.csv")
    if len(manifest) != 37:
        raise RuntimeError(f"Expected 37 test images, found {len(manifest)}")
    rows = []
    for seed in SEEDS:
        for method in ("G", "ST++"):
            b_f1, b_iou = [], []
            totals = {name: [0, 0] for name in SIZE_BINS}
            pred_dir = prediction_dir(seed, method)
            for item in manifest:
                truth = load_mask(Path(item["mask_path"]))
                pred = load_mask(pred_dir / f"{item['sample_id']}.png", truth.shape)
                f1, iou = boundary_metrics(pred == 2, truth == 2)
                if np.isfinite(f1):
                    b_f1.append(f1); b_iou.append(iou)
                for name, (detected, total) in component_counts(pred == 2, truth == 2).items():
                    totals[name][0] += detected; totals[name][1] += total
            row = {
                "seed": seed, "method": method, "test_images": len(manifest),
                "lesion_positive_images": len(b_f1),
                "Boundary_F1_percent": 100 * float(np.mean(b_f1)),
                "Boundary_IoU_percent": 100 * float(np.mean(b_iou)),
            }
            for name, (detected, total) in totals.items():
                row[f"{name}_detected"] = detected
                row[f"{name}_total"] = total
                row[f"{name}_component_recall_percent"] = 100 * detected / total
            rows.append(row)
    write_csv(OUT / "表5_G与STPP五种子边界及分尺度召回.csv", rows)

    metrics = ["Boundary_F1_percent", "Boundary_IoU_percent"] + [
        f"{name}_component_recall_percent" for name in SIZE_BINS
    ]
    summary_rows, diff_rows = [], []
    for method in ("G", "ST++"):
        for metric in metrics:
            values = [r[metric] for r in rows if r["method"] == method]
            summary_rows.append({
                "method": method, "metric": metric, "n_seeds": 5,
                "mean_percent": float(np.mean(values)),
                "sd_percent": float(np.std(values, ddof=1)),
            })
    for metric in metrics:
        g = [r[metric] for r in rows if r["method"] == "G"]
        st = [r[metric] for r in rows if r["method"] == "ST++"]
        mean, low, high = paired_ci(np.asarray(g) - np.asarray(st))
        diff_rows.append({
            "comparison": "G - ST++", "metric": metric, "n_paired_seeds": 5,
            "mean_difference_pp": mean, "ci95_low_pp": low,
            "ci95_high_pp": high, "ci_excludes_zero": bool(low > 0 or high < 0),
        })
    write_csv(OUT / "表6_G与STPP边界分尺度均值标准差.csv", summary_rows)
    write_csv(OUT / "表7_G与STPP边界分尺度配对95CI.csv", diff_rows)

    labels = ["Boundary F1", "Small recall", "Medium recall", "Large recall"]
    plot_metrics = ["Boundary_F1_percent", "small_component_recall_percent",
                    "medium_component_recall_percent", "large_component_recall_percent"]
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    for offset, method, color in ((-width / 2, "G", "#c62828"),
                                  (width / 2, "ST++", "#1565c0")):
        means, sds = [], []
        for metric in plot_metrics:
            values = [r[metric] for r in rows if r["method"] == method]
            means.append(np.mean(values)); sds.append(np.std(values, ddof=1))
        ax.bar(x + offset, means, width, yerr=sds, capsize=3, label=method, color=color)
    ax.set_xticks(x, labels); ax.set_ylabel("Score / recall (%)")
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    fig.savefig(OUT / "图3_G与STPP五种子边界及分尺度召回.png", dpi=400)
    fig.savefig(OUT / "图3_G与STPP五种子边界及分尺度召回.pdf")
    plt.close(fig)
    (OUT / "边界分尺度分析配置.json").write_text(json.dumps({
        "boundary_tolerance_pixels": 2,
        "component_connectivity": 8,
        "component_detection_rule": "predicted lesion overlap / component area >= 0.10",
        "component_size_bins_pixels": {
            k: [v[0], None if math.isinf(v[1]) else v[1]] for k, v in SIZE_BINS.items()
        },
        "model_training_or_selection_performed": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

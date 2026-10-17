#!/usr/bin/env python3
"""Aggregate the fixed-seed, scene-stratified labeled-data fraction study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_materials/投稿前补充实验_20260829"
SEED = 20260803
FRACTIONS = (25, 50, 75, 100)
METRICS = ("mIoU", "mDice", "pixel_accuracy", "IoU_lesion", "Dice_lesion",
           "Precision_lesion", "Recall_lesion")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(path):
    return load(path)["metrics"]


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split_summary = load(ROOT / "data/dadou_label_fraction_subsets/fraction_generation_summary.json")
    rows = []
    audit = []
    for fraction in FRACTIONS:
        tag = f"{fraction:03d}"
        base = ROOT / "work_dirs" / f"label_fraction_{tag}_seed{SEED}"
        a_path = base / "A_test/test_summary.json"
        g_path = (ROOT / "work_dirs/dadou_corrected_F_final_test/test_summary.json"
                  if fraction == 100 else base / "G_final_test/test_summary.json")
        if not a_path.exists() or not g_path.exists():
            raise FileNotFoundError(f"Incomplete fraction {fraction}: {a_path}, {g_path}")
        for method, path in (("Supervised A", a_path), ("Final G", g_path)):
            value = metrics(path)
            row = {
                "requested_label_fraction_percent": fraction,
                "actual_labeled_images": split_summary["fraction_summary"][str(fraction)]["images"],
                "actual_labeled_scenes": split_summary["fraction_summary"][str(fraction)]["scenes"],
                "seed": SEED, "method": method,
            }
            row.update({metric: value[metric] * 100 for metric in METRICS})
            rows.append(row)
        if fraction < 100:
            expected = {"A": 60, "G0": 60, "G1_distilled": 18}
            files = {"A": "epoch_metrics.csv", "G0": "epoch_metrics.csv",
                     "G1_distilled": "training_history.csv"}
            for stage, expected_rows in expected.items():
                with (base / stage / files[stage]).open(encoding="utf-8-sig", newline="") as handle:
                    actual_rows = sum(1 for _ in csv.DictReader(handle))
                audit.append({
                    "fraction_percent": fraction, "stage": stage,
                    "expected_epoch_rows": expected_rows, "actual_epoch_rows": actual_rows,
                    "pass": actual_rows == expected_rows,
                })
    write_csv(OUT / "表8_标签比例逐次冻结测试结果.csv", rows)

    gains = []
    for fraction in FRACTIONS:
        a = next(r for r in rows if r["requested_label_fraction_percent"] == fraction and r["method"] == "Supervised A")
        g = next(r for r in rows if r["requested_label_fraction_percent"] == fraction and r["method"] == "Final G")
        gains.append({
            "requested_label_fraction_percent": fraction,
            "actual_labeled_images": a["actual_labeled_images"],
            "actual_labeled_scenes": a["actual_labeled_scenes"],
            "G_minus_A_mIoU_pp": g["mIoU"] - a["mIoU"],
            "G_minus_A_lesion_IoU_pp": g["IoU_lesion"] - a["IoU_lesion"],
        })
    write_csv(OUT / "表9_标签比例G相对纯监督增益.csv", gains)
    write_csv(OUT / "表10_标签比例训练完整性审计.csv", audit)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("mIoU", "IoU_lesion"), ("mIoU (%)", "Lesion IoU (%)")):
        for method, color, marker in (("Supervised A", "#455a64", "o"),
                                      ("Final G", "#c62828", "s")):
            vals = [next(r for r in rows if r["requested_label_fraction_percent"] == f and r["method"] == method)[metric] for f in FRACTIONS]
            ax.plot(FRACTIONS, vals, marker=marker, color=color, lw=2, label=method)
        ax.set_xticks(FRACTIONS); ax.set_xlabel("Requested labeled fraction (%)")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "图4_场景分层标签比例曲线.png", dpi=400)
    fig.savefig(OUT / "图4_场景分层标签比例曲线.pdf")
    plt.close(fig)

    summary = {
        "protocol": "nested, scene_cluster_id-indivisible, disease-stratified",
        "seed": SEED,
        "requested_fractions": list(FRACTIONS),
        "actual_labeled_images": [split_summary["fraction_summary"][str(f)]["images"] for f in FRACTIONS],
        "actual_labeled_scenes": [split_summary["fraction_summary"][str(f)]["scenes"] for f in FRACTIONS],
        "validation_images_fixed": 36,
        "test_images_fixed": 36,
        "weak_input_pool_fixed": 1390,
        "weak_retained_pool_fixed": 1354,
        "single_seed_exploratory_curve": True,
        "note": (
            "The 25% subset contains 26 rather than 27 images because healthy-leaf "
            "scene clusters were kept indivisible while retaining nested class stratification."
        ),
        "training_audit_pass": all(r["pass"] for r in audit),
        "results": rows,
        "gains": gains,
    }
    (OUT / "标签比例实验汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot validation-only pseudo-label precision/recall/coverage trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="paper_materials/一区提升实验补充_20260826/02_伪标签质量曲线",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    source = root / "data/dadou/06_验证集门控调参_修正版"
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / "selection_summary.json").read_text(encoding="utf-8"))
    if selection.get("split") != "validation only" or selection.get("test_split_used") is not False:
        raise ValueError("The gating grid is not marked validation-only; aborting.")
    selected = selection["selected_lesion"]
    rows = read_csv(source / "lesion_grid.csv")
    for row in rows:
        for key in row:
            row[key] = float(row[key])
        row["coverage_percent_of_all_validation_pixels"] = (
            100.0 * row["accepted_pixels"] / (selection["validation_images"] * 512 * 512)
        )

    teacher_values = sorted({row["lesion_teacher_threshold"] for row in rows})
    semantic_values = sorted({row["lesion_semantic_threshold"] for row in rows})
    maps = {metric: np.full((len(teacher_values), len(semantic_values)), np.nan) for metric in ("precision", "recall", "f1")}
    for row in rows:
        i = teacher_values.index(row["lesion_teacher_threshold"])
        j = semantic_values.index(row["lesion_semantic_threshold"])
        for metric in maps:
            maps[metric][i, j] = row[metric]

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), dpi=220)
    semantic_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(semantic_values)))
    for color, threshold in zip(semantic_colors, semantic_values):
        subset = sorted(
            [row for row in rows if row["lesion_semantic_threshold"] == threshold],
            key=lambda row: row["lesion_teacher_threshold"],
        )
        axes[0].plot(
            [100 * row["recall"] for row in subset],
            [100 * row["precision"] for row in subset],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=fr"$\tau_s$={threshold:.2f}",
            color=color,
        )
        axes[1].plot(
            [row["coverage_percent_of_all_validation_pixels"] for row in subset],
            [100 * row["f1"] for row in subset],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=fr"$\tau_s$={threshold:.2f}",
            color=color,
        )
    selected_row = next(
        row
        for row in rows
        if row["lesion_teacher_threshold"] == selected["lesion_teacher_threshold"]
        and row["lesion_semantic_threshold"] == selected["lesion_semantic_threshold"]
    )
    axes[0].scatter(
        100 * selected_row["recall"],
        100 * selected_row["precision"],
        marker="*",
        s=170,
        color="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="Selected",
    )
    axes[1].scatter(
        selected_row["coverage_percent_of_all_validation_pixels"],
        100 * selected_row["f1"],
        marker="*",
        s=170,
        color="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="Selected",
    )
    axes[0].set(xlabel="Lesion recall (%)", ylabel="Lesion precision (%)", title="Validation precision-recall trade-off")
    axes[1].set(xlabel="Accepted lesion pixels / all validation pixels (%)", ylabel="Lesion F1 (%)", title="Coverage-quality trade-off")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "图_病斑伪标签覆盖率_精度_召回权衡.png", bbox_inches="tight")
    figure.savefig(output / "图_病斑伪标签覆盖率_精度_召回权衡.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), dpi=220)
    for axis, metric in zip(axes, ("precision", "recall", "f1")):
        image = axis.imshow(100 * maps[metric], cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        axis.set(
            xticks=np.arange(len(semantic_values)),
            xticklabels=[f"{value:.2f}" for value in semantic_values],
            yticks=np.arange(len(teacher_values)),
            yticklabels=[f"{value:.2f}" for value in teacher_values],
            xlabel=r"Semantic threshold $\tau_s$",
            ylabel=r"Teacher threshold $\tau_p$",
            title=metric.capitalize() + " (%)",
        )
        for i in range(len(teacher_values)):
            for j in range(len(semantic_values)):
                axis.text(j, i, f"{100 * maps[metric][i, j]:.1f}", ha="center", va="center", fontsize=6)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output / "图_病斑门控阈值敏感性热图.png", bbox_inches="tight")
    figure.savefig(output / "图_病斑门控阈值敏感性热图.pdf", bbox_inches="tight")
    plt.close(figure)

    for filename in ("lesion_grid.csv", "healthy_grid.csv", "background_grid.csv", "selection_summary.json"):
        shutil.copy2(source / filename, output / filename)
    summary = {
        "source": str(source),
        "selection_split": selection["split"],
        "validation_images": selection["validation_images"],
        "test_split_used": selection["test_split_used"],
        "selected_lesion": selected,
        "selected_coverage_percent_of_all_validation_pixels": selected_row["coverage_percent_of_all_validation_pixels"],
        "grid_points": len(rows),
        "note": "This is a visualization of the already frozen validation grid; no threshold was changed and no model was retrained.",
    }
    (output / "实验汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

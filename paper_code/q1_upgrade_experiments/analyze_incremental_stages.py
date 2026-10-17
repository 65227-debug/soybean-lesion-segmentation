#!/usr/bin/env python
"""Summarize the frozen F -> G0 -> G1 -> G2 incremental experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260826
BOOTSTRAP_REPEATS = 20000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="paper_materials/一区提升实验补充_20260826/04_门控蒸馏校准逐项消融",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage_sources = [
        {
            "stage": "F",
            "setting": "严格双证据硬门控",
            "summary": root / "work_dirs/dadou_ablation_comparison/comparison_summary.json",
            "summary_selector": ("groups", "F", "test"),
            "per_image": root / "work_dirs/dadou_ablation_comparison/test_per_image_F.csv",
        },
        {
            "stage": "G0",
            "setting": "连续三态软证据门控+场景均衡+监督预热",
            "summary": root / "work_dirs/dadou_corrected_F/training_summary.json",
            "summary_selector": ("test_metrics",),
            "per_image": root / "work_dirs/dadou_corrected_F/test_per_image_metrics.csv",
        },
        {
            "stage": "G1",
            "setting": "G0+软集成蒸馏",
            "summary": output / "G_门控加蒸馏_冻结测试/test_summary.json",
            "summary_selector": ("metrics",),
            "per_image": output / "G_门控加蒸馏_冻结测试/test_per_image.csv",
        },
        {
            "stage": "G2",
            "setting": "G1+仅验证集类别logit校准（最终G）",
            "summary": root / "work_dirs/dadou_corrected_F_final_test/test_summary.json",
            "summary_selector": ("metrics",),
            "per_image": root / "work_dirs/dadou_corrected_F_final_test/test_per_image.csv",
        },
    ]
    rows = []
    per_image = {}
    previous = None
    for source in stage_sources:
        payload = load_json(source["summary"])
        for key in source["summary_selector"]:
            payload = payload[key]
        row = {
            "stage": source["stage"],
            "setting": source["setting"],
            "mIoU_percent": 100 * payload["mIoU"],
            "lesion_IoU_percent": 100 * payload["IoU_lesion"],
            "lesion_Dice_percent": 100 * payload["Dice_lesion"],
            "lesion_Precision_percent": 100 * payload["Precision_lesion"],
            "lesion_Recall_percent": 100 * payload["Recall_lesion"],
            "delta_previous_mIoU_pp": "" if previous is None else 100 * (payload["mIoU"] - previous["mIoU"]),
            "delta_previous_lesion_IoU_pp": "" if previous is None else 100 * (payload["IoU_lesion"] - previous["IoU_lesion"]),
            "summary_source": str(source["summary"]),
            "per_image_source": str(source["per_image"]),
        }
        rows.append(row)
        previous = payload
        stage_rows = read_csv(source["per_image"])
        if len(stage_rows) != 36:
            raise ValueError(f"{source['stage']} expected 36 test rows, found {len(stage_rows)}")
        per_image[source["stage"]] = {row["sample_id"]: row for row in stage_rows}

    comparisons = []
    rng = np.random.default_rng(SEED)
    for first, second in zip(stage_sources[:-1], stage_sources[1:]):
        first_stage, second_stage = first["stage"], second["stage"]
        ids = sorted(set(per_image[first_stage]) & set(per_image[second_stage]))
        if len(ids) != 36:
            raise ValueError(f"Unaligned stage pair {first_stage}->{second_stage}: {len(ids)}")
        for metric in ("mIoU", "IoU_lesion"):
            valid_ids = [
                sample_id
                for sample_id in ids
                if per_image[first_stage][sample_id].get(metric, "") not in ("", "nan")
                and per_image[second_stage][sample_id].get(metric, "") not in ("", "nan")
            ]
            difference = np.asarray(
                [
                    float(per_image[second_stage][sample_id][metric])
                    - float(per_image[first_stage][sample_id][metric])
                    for sample_id in valid_ids
                ],
                dtype=np.float64,
            )
            bootstrap = np.empty(args.bootstrap_repeats, dtype=np.float64)
            for index in range(args.bootstrap_repeats):
                selected = rng.integers(0, difference.size, difference.size)
                bootstrap[index] = difference[selected].mean()
            lower, upper = np.percentile(bootstrap, [2.5, 97.5])
            comparisons.append(
                {
                    "comparison": f"{second_stage}-{first_stage}",
                    "metric": metric,
                    "valid_image_pairs": len(valid_ids),
                    "mean_per_image_difference_pp": 100 * difference.mean(),
                    "ci95_lower_pp": 100 * lower,
                    "ci95_upper_pp": 100 * upper,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "bootstrap_repeats": args.bootstrap_repeats,
                    "bootstrap_seed": SEED,
                }
            )

    write_csv(output / "表_F至最终G_门控蒸馏校准逐项消融.csv", rows)
    write_csv(output / "表_相邻阶段配对Bootstrap.csv", comparisons)
    labels = [row["stage"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.8, 4.7), dpi=220)
    axis.bar(x - width / 2, [row["mIoU_percent"] for row in rows], width, label="mIoU")
    axis.bar(x + width / 2, [row["lesion_IoU_percent"] for row in rows], width, label="Lesion IoU")
    axis.set(xticks=x, xticklabels=labels, xlabel="Incremental stage", ylabel="Test score (%)", ylim=(0, 100))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    for bar in axis.patches:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "图_F至最终G_门控蒸馏校准逐项消融.png", bbox_inches="tight")
    figure.savefig(output / "图_F至最终G_门控蒸馏校准逐项消融.pdf", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "protocol": {
            "split": "same frozen 36-image internal test",
            "all_checkpoints_selected_on": "validation only",
            "retraining_for_this_analysis": False,
            "new_test_evaluation": "G1 distilled checkpoint only; one evaluation after validation selection",
            "bootstrap": f"paired by image, {args.bootstrap_repeats} repeats, seed {SEED}",
        },
        "stages": rows,
        "paired_bootstrap": comparisons,
        "interpretation": [
            "G0 improves both aggregate mIoU and lesion IoU over the strict hard gate F.",
            "Distillation (G1) improves aggregate mIoU but slightly lowers aggregate lesion IoU; it should not be described as uniformly beneficial.",
            "Validation-only logit calibration (G2/final G) provides the largest final lesion-IoU gain.",
        ],
    }
    (output / "实验汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

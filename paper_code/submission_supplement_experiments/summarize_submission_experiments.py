#!/usr/bin/env python3
"""Aggregate submission supplement experiments without touching checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_materials" / "投稿前补充实验_20260829"
SEEDS = (20260803, 20260804, 20260805, 20260806, 20260807)
METRICS = (
    "mIoU", "mDice", "pixel_accuracy", "IoU_lesion", "Dice_lesion",
    "Precision_lesion", "Recall_lesion",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_per_image(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return {r["sample_id"]: r for r in csv.DictReader(handle)}


def count_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def paths_for_seed(seed):
    if seed == 20260803:
        return {
            "G": ROOT / "work_dirs/dadou_corrected_F_final_test/test_summary.json",
            "G_per": ROOT / "work_dirs/dadou_corrected_F_final_test/test_per_image.csv",
            "A": ROOT / "work_dirs/label_fraction_100_seed20260803/A_test/test_summary.json",
            "A_per": ROOT / "work_dirs/label_fraction_100_seed20260803/A_test/test_per_image.csv",
            "ST++": ROOT / "work_dirs/dadou_stpp_aligned_seed20260803/training_summary.json",
            "ST++_per": ROOT / "work_dirs/dadou_stpp_aligned_seed20260803/test_per_image.csv",
            "G0": ROOT / "work_dirs/dadou_corrected_F/training_summary.json",
            "G0_per": ROOT / "work_dirs/dadou_corrected_F/test_per_image_metrics.csv",
            "G0-Cal": ROOT / "work_dirs/dadou_G0_Cal_final_test/test_summary.json",
            "G0-Cal_per": ROOT / "work_dirs/dadou_G0_Cal_final_test/test_per_image.csv",
            "G1": ROOT / "paper_materials/一区提升实验补充_20260826/04_门控蒸馏校准逐项消融/G_门控加蒸馏_冻结测试/test_summary.json",
            "G1_per": ROOT / "paper_materials/一区提升实验补充_20260826/04_门控蒸馏校准逐项消融/G_门控加蒸馏_冻结测试/test_per_image.csv",
        }
    base = ROOT / "work_dirs" / f"submission_seed_{seed}"
    return {
        "G": base / "G_final_test/test_summary.json",
        "G_per": base / "G_final_test/test_per_image.csv",
        "A": base / "A_test/test_summary.json",
        "A_per": base / "A_test/test_per_image.csv",
        "ST++": base / "STPP/training_summary.json",
        "ST++_per": base / "STPP/test_per_image.csv",
        "G0": base / "G0_test/test_summary.json",
        "G0_per": base / "G0_test/test_per_image.csv",
        "G0-Cal": base / "G0_calibrated_test/test_summary.json",
        "G0-Cal_per": base / "G0_calibrated_test/test_per_image.csv",
        "G1": base / "G1_test/test_summary.json",
        "G1_per": base / "G1_test/test_per_image.csv",
    }


def extract_metrics(path, method):
    data = load_json(path)
    if method == "ST++":
        return data["frozen_test_metrics"]
    if method == "G0" and "test_metrics" in data:
        return data["test_metrics"]
    return data["metrics"]


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def paired_t_ci(differences):
    values = np.asarray(differences, dtype=float)
    mean = values.mean()
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(mean), float(mean - half), float(mean + half)


def paired_image_bootstrap(path_a, path_b, metric, rng, repeats=20000):
    a, b = read_per_image(path_a), read_per_image(path_b)
    ids = sorted(set(a) & set(b))
    if len(ids) != 37:
        raise RuntimeError(f"Expected 37 paired images, got {len(ids)}")
    diffs = np.asarray([
        float(a[i][metric]) - float(b[i][metric]) for i in ids
        if math.isfinite(float(a[i][metric])) and math.isfinite(float(b[i][metric]))
    ])
    if not len(diffs):
        raise RuntimeError(f"No finite paired values for {metric}")
    draws = rng.integers(0, len(diffs), size=(repeats, len(diffs)))
    means = diffs[draws].mean(axis=1)
    return len(diffs), float(diffs.mean()), *(float(x) for x in np.quantile(means, [0.025, 0.975]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {}
    per_seed_rows = []
    for seed in SEEDS:
        paths = paths_for_seed(seed)
        raw[seed] = {}
        for method in ("G", "ST++", "A", "G0", "G0-Cal", "G1"):
            if not paths[method].exists():
                raise FileNotFoundError(paths[method])
            metrics = extract_metrics(paths[method], method)
            raw[seed][method] = metrics
            row = {"seed": seed, "method": method}
            row.update({m: metrics[m] * 100 for m in METRICS})
            per_seed_rows.append(row)
    write_csv(OUT / "表1_五种子逐次结果.csv", per_seed_rows, ["seed", "method", *METRICS])

    summary_rows = []
    for method in ("G", "ST++", "A", "G0", "G0-Cal", "G1"):
        for metric in METRICS:
            values = [raw[s][method][metric] * 100 for s in SEEDS]
            mean, sd = mean_sd(values)
            summary_rows.append({
                "method": method, "metric": metric, "n_seeds": len(SEEDS),
                "mean_percent": mean, "sd_percent": sd,
                "mean_plus_minus_sd": f"{mean:.2f} ± {sd:.2f}",
            })
    write_csv(
        OUT / "表2_五种子均值标准差.csv", summary_rows,
        ["method", "metric", "n_seeds", "mean_percent", "sd_percent", "mean_plus_minus_sd"],
    )

    diff_rows = []
    for left, right in (("G", "ST++"), ("G", "A"), ("G0-Cal", "G0"), ("G1", "G0-Cal"), ("G", "G1"), ("G", "G0")):
        for metric in METRICS:
            diffs = [(raw[s][left][metric] - raw[s][right][metric]) * 100 for s in SEEDS]
            mean, low, high = paired_t_ci(diffs)
            diff_rows.append({
                "comparison": f"{left} - {right}", "metric": metric,
                "n_paired_seeds": len(SEEDS), "mean_difference_pp": mean,
                "ci95_low_pp": low, "ci95_high_pp": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            })
    write_csv(
        OUT / "表3_五种子配对差值95CI.csv", diff_rows,
        ["comparison", "metric", "n_paired_seeds", "mean_difference_pp",
         "ci95_low_pp", "ci95_high_pp", "ci_excludes_zero"],
    )

    # The document explicitly asks for a paired bootstrap.  Use the original
    # pre-declared seed and matched 37-image test set; report per-image means.
    rng = np.random.default_rng(20260829)
    p = paths_for_seed(20260803)
    bootstrap_rows = []
    for left, right in (("G", "A"), ("G0-Cal", "G0"), ("G1", "G0-Cal"), ("G", "G1"), ("G", "G0")):
        for metric in ("mIoU", "IoU_lesion"):
            paired_images, mean, low, high = paired_image_bootstrap(
                p[f"{left}_per"], p[f"{right}_per"], metric, rng)
            bootstrap_rows.append({
                "seed": 20260803, "comparison": f"{left} - {right}",
                "metric": metric, "paired_images": paired_images, "bootstrap_repeats": 20000,
                "mean_per_image_difference_pp": mean * 100,
                "ci95_low_pp": low * 100, "ci95_high_pp": high * 100,
            })
    write_csv(
        OUT / "表4_原始种子逐图配对Bootstrap.csv", bootstrap_rows,
        ["seed", "comparison", "metric", "paired_images", "bootstrap_repeats",
         "mean_per_image_difference_pp", "ci95_low_pp", "ci95_high_pp"],
    )

    # Figure 1: paired seed results for the principal comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("mIoU", "IoU_lesion"), ("mIoU (%)", "Lesion IoU (%)")):
        g = np.asarray([raw[s]["G"][metric] * 100 for s in SEEDS])
        st = np.asarray([raw[s]["ST++"][metric] * 100 for s in SEEDS])
        for i in range(len(SEEDS)):
            ax.plot([0, 1], [g[i], st[i]], color="#9aa0a6", lw=1, alpha=0.8)
        ax.scatter(np.zeros(len(g)), g, color="#c62828", label="Final G", zorder=3)
        ax.scatter(np.ones(len(st)), st, color="#1565c0", label="ST++", zorder=3)
        ax.errorbar([0, 1], [g.mean(), st.mean()],
                    yerr=[g.std(ddof=1), st.std(ddof=1)], fmt="D", color="black",
                    capsize=4, label="Mean ± SD", zorder=4)
        ax.set_xticks([0, 1], ["Final G", "ST++"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "图1_G与STPP五种子配对比较.png", dpi=400)
    fig.savefig(OUT / "图1_G与STPP五种子配对比较.pdf")
    plt.close(fig)

    # Figure 2: staged evidence for calibration/distillation.
    methods = ("G0", "G0-Cal", "G1", "G")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("mIoU", "IoU_lesion"), ("mIoU (%)", "Lesion IoU (%)")):
        means, sds = [], []
        for method in methods:
            values = [raw[s][method][metric] * 100 for s in SEEDS]
            mean, sd = mean_sd(values); means.append(mean); sds.append(sd)
        ax.bar(methods, means, yerr=sds, capsize=4,
               color=["#607d8b", "#90a4ae", "#ef6c00", "#c62828"])
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "图2_G0校准蒸馏链五种子比较.png", dpi=400)
    fig.savefig(OUT / "图2_G0校准蒸馏链五种子比较.pdf")
    plt.close(fig)

    # Integrity audit with unambiguous actual seeds and distinct checkpoint hashes.
    hashes = {}
    integrity_per_seed = {}
    for seed in SEEDS[1:]:
        base = ROOT / "work_dirs" / f"submission_seed_{seed}"
        for label, rel in {
            "A": "A/best_validation_mIoU.pth",
            "G0": "G0/best_validation_mIoU.pth",
            "G1": "G1_distilled/best_validation_mIoU.pth",
            "G": "G_final_calibrated/best_calibrated_student.pth",
            "ST++": "STPP/stage3_best_validation_mIoU.pth",
        }.items():
            path = base / rel
            hashes[f"{seed}_{label}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        effective_seeds = {
            "A": load_json(base / "A/effective_training_config.json")["seed"],
            "G0": load_json(base / "G0/effective_training_config.json")["seed"],
            "ST++": load_json(base / "STPP/aligned_protocol.json")["seed"],
        }
        epoch_rows = {
            "A": count_csv_rows(base / "A/epoch_metrics.csv"),
            "G0": count_csv_rows(base / "G0/epoch_metrics.csv"),
            "G1": count_csv_rows(base / "G1_distilled/training_history.csv"),
            "ST++_stage1": count_csv_rows(base / "STPP/stage1_epoch_metrics.csv"),
            "ST++_stage2": count_csv_rows(base / "STPP/stage2_epoch_metrics.csv"),
            "ST++_stage3": count_csv_rows(base / "STPP/stage3_epoch_metrics.csv"),
        }
        per_image_rows = {
            "G0": count_csv_rows(base / "G0_test/test_per_image.csv"),
            "G0-Cal": count_csv_rows(base / "G0_calibrated_test/test_per_image.csv"),
            "G1": count_csv_rows(base / "G1_test/test_per_image.csv"),
            "G": count_csv_rows(base / "G_final_test/test_per_image.csv"),
            "ST++": count_csv_rows(base / "STPP/test_per_image.csv"),
            "A": count_csv_rows(base / "A_test/test_per_image.csv"),
        }
        prediction_masks = {
            "G0": len(list((base / "G0_test/predictions/masks").glob("*.png"))),
            "G0-Cal": len(list((base / "G0_calibrated_test/predictions/masks").glob("*.png"))),
            "G1": len(list((base / "G1_test/predictions/masks").glob("*.png"))),
            "G": len(list((base / "G_final_test/predictions/masks").glob("*.png"))),
            "ST++": len(list((base / "STPP/test_predictions/masks").glob("*.png"))),
            "A": len(list((base / "A_test/predictions/masks").glob("*.png"))),
        }
        passed = (
            all(value == seed for value in effective_seeds.values())
            and epoch_rows == {"A": 60, "G0": 60, "G1": 18,
                              "ST++_stage1": 20, "ST++_stage2": 20,
                              "ST++_stage3": 20}
            and all(value == 37 for value in per_image_rows.values())
            and all(value == 37 for value in prediction_masks.values())
        )
        integrity_per_seed[str(seed)] = {
            "effective_seeds": effective_seeds,
            "epoch_rows": epoch_rows,
            "test_per_image_rows": per_image_rows,
            "prediction_mask_counts": prediction_masks,
            "pass": passed,
        }
    audit = {
        "seeds_declared_before_new_runs": list(SEEDS),
        "all_new_runs_pass_integrity_audit": all(
            value["pass"] for value in integrity_per_seed.values()),
        "integrity_per_seed": integrity_per_seed,
        "new_training_lengths": {
            "A": "60 epochs x 98 steps", "G0": "60 epochs x 98 steps",
            "G1": "18 epochs x 50 steps", "ST++": "3 stages x 20 epochs x 98 steps",
        },
        "all_new_checkpoint_hashes_unique_within_method": all(
            len({v for k, v in hashes.items() if k.endswith("_" + method)}) == 4
            for method in ("A", "G0", "G1", "G", "ST++")
        ),
        "checkpoint_sha256": hashes,
        "important_compute_caveat": (
            "Final G reproduces the existing multi-stage method: A=5880 updates, "
            "G0=5880 updates, and G1=900 updates. ST++ uses 5880 updates. "
            "Architecture and frozen splits are aligned, but total training compute is not equal."
        ),
        "stpp_protocol_metadata": (
            "Each new training_summary.json and aligned_protocol.json now records the "
            "same actual seed; a prior template-copy metadata mismatch was repaired "
            "without changing checkpoints or metrics."
        ),
    }
    (OUT / "完整性审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    key = {(r["comparison"], r["metric"]): r for r in diff_rows}
    g_m = next(r for r in summary_rows if r["method"] == "G" and r["metric"] == "mIoU")
    st_m = next(r for r in summary_rows if r["method"] == "ST++" and r["metric"] == "mIoU")
    a_m = next(r for r in summary_rows if r["method"] == "A" and r["metric"] == "mIoU")
    d_m = key[("G - ST++", "mIoU")]
    d_a = key[("G - A", "mIoU")]
    d_cal = key[("G0-Cal - G0", "mIoU")]
    d_final = key[("G - G1", "mIoU")]
    with (OUT / "表7_G与STPP边界分尺度配对95CI.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        boundary_differences = {r["metric"]: r for r in csv.DictReader(handle)}
    b_f1 = boundary_differences["Boundary_F1_percent"]
    small = boundary_differences["small_component_recall_percent"]
    label_summary = load_json(OUT / "标签比例实验汇总.json")
    label_gains = label_summary["gains"]
    readme = f"""# 投稿前补充实验（2026-08-29）

## 已完成

- 最终 G 与 ST++-aligned：5 个预先固定种子，统一 108/27/37/1390 划分，逐种子冻结测试。
- 纯监督 A：同一 5 个种子逐种子冻结测试。
- G0、G0-Cal、G1、最终 G：5 种子逐阶段对照。
- 原始种子 37 张测试图上的逐图配对 Bootstrap（20,000 次）。
- G 与 ST++ 的五种子 Boundary F1 和小/中/大病斑连通域召回。
- 25/50/75/100% 标签比例曲线（scene_cluster_id 整组、四类分层、固定弱标注池）。
- 所有新增权重均有独立 SHA-256，训练轮数与测试样本数已审计。

## 主要结果

- 最终 G mIoU：{g_m['mean_percent']:.2f} ± {g_m['sd_percent']:.2f}%（5 seeds）。
- ST++ mIoU：{st_m['mean_percent']:.2f} ± {st_m['sd_percent']:.2f}%（5 seeds）。
- 纯监督 A mIoU：{a_m['mean_percent']:.2f} ± {a_m['sd_percent']:.2f}%（5 seeds）。
- G−A：{d_a['mean_difference_pp']:.2f} pp，95% CI [{d_a['ci95_low_pp']:.2f}, {d_a['ci95_high_pp']:.2f}]，支持最终 G 稳定优于纯监督基线。
- G−ST++：{d_m['mean_difference_pp']:.2f} pp，95% CI [{d_m['ci95_low_pp']:.2f}, {d_m['ci95_high_pp']:.2f}]。
- 置信区间跨 0，因此不能写“G 显著优于 ST++”；可写成性能相当，并强调门控可解释性与像素级证据链。
- Boundary F1：G−ST++ = {float(b_f1['mean_difference_pp']):.2f} pp，95% CI [{float(b_f1['ci95_low_pp']):.2f}, {float(b_f1['ci95_high_pp']):.2f}]。
- 小病斑召回：G−ST++ = {float(small['mean_difference_pp']):.2f} pp，95% CI [{float(small['ci95_low_pp']):.2f}, {float(small['ci95_high_pp']):.2f}]；固定定义为连通域 <64 像素、预测重叠率 ≥10%。
- G0-Cal 相对 G0 的 mIoU 平均变化为 {d_cal['mean_difference_pp']:.2f} pp，说明直接校准不能解释最终提升；最终 G 相对 G1 提升 {d_final['mean_difference_pp']:.2f} pp。
- 标签比例下 G 相对纯监督 A 的 mIoU 增益依次为 {', '.join(f"{r['G_minus_A_mIoU_pp']:.2f}" for r in label_gains)} pp；病斑 IoU 增益依次为 {', '.join(f"{r['G_minus_A_lesion_IoU_pp']:.2f}" for r in label_gains)} pp。

## 论文表述边界

1. 本项目是 ST++ 的对齐复现（official method core + project adaptation），不是官方仓库逐行原样运行。
2. 最终 G 的推理模型与 ST++ 同为单个 DeepLabV3+–MobileNetV3；但 G 的训练链包含 A、G0、G1 三段，总训练计算量高于 ST++。不能声称严格等训练预算公平。
3. 校准参数只由 27 张验证集选择；每个冻结 checkpoint 的 37 张测试集只评估一次。
4. 图表中的误差条为种子间样本标准差；差异置信区间为 5 个配对种子的 t 区间。
5. 标签比例曲线只有预先固定种子 20260803，是趋势性/探索性补充，不能写成多种子显著性结论。
6. 为保证场景整组且四类嵌套分层，四个标签端点实际为 26/55/81/108 张（21/39/59/78 个场景）。

## 尚不能完成并应 pass 的文档项目

- 新增“第三个独立公共像素级外部数据集”：当前工作区没有新的、来源独立且带三类像素标注的数据，不能用现有数据换名补数；已有 MH-SoyaHealthVision 外部集结果继续保留。
- 两名植保专家严重度评分与一致性：当前没有专家评分数据或专家授权，不能由模型代替专家填写。
"""
    (OUT / "README_补充实验结论与边界.md").write_text(readme, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

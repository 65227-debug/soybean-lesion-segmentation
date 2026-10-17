#!/usr/bin/env python3
"""Audit and summarize protocol-aligned strong SSL baseline reproductions.

This script never loads a test set for model selection and never modifies model
checkpoints.  It only reads completed run artifacts and creates paper-ready
tables/curves plus a machine-readable integrity report.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mmcv
import mmengine
import mmseg
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "paper_code" / "strong_ssl_baselines"
OUT = ROOT / "paper_materials" / "强半监督基线复现_20260827"

RUNS = {
    "UniMatch": ROOT / "work_dirs" / "dadou_unimatch_aligned_seed20260803",
    "ST++": ROOT / "work_dirs" / "dadou_stpp_aligned_seed20260803",
    "U2PL": ROOT / "work_dirs" / "dadou_u2pl_aligned_seed20260803",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem: str):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def check_run(name: str, run: Path, summary: dict) -> dict:
    if name == "ST++":
        epoch_files = [run / f"stage{i}_epoch_metrics.csv" for i in (1, 2, 3)]
        epoch_rows = sum(len(read_csv(path)) for path in epoch_files)
        checkpoint = run / "stage3_best_validation_mIoU.pth"
        pseudo_count = len(list((run / "pseudo_masks").glob("*.png")))
        ranking = read_csv(run / "reliability_ranking.csv")
        reliable_count = sum(row["reliable"].strip().lower() == "true" for row in ranking)
        unreliable_count = len(ranking) - reliable_count
    else:
        epoch_rows = len(read_csv(run / "epoch_metrics.csv"))
        checkpoint = run / "best_validation_mIoU.pth"
        pseudo_count = None
        reliable_count = None
        unreliable_count = None

    masks = list((run / "test_predictions" / "masks").glob("*.png"))
    overlays = [path for path in (run / "test_predictions" / "overlays").iterdir() if path.is_file()]
    per_image = read_csv(run / "test_per_image.csv")
    checks = {
        "summary_exists": (run / "training_summary.json").is_file(),
        "checkpoint_exists": checkpoint.is_file(),
        "epoch_rows": epoch_rows,
        "epoch_rows_expected": 60,
        "total_optimizer_steps": summary["total_optimizer_steps"],
        "total_optimizer_steps_expected": 5880,
        "test_mask_count": len(masks),
        "test_overlay_count": len(overlays),
        "test_per_image_rows": len(per_image),
        "test_count_expected": 36,
        "pseudo_mask_count": pseudo_count,
        "reliable_count": reliable_count,
        "unreliable_count": unreliable_count,
    }
    checks["pass"] = (
        checks["summary_exists"]
        and checks["checkpoint_exists"]
        and epoch_rows == 60
        and summary["total_optimizer_steps"] == 5880
        and len(masks) == len(overlays) == len(per_image) == 36
        and (
            name != "ST++"
            or (pseudo_count == 1390 and reliable_count == unreliable_count == 695)
        )
    )
    return checks


def make_curves():
    uni = read_csv(RUNS["UniMatch"] / "epoch_metrics.csv")
    u2pl = read_csv(RUNS["U2PL"] / "epoch_metrics.csv")
    stpp = []
    for stage_idx in (1, 2, 3):
        rows = read_csv(RUNS["ST++"] / f"stage{stage_idx}_epoch_metrics.csv")
        for row in rows:
            row = dict(row)
            row["global_epoch"] = str((stage_idx - 1) * 20 + int(row["epoch"]))
            stpp.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for name, rows in (("UniMatch", uni), ("ST++", stpp), ("U2PL", u2pl)):
        xkey = "global_epoch" if name == "ST++" else "epoch"
        x = np.array([int(row[xkey]) for row in rows])
        axes[0].plot(x, [100 * float(row["val_mIoU"]) for row in rows], label=name, lw=1.8)
        axes[1].plot(x, [100 * float(row["val_IoU_lesion"]) for row in rows], label=name, lw=1.8)
    for ax, ylabel in zip(axes, ("Validation mIoU (%)", "Validation lesion IoU (%)")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 60)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes:
        ax.axvline(20, color="0.65", ls="--", lw=0.8)
        ax.axvline(40, color="0.65", ls="--", lw=0.8)
    axes[0].text(20.5, axes[0].get_ylim()[0] + 0.5, "ST++ stage boundary", fontsize=7, color="0.4")
    save_figure(fig, "图01_强半监督基线验证曲线")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    axes[0].plot([int(r["epoch"]) for r in uni], [float(r["loss_total"]) for r in uni], label="total")
    axes[0].plot([int(r["epoch"]) for r in uni], [float(r["loss_supervised"]) for r in uni], label="supervised")
    axes[0].set_title("UniMatch")
    axes[1].plot([int(r["global_epoch"]) for r in stpp], [float(r["loss_total"]) for r in stpp], label="total")
    axes[1].plot([int(r["global_epoch"]) for r in stpp], [float(r["loss_supervised"]) for r in stpp], label="supervised")
    axes[1].plot([int(r["global_epoch"]) for r in stpp], [float(r["loss_pseudo"]) for r in stpp], label="pseudo")
    axes[1].axvline(20, color="0.65", ls="--", lw=0.8)
    axes[1].axvline(40, color="0.65", ls="--", lw=0.8)
    axes[1].set_title("ST++")
    axes[2].plot([int(r["epoch"]) for r in u2pl], [float(r["loss_total"]) for r in u2pl], label="total")
    axes[2].plot([int(r["epoch"]) for r in u2pl], [float(r["loss_supervised"]) for r in u2pl], label="supervised")
    axes[2].plot([int(r["epoch"]) for r in u2pl], [float(r["loss_unsupervised"]) for r in u2pl], label="unsupervised")
    axes[2].plot([int(r["epoch"]) for r in u2pl], [float(r["loss_contrastive"]) for r in u2pl], label="contrastive")
    axes[2].set_title("U2PL")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "图02_强半监督基线损失曲线")


def make_comparison(rows: list[dict]):
    labels = ["A", "C", "G", "UniMatch", "ST++", "U2PL"]
    miou = np.array([float(row["test_mIoU_percent"]) for row in rows])
    lesion = np.array([float(row["test_lesion_IoU_percent"]) for row in rows])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    bars1 = ax.bar(x - width / 2, miou, width, label="mIoU")
    bars2 = ax.bar(x + width / 2, lesion, width, label="Lesion IoU")
    ax.set_ylabel("Test score (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(45, max(miou.max(), lesion.max()) + 6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)
    save_figure(fig, "图03_与原A_C_G的冻结测试对比")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {name: read_json(path / "training_summary.json") for name, path in RUNS.items()}
    integrity = {
        "status": "PASS",
        "runs": {name: check_run(name, RUNS[name], summaries[name]) for name in RUNS},
    }
    if not all(item["pass"] for item in integrity["runs"].values()):
        integrity["status"] = "FAIL"

    legacy = read_csv(ROOT / "work_dirs" / "dadou_final_G_comparison" / "ABCDEFG_统一测试对比.csv")
    selected_legacy = {row["group"]: row for row in legacy if row["group"] in {"A", "C", "G"}}
    table = []
    for group in ("A", "C", "G"):
        row = selected_legacy[group]
        table.append({
            "method": f"{group}: {row['method']}",
            "category": "existing_internal_baseline",
            "seed": "20260803",
            "validation_best_epoch": "see original run",
            "test_mIoU": f"{float(row['test_mIoU']):.10f}",
            "test_lesion_IoU": f"{float(row['test_IoU_lesion']):.10f}",
            "test_lesion_Dice": f"{float(row['test_Dice_lesion']):.10f}",
            "test_lesion_precision": f"{float(row['test_Precision_lesion']):.10f}",
            "test_lesion_recall": f"{float(row['test_Recall_lesion']):.10f}",
            "test_mIoU_percent": f"{100 * float(row['test_mIoU']):.2f}",
            "test_lesion_IoU_percent": f"{100 * float(row['test_IoU_lesion']):.2f}",
        })
    for name in ("UniMatch", "ST++", "U2PL"):
        summary = summaries[name]
        metrics = summary["frozen_test_metrics"]
        best_epoch = summary.get("best_epoch", summary.get("stage3_best_epoch"))
        table.append({
            "method": name,
            "category": "new_protocol_aligned_strong_baseline",
            "seed": str(summary["seed"]),
            "validation_best_epoch": str(best_epoch),
            "test_mIoU": f"{metrics['mIoU']:.10f}",
            "test_lesion_IoU": f"{metrics['IoU_lesion']:.10f}",
            "test_lesion_Dice": f"{metrics['Dice_lesion']:.10f}",
            "test_lesion_precision": f"{metrics['Precision_lesion']:.10f}",
            "test_lesion_recall": f"{metrics['Recall_lesion']:.10f}",
            "test_mIoU_percent": f"{100 * metrics['mIoU']:.2f}",
            "test_lesion_IoU_percent": f"{100 * metrics['IoU_lesion']:.2f}",
        })
    fields = list(table[0].keys())
    write_csv(OUT / "表01_强半监督基线与A_C_G冻结测试对比.csv", table, fields)

    details = []
    for name in ("UniMatch", "ST++", "U2PL"):
        summary = summaries[name]
        val = summary.get("best_validation_metrics", summary.get("stage3_best_validation_metrics"))
        test = summary["frozen_test_metrics"]
        details.append({
            "method": name,
            "best_epoch": summary.get("best_epoch", summary.get("stage3_best_epoch")),
            "val_mIoU_percent": f"{100 * val['mIoU']:.2f}",
            "val_lesion_IoU_percent": f"{100 * val['IoU_lesion']:.2f}",
            "test_mIoU_percent": f"{100 * test['mIoU']:.2f}",
            "test_background_IoU_percent": f"{100 * test['IoU_background']:.2f}",
            "test_healthy_IoU_percent": f"{100 * test['IoU_healthy_leaf']:.2f}",
            "test_lesion_IoU_percent": f"{100 * test['IoU_lesion']:.2f}",
            "test_mDice_percent": f"{100 * test['mDice']:.2f}",
            "test_lesion_Dice_percent": f"{100 * test['Dice_lesion']:.2f}",
            "test_lesion_precision_percent": f"{100 * test['Precision_lesion']:.2f}",
            "test_lesion_recall_percent": f"{100 * test['Recall_lesion']:.2f}",
            "optimizer_steps": summary["total_optimizer_steps"],
            "elapsed_seconds": f"{summary['elapsed_seconds']:.1f}",
        })
    write_csv(OUT / "表02_三种强基线完整指标.csv", details, list(details[0].keys()))

    stpp = summaries["ST++"]["frozen_test_metrics"]
    g = selected_legacy["G"]
    conclusions = {
        "protocol": read_json(CODE_DIR / "aligned_protocol.json"),
        "official_references": read_json(CODE_DIR / "OFFICIAL_REFERENCES.json"),
        "integrity": integrity,
        "summaries": summaries,
        "single_seed_comparison": {
            "ST++_minus_G_mIoU_percentage_points": 100 * (stpp["mIoU"] - float(g["test_mIoU"])),
            "ST++_minus_G_lesion_IoU_percentage_points": 100 * (stpp["IoU_lesion"] - float(g["test_IoU_lesion"])),
            "interpretation": "Single-seed result only; do not claim statistical superiority without repeated seeds.",
        },
    }
    with (OUT / "强基线复现汇总.json").open("w", encoding="utf-8") as handle:
        json.dump(conclusions, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (OUT / "01_产物完整性核验.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, ensure_ascii=False, indent=2)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "mmsegmentation": mmseg.__version__,
        "mmengine": mmengine.__version__,
        "mmcv": mmcv.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }
    with (OUT / "02_运行环境.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, ensure_ascii=False, indent=2)

    make_curves()
    make_comparison(table)

    for filename in ("aligned_protocol.json", "OFFICIAL_REFERENCES.json"):
        shutil.copy2(CODE_DIR / filename, OUT / filename)
    shutil.copy2(Path(__file__), OUT / "本次实际执行的汇总脚本.py")
    for name, run in RUNS.items():
        shutil.copy2(run / "training_summary.json", OUT / f"原始训练摘要_{name}.json")
        if name == "ST++":
            for stage in (1, 2, 3):
                shutil.copy2(run / f"stage{stage}_epoch_metrics.csv", OUT / f"原始逐轮指标_{name}_stage{stage}.csv")
        else:
            shutil.copy2(run / "epoch_metrics.csv", OUT / f"原始逐轮指标_{name}.csv")
        shutil.copy2(run / "test_per_image.csv", OUT / f"原始逐图测试指标_{name}.csv")

    comparison_table = "\n".join(
        f"|{row['method']}|{100 * float(row['test_mIoU']):.2f}%|"
        f"{100 * float(row['test_lesion_IoU']):.2f}%|"
        f"{100 * float(row['test_lesion_Dice']):.2f}%|"
        f"{100 * float(row['test_lesion_precision']):.2f}%|"
        f"{100 * float(row['test_lesion_recall']):.2f}%|"
        for row in table
    )
    readme = f"""# UniMatch、ST++、U²PL 协议对齐复现结果

## 结论先行

三种方法均已在冻结协议下完成 60 epoch / 5880 次优化，最佳模型只依据 36 张验证图像的 mIoU 选择，随后对 36 张测试图像评估一次。完整性核验：**{integrity['status']}**。

|方法|测试 mIoU|测试病斑 IoU|病斑 Dice|病斑 Precision|病斑 Recall|
|---|---:|---:|---:|---:|---:|
{comparison_table}

ST++−G：mIoU 差值 {100 * (stpp['mIoU'] - float(g['test_mIoU'])):.2f} 个百分点；病斑 IoU 差值 {100 * (stpp['IoU_lesion'] - float(g['test_IoU_lesion'])):.2f} 个百分点。

## 公平协议

- 数据：108 张像素级训练图、1390 张纯未标注图、36 张验证图、36 张冻结测试图；协议审计未发现跨划分 SHA-256 重复。
- 模型：DeepLabV3+ + MobileNetV3-Large，512×512，ImageNet 初始化。
- 预算：每种方法总计 5880 个优化步；AdamW、初始学习率 3e-4、余弦衰减、相同监督损失与基础增强。
- 未标注数据加载器只使用 `sample_id`、`image_path`、`source_image_path`，没有使用病种名、LayerCAM、叶片模型、G 的掩膜或连续权重。
- 选模：仅按验证 mIoU；测试集不参与阈值、轮数、早停或模型选择。

## 方法实现边界

本工作属于**在大豆任务上的协议对齐方法复现**，不是对 VOC/Cityscapes 论文主结果的逐字节复刻。方法核心来自固定提交号的官方实现，详见 `OFFICIAL_REFERENCES.json`。

- UniMatch：保留双强增强、CutMix、特征扰动与 0.95 置信度门控。
- ST++：保留三阶段训练、跨 checkpoint 一致性排序、前 50% 可靠样本优先自训练，再扩展至全部未标注样本。
- U²PL：保留 EMA 教师、熵可靠/不可靠像素划分和类别记忆队列。官方面向 19/21 类数据的负类排序区间 3:20 无法直接用于 3 类问题，因此在训练前固定映射为 rank 1:3（Top-1 之外的类别），未根据测试结果调整。

## 目录说明

- `表01_强半监督基线与A_C_G冻结测试对比.csv`：论文主表或补充表数据。
- `表02_三种强基线完整指标.csv`：三类 IoU、Dice、病斑精确率/召回率及训练时间。
- `图01_强半监督基线验证曲线.*`：验证 mIoU 与病斑 IoU。
- `图02_强半监督基线损失曲线.*`：各方法损失分量；ST++ 虚线表示阶段重启。
- `图03_与原A_C_G的冻结测试对比.*`：A/C/G 与三种强基线对比。
- `原始逐轮指标_*`、`原始逐图测试指标_*`：Origin 可直接导入的原始 CSV。
- 模型 checkpoint 和 1390 张 ST++ 中间伪标签只保留在 `work_dirs`，未复制到论文材料目录。

## 论文使用限制与下一步

当前汇总只有 seed=20260803，不能报告均值±标准差或显著性。多种子比较应使用相同的预设种子，并报告 mean±std 及配对 bootstrap/置换检验。
"""
    (OUT / "README_强半监督基线复现结论.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "integrity": integrity, "rows": table}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

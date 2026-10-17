#!/usr/bin/env python
"""Unified 512x512 inference benchmark for the frozen teacher and final G."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="paper_materials/一区提升实验补充_20260826/03_统一效率测试",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations-per-repeat", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    init_default_scope("mmseg")
    config = Config.fromfile(config_path)
    model = MODELS.build(config.model)
    missing, unexpected = model.load_state_dict(checkpoint_state(checkpoint_path), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if hasattr(model, "auxiliary_head") and model.auxiliary_head is not None:
        model.auxiliary_head = None
    active_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.eval().to(device)
    return model, total_parameters, active_parameters


def forward_prediction(model, tensor: torch.Tensor) -> torch.Tensor:
    features = model.extract_feat(tensor)
    logits = model.decode_head.forward(features)
    logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


@torch.inference_mode()
def benchmark_model(
    model,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    if device.type != "cuda":
        raise ValueError("This benchmark protocol requires CUDA for event timing and AMP.")
    tensor = torch.randn(1, 3, 512, 512, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            forward_prediction(model, tensor)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        forward_prediction(model, tensor)
    torch.cuda.synchronize(device)
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    block_latency_ms = []
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                forward_prediction(model, tensor)
            end.record()
            torch.cuda.synchronize(device)
            block_latency_ms.append(start.elapsed_time(end) / iterations)
    median_ms = float(statistics.median(block_latency_ms))
    return {
        "warmup_iterations": warmup,
        "timed_iterations": iterations * repeats,
        "repeat_block_latency_ms": block_latency_ms,
        "latency_median_ms": median_ms,
        "latency_mean_ms": float(statistics.mean(block_latency_ms)),
        "latency_std_ms": float(statistics.stdev(block_latency_ms)) if repeats > 1 else 0.0,
        "fps_from_median_latency": 1000.0 / median_ms,
        "peak_gpu_memory_mib": float(peak_memory_mib),
        "precision": "PyTorch autocast FP16",
        "batch_size": 1,
        "input_size": [512, 512],
        "timing_scope": "network forward + logits upsampling + argmax; excludes image I/O and preprocessing",
    }


def flops_with_official_tool(project_root: Path, config_path: Path) -> tuple[float, str]:
    command = [
        str(Path(torch.__file__).resolve().parents[2] / "bin/python"),
        str(project_root / "tools/analysis_tools/get_flops.py"),
        str(config_path),
        "--shape",
        "512",
        "512",
    ]
    # sys.executable is more reliable than reconstructing a conda path.
    import sys

    command[0] = sys.executable
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout
    line = next(line for line in output.splitlines() if line.startswith("Flops:"))
    value = line.split(":", 1)[1].strip()
    if value.endswith("G"):
        gflops = float(value[:-1])
    elif value.endswith("T"):
        gflops = float(value[:-1]) * 1000.0
    else:
        raise ValueError(f"Unsupported FLOPs format: {value}")
    return gflops, output


def gpu_name() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().splitlines()[0]


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    specifications = {
        "Teacher_R50": {
            "config": root / "configs/dadou/deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py",
            "checkpoint": root / "work_dirs/dadou_deeplabv3plus_r50_teacher/best_mIoU_epoch_23_inference.pth",
        },
        "Final_G_student": {
            "config": root / "paper_code/step9_student_training/mmseg_student_config.py",
            "checkpoint": root / "work_dirs/dadou_corrected_F_distilled_calibrated_fine/best_calibrated_student.pth",
        },
    }
    results = []
    for name, specification in specifications.items():
        config_path = specification["config"]
        checkpoint_path = specification["checkpoint"]
        gflops, flops_log = flops_with_official_tool(root, config_path)
        (output_dir / f"{name}_FLOPs原始日志.txt").write_text(flops_log, encoding="utf-8")
        model, total_parameters, active_parameters = build_model(config_path, checkpoint_path, device)
        timing = benchmark_model(
            model,
            device,
            warmup=args.warmup,
            iterations=args.iterations_per_repeat,
            repeats=args.repeats,
        )
        results.append(
            {
                "model": name,
                "architecture": "DeepLabV3+ ResNet-50" if name == "Teacher_R50" else "DeepLabV3+ MobileNetV3-Large",
                "total_parameters": total_parameters,
                "total_parameters_million": total_parameters / 1e6,
                "active_inference_parameters": active_parameters,
                "active_inference_parameters_million": active_parameters / 1e6,
                "GFLOPs_512x512": gflops,
                "checkpoint_size_mib": checkpoint_path.stat().st_size / (1024**2),
                "checkpoint": str(checkpoint_path),
                **timing,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    teacher, student = results
    reduction = {
        "total_parameter_reduction_percent": 100.0 * (1.0 - student["total_parameters"] / teacher["total_parameters"]),
        "gflops_reduction_percent": 100.0 * (1.0 - student["GFLOPs_512x512"] / teacher["GFLOPs_512x512"]),
        "checkpoint_size_reduction_percent": 100.0 * (1.0 - student["checkpoint_size_mib"] / teacher["checkpoint_size_mib"]),
        "latency_reduction_percent": 100.0 * (1.0 - student["latency_median_ms"] / teacher["latency_median_ms"]),
        "speedup_times": teacher["latency_median_ms"] / student["latency_median_ms"],
        "peak_gpu_memory_reduction_percent": 100.0 * (1.0 - student["peak_gpu_memory_mib"] / teacher["peak_gpu_memory_mib"]),
    }
    environment = {
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    summary = {
        "protocol": {
            "weights_frozen": True,
            "model_retraining": False,
            "same_input_and_timing_scope": True,
            "FLOPs_method": "MMSegmentation tools/analysis_tools/get_flops.py; unsupported-op warnings retained in raw logs",
        },
        "environment": environment,
        "models": results,
        "student_vs_teacher": reduction,
    }
    (output_dir / "统一效率测试汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in results:
        rows.append(
            {
                "模型": item["model"],
                "结构": item["architecture"],
                "总参数量_M": f"{item['total_parameters_million']:.3f}",
                "有效推理参数量_M": f"{item['active_inference_parameters_million']:.3f}",
                "GFLOPs_512x512": f"{item['GFLOPs_512x512']:.3f}",
                "Checkpoint_MiB": f"{item['checkpoint_size_mib']:.2f}",
                "延迟中位数_ms": f"{item['latency_median_ms']:.3f}",
                "FPS": f"{item['fps_from_median_latency']:.2f}",
                "峰值显存_MiB": f"{item['peak_gpu_memory_mib']:.2f}",
            }
        )
    import csv

    with (output_dir / "表_教师与最终G统一效率测试.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

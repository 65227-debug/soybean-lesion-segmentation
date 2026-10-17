#!/usr/bin/env python3
"""Check a source-only release; no data, checkpoint, training or network needed."""

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RESOURCES = {
    "mmseg/utils/bpe_simple_vocab_16e6.txt.gz",
    "projects/CAT-Seg/cat_seg/utils/bpe_vocab/bpe_simple_vocab_16e6.txt.gz",
}
FORBIDDEN_TOP = {"data", "datasets", "work_dirs", "outputs", "checkpoints", "paper_materials", "paper_release"}
FORBIDDEN_SUFFIX = {".pth", ".pt", ".ckpt", ".onnx", ".npy", ".npz", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".csv", ".tsv", ".pdf", ".docx", ".pkl", ".pickle", ".zip", ".gz"}
REQUIRED = (
    "LICENSE", "CITATION.cff", "README.md", "setup.py", "mmseg/__init__.py",
    "projects/dadou_teacher/__init__.py", "projects/dadou_teacher/dadou_dataset.py",
    "tools/dadou/generate_three_base_maps.py", "tools/dadou_semantic/train_efficientnet_layercam.py",
    "paper_code/corrected_f_training/corrected_gating.py",
    "paper_code/corrected_f_training/selected_gating_config.json",
    "paper_code/step9_student_training/train_student.py",
    "paper_code/submission_supplement_experiments/frozen_experiment_plan.json",
)


def synthetic_gate_check():
    import numpy as np
    module_path = ROOT / "paper_code/corrected_f_training/corrected_gating.py"
    spec = importlib.util.spec_from_file_location("release_gating_check", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.load_config(ROOT / "paper_code/corrected_f_training/selected_gating_config.json")
    # Three unambiguous regions plus an intentionally uncertain pixel.
    p = np.zeros((3, 32, 32), dtype=np.float32)
    p[0] = 0.95; p[1] = 0.03; p[2] = 0.02
    leaf = np.zeros((32, 32), dtype=np.float32)
    leaf[6:26, 6:26] = 0.99
    p[:, 6:26, 6:26] = np.array([0.02, 0.93, 0.05], dtype=np.float32)[:, None, None]
    p[:, 14:18, 14:18] = np.array([0.02, 0.03, 0.95], dtype=np.float32)[:, None, None]
    p[:, 0, 0] = [0.34, 0.33, 0.33]
    sem = np.full((32, 32), 0.4, dtype=np.float32)
    label, weight, masks = module.gate_pixels(p, leaf, sem, cfg)
    assert set(np.unique(label)).issubset({0, 1, 2, 255})
    assert set(np.unique(label)) == {0, 1, 2, 255}
    assert np.allclose(p.sum(axis=0), 1)
    assert np.all(weight[label == 255] == 0)
    assert np.all((weight >= 0) & (weight <= 1))
    assert np.all(~masks["inner"] | (leaf >= cfg.leaf_threshold))
    assert label[0, 0] == 255 and label[16, 16] == 2
    assert np.isclose(weight[16, 16], 0.95 * (cfg.semantic_weight_floor + (1-cfg.semantic_weight_floor) * 0.4))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true", help="Skip the NumPy/OpenCV synthetic gate check")
    args = parser.parse_args()
    errors = []
    python_count = json_count = file_count = 0
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"Required source missing: {name}")
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        if path.is_symlink():
            errors.append(f"Symlink is not allowed: {rel}"); continue
        if not path.is_file():
            continue
        file_count += 1
        if rel.parts[0] in FORBIDDEN_TOP or (path.suffix.lower() in FORBIDDEN_SUFFIX and str(rel) not in PACKAGE_RESOURCES):
            errors.append(f"Data/artifact file must not be published: {rel}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Unexpected large source file: {rel}")
        if path.suffix == ".py":
            python_count += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
            except SyntaxError as exc:
                errors.append(f"Python syntax: {rel}:{exc.lineno}")
        if path.suffix == ".json":
            json_count += 1
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix in {".py", ".sh", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"/(?:home|Users)/zds(?:/|\b)", text):
                errors.append(f"Original machine path remains: {rel}")
            if re.search(r"gh[pousr]_" + r"[A-Za-z0-9]{30,}|github_pat_" + r"[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text):
                errors.append(f"Possible credential material: {rel}")
    manifest = json.loads((ROOT / "CODE_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, record in manifest["frozen_configs"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"Frozen config hash mismatch: {name}")
    protocol = json.loads((ROOT / "paper_code/strong_ssl_baselines/aligned_protocol.json").read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "paper_code/submission_supplement_experiments/frozen_experiment_plan.json").read_text(encoding="utf-8"))
    expected = {"labeled_train": 108, "validation": 36, "test": 36, "unlabeled": 1390}
    if protocol.get("split_sizes") != expected:
        errors.append("Aligned protocol split sizes must be 108/36/36/1390")
    plan_expected = {**expected, "weak_pool": expected["unlabeled"]}
    del plan_expected["unlabeled"]
    if plan["fixed_internal_splits"] != plan_expected:
        errors.append("Supplementary plan split sizes must match the aligned protocol")
    result = {
        "files_checked": file_count, "python_files_checked": python_count,
        "json_files_checked": json_count, "errors": errors,
        "synthetic_gating_check": "skipped" if args.static_only else synthetic_gate_check(),
        "status": "FAIL" if errors else "PASS",
        "training_performed": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the model and inference source tree without external data."""

import argparse
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RESOURCES = {
    "mmseg/utils/bpe_simple_vocab_16e6.txt.gz",
    "projects/CAT-Seg/cat_seg/utils/bpe_vocab/bpe_simple_vocab_16e6.txt.gz",
}
FORBIDDEN_TOP = {"data", "datasets", "work_dirs", "outputs", "checkpoints", "paper_materials", "paper_release", "paper_code"}
FORBIDDEN_SUFFIX = {".pth", ".pt", ".ckpt", ".onnx", ".npy", ".npz", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".csv", ".tsv", ".pdf", ".docx", ".pkl", ".pickle", ".zip", ".gz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.parse_args()
    manifest = json.loads((ROOT / "CODE_RELEASE_MANIFEST.json").read_text())
    errors = []
    required = ["LICENSE", "CITATION.cff", "README.md", "setup.py", "mmseg/__init__.py"]
    required += manifest["model_configs"] + manifest["inference_tools"]
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"Required source missing: {name}")
    python_count = json_count = file_count = 0
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "__pycache__" in rel.parts or not path.is_file():
            continue
        file_count += 1
        if path.is_symlink():
            errors.append(f"Unexpected symlink: {rel}")
        if rel.parts[0] in FORBIDDEN_TOP or (path.suffix.lower() in FORBIDDEN_SUFFIX and str(rel) not in PACKAGE_RESOURCES):
            errors.append(f"Unexpected artifact: {rel}")
        if path.suffix == ".py":
            python_count += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
            except SyntaxError:
                errors.append(f"Invalid Python syntax: {rel}")
        if path.suffix == ".json":
            json_count += 1
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix in {".py", ".sh", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"/(?:home|Users)/zds(?:/|\b)", text):
                errors.append(f"Machine-specific path: {rel}")
            if re.search(r"gh[pousr]_" + r"[A-Za-z0-9]{30,}|github_pat_" + r"[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text):
                errors.append(f"Possible credential: {rel}")
    mmseg_files = list((ROOT / "mmseg").rglob("*.py"))
    if len(mmseg_files) != manifest["mmseg_python_files"]:
        errors.append("MMSegmentation source count mismatch")
    for name in manifest["model_configs"]:
        tree = ast.parse((ROOT / name).read_text())
        forbidden = {"train_cfg", "train_pipeline", "train_dataloader", "optim_wrapper", "param_scheduler", "loss_decode", "class_weight", "loss_weight"}
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in forbidden:
                errors.append(f"Unexpected model configuration field: {name}:{node.arg}")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in forbidden:
                errors.append(f"Unexpected model configuration field: {name}:{node.id}")
    print(json.dumps({"status": "FAIL" if errors else "PASS", "files_checked": file_count,
                      "python_files_checked": python_count, "json_files_checked": json_count,
                      "mmseg_python_files": len(mmseg_files), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

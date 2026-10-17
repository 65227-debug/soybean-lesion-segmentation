#!/usr/bin/env python3
"""Package only supplementary results, manifests, configs, and source code."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "paper_materials/投稿前补充实验_20260829"
ARCHIVE = ROOT / "paper_materials/投稿前补充实验_20260829_结果与代码_不含模型.tar.gz"
SHA_FILE = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
INCLUDE = [
    RESULT_DIR,
    ROOT / "paper_code/submission_supplement_experiments",
    ROOT / "data/dadou_label_fraction_subsets",
    ROOT / "paper_code/ablation_abdec/train_ablation_student.py",
    ROOT / "paper_code/corrected_f_training/train_corrected_f.py",
    ROOT / "paper_code/corrected_f_training/train_distilled_student.py",
    ROOT / "paper_code/strong_ssl_baselines/train_stpp.py",
]


def allowed(path: Path) -> bool:
    return (
        "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pth", ".pt", ".ckpt"}
        and path != ARCHIVE
        and path != SHA_FILE
    )


def iter_files():
    seen = set()
    for item in INCLUDE:
        candidates = item.rglob("*") if item.is_dir() else [item]
        for path in candidates:
            if path.is_file() and allowed(path):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def main():
    files = sorted(iter_files(), key=lambda p: str(p.relative_to(ROOT)))
    manifest = {
        "archive_excludes_model_weights_and_checkpoints": True,
        "file_count": len(files),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    manifest_path = RESULT_DIR / "打包内容清单.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(iter_files(), key=lambda p: str(p.relative_to(ROOT)))
    manifest["file_count"] = len(files)
    manifest["files"] = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
        for path in files
    ]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(iter_files(), key=lambda p: str(p.relative_to(ROOT)))
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    SHA_FILE.write_text(f"{digest}  {ARCHIVE.name}\n", encoding="utf-8")
    print(json.dumps({
        "archive": str(ARCHIVE),
        "bytes": ARCHIVE.stat().st_size,
        "sha256": digest,
        "sha256_file": str(SHA_FILE),
        "file_count": len(files),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

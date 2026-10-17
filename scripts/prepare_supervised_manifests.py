#!/usr/bin/env python3
"""Prepare only GT student manifests, independently of historical F pseudo labels."""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/dadou_mmseg")
    parser.add_argument("--output-root", default="data/dadou_student")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    source_root, output_root = Path(args.source_root), Path(args.output_root)
    # Fail before writing if any original student output already exists.
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {output_root}")
    for split in ("train", "val", "test"):
        if not (source_root / split / "images").is_dir() or not (source_root / split / "masks").is_dir():
            raise FileNotFoundError(f"Missing converted {split}/images or {split}/masks in {source_root}")
    spec = importlib.util.spec_from_file_location(
        "student_preparation", ROOT / "paper_code/step9_student_training/prepare_student_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"image_size": args.image_size, "supervised": {}, "pseudo_preparation": "not performed"}
    for split in ("train", "val", "test"):
        rows, pixels = module.prepare_supervised_split(source_root, output_root, split, args.image_size)
        if not rows:
            raise ValueError(f"No supervised images in split {split}")
        module.write_csv(output_root / f"supervised_{split}_manifest.csv", rows)
        summary["supervised"][split] = {"images": len(rows), "pixel_counts": dict(pixels)}
    (output_root / "supervised_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproducible SHA-256 and scene-disjointness audit for the frozen protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_materials/投稿前补充实验_20260829/数据哈希去重审计.json"


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    metadata = {
        Path(r["annotation_filename"]).stem: r
        for r in rows(ROOT / "data/dadou/metadata/精标总清单.csv")
    }
    split_rows = {
        split: rows(ROOT / f"data/dadou_student/supervised_{split}_manifest.csv")
        for split in ("train", "val", "test")
    }
    weak_rows = rows(ROOT / "data/dadou_abdec/pseudo_B_manifest.csv")

    hashes = defaultdict(list)
    for split, records in split_rows.items():
        for record in records:
            path = Path(record["image_path"])
            hashes[sha256(path)].append({
                "pool": "supervised", "split": split,
                "sample_id": record["sample_id"], "path": str(path),
            })
    for record in weak_rows:
        path = Path(record["image_path"])
        hashes[sha256(path)].append({
            "pool": "weak", "split": "unlabeled",
            "sample_id": record["sample_id"], "path": str(path),
        })
    duplicates = [value for value in hashes.values() if len(value) > 1]
    cross_pool = [
        value for value in duplicates
        if {r["pool"] for r in value} == {"supervised", "weak"}
    ]

    split_scenes = {}
    for split, records in split_rows.items():
        split_scenes[split] = {
            metadata[r["sample_id"]]["scene_cluster_id"] for r in records
        }
    scene_overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        scene_overlaps[f"{left}-{right}"] = sorted(split_scenes[left] & split_scenes[right])

    fraction_sets = {}
    for fraction in (25, 50, 75, 100):
        path = ROOT / f"data/dadou_label_fraction_subsets/fraction_{fraction:03d}/supervised_train_manifest.csv"
        fraction_sets[fraction] = {r["sample_id"] for r in rows(path)}
    nested = all(
        fraction_sets[left].issubset(fraction_sets[right])
        for left, right in ((25, 50), (50, 75), (75, 100))
    )
    result = {
        "supervised_images": sum(len(v) for v in split_rows.values()),
        "supervised_split_images": {k: len(v) for k, v in split_rows.items()},
        "supervised_split_scenes": {k: len(v) for k, v in split_scenes.items()},
        "weak_images": len(weak_rows),
        "unique_image_sha256": len(hashes),
        "duplicate_hash_groups_all": len(duplicates),
        "supervised_weak_cross_pool_duplicate_hash_groups": len(cross_pool),
        "scene_cluster_overlap": scene_overlaps,
        "label_fraction_images": {str(k): len(v) for k, v in fraction_sets.items()},
        "label_fraction_subsets_are_nested": nested,
        "status": "PASS" if not duplicates and not any(scene_overlaps.values()) and nested else "FAIL",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build deterministic, nested, scene-disjoint labeled-data fractions.

The validation/test manifests and the 1,390-image weak pool are not changed.
Training scenes are assigned as indivisible groups and stratified by disease
class.  A small dynamic program finds class-wise nested subsets closest to the
requested 25/50/75% image counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


FRACTIONS = (25, 50, 75, 100)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_nested_scene_tiers(scene_sizes, target_counts, rng):
    """Return scene->first included fraction index (0,1,2,3)."""
    items = list(scene_sizes.items())
    rng.shuffle(items)
    # State values are tuples of the tier decisions made so far.
    states = {(0, 0, 0): ()}
    max_size = max(scene_sizes.values())
    limits = tuple(t + max_size for t in target_counts)
    for _, size in items:
        updated = {}
        for counts, decisions in states.items():
            candidates = (
                ((counts[0] + size, counts[1] + size, counts[2] + size), 0),
                ((counts[0], counts[1] + size, counts[2] + size), 1),
                ((counts[0], counts[1], counts[2] + size), 2),
                (counts, 3),
            )
            for new_counts, tier in candidates:
                if any(v > lim for v, lim in zip(new_counts, limits)):
                    continue
                if new_counts not in updated:
                    updated[new_counts] = decisions + (tier,)
        states = updated
    target = tuple(target_counts)
    best_counts = min(
        states,
        key=lambda c: (
            sum((a - b) ** 2 for a, b in zip(c, target)),
            sum(abs(a - b) for a, b in zip(c, target)),
            c,
        ),
    )
    return {
        scene: tier for (scene, _), tier in zip(items, states[best_counts])
    }, best_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/dadou_student")
    parser.add_argument(
        "--metadata", default="data/dadou/metadata/精标总清单.csv")
    parser.add_argument(
        "--output-root", default="data/dadou_label_fraction_subsets")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    train_rows, train_fields = read_csv(source_root / "supervised_train_manifest.csv")
    metadata_rows, _ = read_csv(Path(args.metadata))
    metadata = {
        Path(row["annotation_filename"]).stem: row for row in metadata_rows
    }

    missing = [r["sample_id"] for r in train_rows if r["sample_id"] not in metadata]
    if missing:
        raise RuntimeError(f"Missing metadata for {len(missing)} samples: {missing[:5]}")

    grouped = defaultdict(lambda: defaultdict(list))
    for row in train_rows:
        meta = metadata[row["sample_id"]]
        grouped[meta["disease_class"]][meta["scene_cluster_id"]].append(row)

    selected_ids = {fraction: set() for fraction in FRACTIONS}
    audit_rows = []
    class_summary = {}
    for class_index, class_name in enumerate(sorted(grouped)):
        scenes = grouped[class_name]
        total = sum(len(rows) for rows in scenes.values())
        targets = tuple(int(total * fraction / 100 + 0.5) for fraction in (25, 50, 75))
        tiers, achieved = select_nested_scene_tiers(
            {scene: len(rows) for scene, rows in scenes.items()},
            targets,
            random.Random(args.seed + class_index * 1009),
        )
        for scene, rows in scenes.items():
            tier = tiers[scene]
            first_fraction = FRACTIONS[tier]
            for fraction_index, fraction in enumerate(FRACTIONS):
                if fraction_index >= tier:
                    selected_ids[fraction].update(r["sample_id"] for r in rows)
            for row in rows:
                audit_rows.append({
                    "sample_id": row["sample_id"],
                    "disease_class": class_name,
                    "scene_id": metadata[row["sample_id"]]["scene_id"],
                    "scene_cluster_id": scene,
                    "first_included_fraction": first_fraction,
                })
        class_summary[class_name] = {
            "total_images": total,
            "total_scenes": len(scenes),
            "target_images_25_50_75": list(targets),
            "actual_images_25_50_75": list(achieved),
        }

    previous = set()
    fraction_summary = {}
    for fraction in FRACTIONS:
        ids = selected_ids[fraction]
        if not previous.issubset(ids):
            raise RuntimeError(f"Fraction {fraction} is not nested")
        previous = ids
        rows = [r for r in train_rows if r["sample_id"] in ids]
        out = output_root / f"fraction_{fraction:03d}"
        write_csv(out / "supervised_train_manifest.csv", rows, train_fields)
        for split in ("val", "test"):
            split_rows, split_fields = read_csv(
                source_root / f"supervised_{split}_manifest.csv")
            write_csv(
                out / f"supervised_{split}_manifest.csv", split_rows, split_fields)
        counts = Counter(metadata[r["sample_id"]]["disease_class"] for r in rows)
        scenes = {
            metadata[r["sample_id"]]["scene_cluster_id"] for r in rows
        }
        fraction_summary[str(fraction)] = {
            "images": len(rows),
            "scenes": len(scenes),
            "class_images": dict(sorted(counts.items())),
            "manifest": str(out / "supervised_train_manifest.csv"),
        }

    write_csv(
        output_root / "scene_stratified_assignment_audit.csv",
        sorted(audit_rows, key=lambda r: r["sample_id"]),
        ["sample_id", "disease_class", "scene_id", "scene_cluster_id",
         "first_included_fraction"],
    )
    summary = {
        "protocol": "nested scene-cluster-stratified labeled fractions",
        "seed_fixed_before_training": args.seed,
        "source_train_images": len(train_rows),
        "validation_and_test_manifests": "unchanged byte-equivalent rows",
        "weak_pool": "unchanged data/dadou_student_corrected_f (1390 input; 1354 retained)",
        "class_summary": class_summary,
        "fraction_summary": fraction_summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "fraction_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

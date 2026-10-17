#!/usr/bin/env python3
"""Repair only effective ST++ protocol metadata; never touch checkpoints."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = json.loads(
    (ROOT / "paper_code/strong_ssl_baselines/aligned_protocol.json").read_text(
        encoding="utf-8"))

for seed in (20260804, 20260805, 20260806, 20260807):
    work_dir = ROOT / "work_dirs" / f"submission_seed_{seed}" / "STPP"
    summary = json.loads((work_dir / "training_summary.json").read_text(encoding="utf-8"))
    if summary["seed"] != seed:
        raise RuntimeError(f"Training summary seed mismatch in {work_dir}")
    effective = dict(TEMPLATE)
    effective["seed"] = seed
    (work_dir / "aligned_protocol.json").write_text(
        json.dumps(effective, ensure_ascii=False, indent=2), encoding="utf-8")
    print(seed, "metadata repaired; checkpoints unchanged")

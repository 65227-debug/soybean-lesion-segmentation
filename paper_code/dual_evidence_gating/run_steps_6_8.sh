#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" paper_code/dual_evidence_gating/prepare_validation_manifest.py

"${PYTHON_BIN}" tools/dadou/generate_three_base_maps.py \
  --manifest data/dadou/06_验证集基础图/validation_manifest.csv \
  --output-dir data/dadou/06_验证集基础图 \
  --preview-limit 20

"${PYTHON_BIN}" paper_code/dual_evidence_gating/tune_thresholds.py
"${PYTHON_BIN}" paper_code/dual_evidence_gating/tune_image_filter.py
"${PYTHON_BIN}" paper_code/dual_evidence_gating/generate_gated_pseudo_labels.py
"${PYTHON_BIN}" paper_code/dual_evidence_gating/audit_outputs.py

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" tools/dadou/generate_three_base_maps.py \
  --manifest data/dadou/06_验证集基础图/validation_manifest.csv \
  --output-dir data/dadou/06_验证集基础图_修正版 \
  --preview-limit 20

"$PYTHON_BIN" paper_code/corrected_f_training/tune_corrected_gating.py \
  --base-maps data/dadou/06_验证集基础图_修正版 \
  --output-dir data/dadou/06_验证集门控调参_修正版 \
  --selected-config paper_code/corrected_f_training/selected_gating_config.json

"$PYTHON_BIN" tools/dadou/generate_three_base_maps.py \
  --manifest data/dadou_semantic/train_manifest.csv \
  --output-dir data/dadou/05_三张基础图_修正版 \
  --preview-limit 24

"$PYTHON_BIN" paper_code/corrected_f_training/generate_corrected_pseudo_labels.py \
  --base-maps data/dadou/05_三张基础图_修正版 \
  --config paper_code/corrected_f_training/selected_gating_config.json \
  --output-dir data/dadou_student_corrected_f

"$PYTHON_BIN" paper_code/corrected_f_training/train_corrected_f.py \
  --pseudo-root data/dadou_student_corrected_f \
  --work-dir work_dirs/dadou_corrected_F


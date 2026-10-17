#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"
SEG_PYTHON="${SEG_PYTHON:-python}"

cd "$PROJECT_ROOT"
"$SEG_PYTHON" paper_code/q1_upgrade_experiments/analyze_severity_boundary.py
"$SEG_PYTHON" paper_code/q1_upgrade_experiments/plot_pseudo_quality_tradeoff.py
"$SEG_PYTHON" paper_code/q1_upgrade_experiments/benchmark_efficiency.py
DISTILLED_TEST_DIR="paper_materials/一区提升实验补充_20260826/04_门控蒸馏校准逐项消融/G_门控加蒸馏_冻结测试"
if [[ ! -f "$DISTILLED_TEST_DIR/test_summary.json" ]]; then
  "$SEG_PYTHON" paper_code/corrected_f_training/evaluate_frozen_student.py \
    --checkpoint work_dirs/dadou_corrected_F_distilled/best_validation_mIoU.pth \
    --output-dir "$DISTILLED_TEST_DIR" \
    --workers 4 \
    --batch-size 4
fi
"$SEG_PYTHON" paper_code/q1_upgrade_experiments/analyze_incremental_stages.py

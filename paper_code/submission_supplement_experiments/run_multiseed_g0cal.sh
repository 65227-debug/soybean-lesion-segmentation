#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260804 20260805 20260806 20260807)

# The grid is frozen from the original seed-20260803 G0-Cal experiment.
# Model selection uses validation mIoU only; the test set is evaluated once.
for SEED in "${SEEDS[@]}"; do
  BASE_DIR="work_dirs/submission_seed_${SEED}"
  A_TEST_DIR="$BASE_DIR/A_test"
  G0_TEST_DIR="$BASE_DIR/G0_test"
  CAL_DIR="$BASE_DIR/G0_calibrated"
  TEST_DIR="$BASE_DIR/G0_calibrated_test"
  G1_TEST_DIR="$BASE_DIR/G1_test"
  mkdir -p "$A_TEST_DIR" "$G0_TEST_DIR" "$CAL_DIR" "$TEST_DIR" "$G1_TEST_DIR"

  if [[ ! -f "$A_TEST_DIR/test_summary.json" ]]; then
    echo "[$SEED] frozen supervised-A test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$BASE_DIR/A/best_validation_mIoU.pth" \
      --output-dir "$A_TEST_DIR" --workers 4 \
      > "$A_TEST_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G0_TEST_DIR/test_summary.json" ]]; then
    echo "[$SEED] frozen uncalibrated G0 test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$BASE_DIR/G0/best_validation_mIoU.pth" \
      --output-dir "$G0_TEST_DIR" --workers 4 \
      > "$G0_TEST_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$CAL_DIR/calibration_summary.json" ]]; then
    echo "[$SEED] validation-only G0 calibration"
    "$PYTHON_BIN" paper_code/corrected_f_training/calibrate_single_student.py \
      --checkpoint "$BASE_DIR/G0/best_validation_mIoU.pth" \
      --output-dir "$CAL_DIR" \
      --healthy-biases=-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4 \
      --lesion-biases=-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2 \
      --selection-metric mIoU --workers 4 \
      > "$CAL_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$TEST_DIR/test_summary.json" ]]; then
    echo "[$SEED] frozen G0-Cal test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$CAL_DIR/best_calibrated_student.pth" \
      --output-dir "$TEST_DIR" --workers 4 \
      > "$TEST_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G1_TEST_DIR/test_summary.json" ]]; then
    echo "[$SEED] frozen uncalibrated G1 test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$BASE_DIR/G1_distilled/best_validation_mIoU.pth" \
      --output-dir "$G1_TEST_DIR" --workers 4 \
      > "$G1_TEST_DIR/run.log" 2>&1
  fi
done

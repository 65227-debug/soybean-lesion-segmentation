#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260804 20260805 20260806 20260807)

for SEED in "${SEEDS[@]}"; do
  A_DIR="work_dirs/submission_seed_${SEED}/A"
  G0_DIR="work_dirs/submission_seed_${SEED}/G0"
  G1_DIR="work_dirs/submission_seed_${SEED}/G1_distilled"
  G_CAL_DIR="work_dirs/submission_seed_${SEED}/G_final_calibrated"
  G_TEST_DIR="work_dirs/submission_seed_${SEED}/G_final_test"
  STPP_DIR="work_dirs/submission_seed_${SEED}/STPP"
  mkdir -p "$A_DIR" "$G0_DIR" "$G1_DIR" "$G_CAL_DIR" "$G_TEST_DIR" "$STPP_DIR"

  if [[ ! -f "$A_DIR/training_summary.json" ]]; then
    echo "[$SEED] train A"
    "$PYTHON_BIN" paper_code/ablation_abdec/train_ablation_student.py \
      --mode A --seed "$SEED" --workers 4 --work-dir "$A_DIR" \
      > "$A_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G0_DIR/training_summary.json" ]]; then
    echo "[$SEED] train G0"
    "$PYTHON_BIN" paper_code/corrected_f_training/train_corrected_f.py \
      --pseudo-root data/dadou_student_corrected_f \
      --seed "$SEED" --workers 4 --skip-test --work-dir "$G0_DIR" \
      > "$G0_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G1_DIR/training_summary.json" ]]; then
    echo "[$SEED] distil G1"
    "$PYTHON_BIN" paper_code/corrected_f_training/train_distilled_student.py \
      --checkpoint-a "$A_DIR/best_validation_mIoU.pth" \
      --checkpoint-f "$G0_DIR/best_validation_mIoU.pth" \
      --seed "$SEED" --work-dir "$G1_DIR" \
      > "$G1_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G_CAL_DIR/calibration_summary.json" ]]; then
    echo "[$SEED] validation-only calibration"
    "$PYTHON_BIN" paper_code/corrected_f_training/calibrate_single_student.py \
      --checkpoint "$G1_DIR/best_validation_mIoU.pth" \
      --output-dir "$G_CAL_DIR" \
      --healthy-biases=-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0 \
      --lesion-biases=0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
      --selection-metric mIoU --workers 4 \
      > "$G_CAL_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G_TEST_DIR/test_summary.json" ]]; then
    echo "[$SEED] frozen G test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$G_CAL_DIR/best_calibrated_student.pth" \
      --output-dir "$G_TEST_DIR" --workers 4 \
      > "$G_TEST_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$STPP_DIR/training_summary.json" ]]; then
    echo "[$SEED] train ST++"
    "$PYTHON_BIN" paper_code/strong_ssl_baselines/train_stpp.py \
      --seed "$SEED" --workers 4 --work-dir "$STPP_DIR" \
      > "$STPP_DIR/run.log" 2>&1
  fi
  echo "[$SEED] complete"
done

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=20260803
FRACTIONS=(25 50 75)

"$PYTHON_BIN" paper_code/submission_supplement_experiments/build_scene_stratified_label_fractions.py

for FRACTION in "${FRACTIONS[@]}"; do
  TAG=$(printf "%03d" "$FRACTION")
  SUP_ROOT="data/dadou_label_fraction_subsets/fraction_${TAG}"
  BASE_DIR="work_dirs/label_fraction_${TAG}_seed${SEED}"
  A_DIR="$BASE_DIR/A"
  A_TEST_DIR="$BASE_DIR/A_test"
  G0_DIR="$BASE_DIR/G0"
  G1_DIR="$BASE_DIR/G1_distilled"
  G_CAL_DIR="$BASE_DIR/G_final_calibrated"
  G_TEST_DIR="$BASE_DIR/G_final_test"
  mkdir -p "$A_DIR" "$A_TEST_DIR" "$G0_DIR" "$G1_DIR" "$G_CAL_DIR" "$G_TEST_DIR"

  if [[ ! -f "$A_DIR/training_summary.json" ]]; then
    echo "[$TAG%] train supervised A"
    "$PYTHON_BIN" paper_code/ablation_abdec/train_ablation_student.py \
      --mode A --supervised-root "$SUP_ROOT" --seed "$SEED" --workers 4 \
      --work-dir "$A_DIR" > "$A_DIR/run.log" 2>&1
  fi
  if [[ ! -f "$A_TEST_DIR/test_summary.json" ]]; then
    echo "[$TAG%] frozen A test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$A_DIR/best_validation_mIoU.pth" --data-root "$SUP_ROOT" \
      --output-dir "$A_TEST_DIR" --workers 4 > "$A_TEST_DIR/run.log" 2>&1
  fi

  if [[ ! -f "$G0_DIR/training_summary.json" ]]; then
    echo "[$TAG%] train G0"
    "$PYTHON_BIN" paper_code/corrected_f_training/train_corrected_f.py \
      --supervised-root "$SUP_ROOT" --pseudo-root data/dadou_student_corrected_f \
      --seed "$SEED" --workers 4 --skip-test --work-dir "$G0_DIR" \
      > "$G0_DIR/run.log" 2>&1
  fi
  if [[ ! -f "$G1_DIR/training_summary.json" ]]; then
    echo "[$TAG%] distil G1"
    "$PYTHON_BIN" paper_code/corrected_f_training/train_distilled_student.py \
      --checkpoint-a "$A_DIR/best_validation_mIoU.pth" \
      --checkpoint-f "$G0_DIR/best_validation_mIoU.pth" \
      --supervised-root "$SUP_ROOT" --pseudo-root data/dadou_student_corrected_f \
      --seed "$SEED" --work-dir "$G1_DIR" > "$G1_DIR/run.log" 2>&1
  fi
  if [[ ! -f "$G_CAL_DIR/calibration_summary.json" ]]; then
    echo "[$TAG%] validation-only final-G calibration"
    "$PYTHON_BIN" paper_code/corrected_f_training/calibrate_single_student.py \
      --checkpoint "$G1_DIR/best_validation_mIoU.pth" \
      --supervised-root "$SUP_ROOT" --output-dir "$G_CAL_DIR" \
      --healthy-biases=-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0 \
      --lesion-biases=0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
      --selection-metric mIoU --workers 4 > "$G_CAL_DIR/run.log" 2>&1
  fi
  if [[ ! -f "$G_TEST_DIR/test_summary.json" ]]; then
    echo "[$TAG%] frozen G test"
    "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
      --checkpoint "$G_CAL_DIR/best_calibrated_student.pth" \
      --data-root "$SUP_ROOT" --output-dir "$G_TEST_DIR" --workers 4 \
      > "$G_TEST_DIR/run.log" 2>&1
  fi
done

# Complete the 100% supervised endpoint without retraining the existing seed.
FULL_A_TEST="work_dirs/label_fraction_100_seed${SEED}/A_test"
mkdir -p "$FULL_A_TEST"
if [[ ! -f "$FULL_A_TEST/test_summary.json" ]]; then
  "$PYTHON_BIN" paper_code/corrected_f_training/evaluate_frozen_student.py \
    --checkpoint work_dirs/dadou_ablation_A/best_validation_mIoU.pth \
    --data-root data/dadou_student --output-dir "$FULL_A_TEST" --workers 4 \
    > "$FULL_A_TEST/run.log" 2>&1
fi

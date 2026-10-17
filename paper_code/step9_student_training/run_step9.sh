#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-work_dirs/dadou_student_deeplabv3plus_mobilenetv3_full60}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" \
  paper_code/step9_student_training/prepare_student_dataset.py

"${PYTHON_BIN}" \
  paper_code/step9_student_training/train_student.py \
  --work-dir "${RUN_DIR}"

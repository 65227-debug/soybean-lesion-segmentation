#!/usr/bin/env bash
set -euo pipefail

SEG_PYTHON="${SEG_PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$SEG_PYTHON" paper_code/ablation_abdec/prepare_ablation_pseudo_labels.py

"$SEG_PYTHON" paper_code/ablation_abdec/train_ablation_student.py \
  --mode A --workers 4 --work-dir work_dirs/dadou_ablation_A
"$SEG_PYTHON" paper_code/ablation_abdec/train_ablation_student.py \
  --mode B --workers 4 --work-dir work_dirs/dadou_ablation_B
"$SEG_PYTHON" paper_code/ablation_abdec/train_ablation_student.py \
  --mode D --workers 4 --work-dir work_dirs/dadou_ablation_D
"$SEG_PYTHON" paper_code/ablation_abdec/train_ablation_student.py \
  --mode E --workers 4 --work-dir work_dirs/dadou_ablation_E
"$SEG_PYTHON" paper_code/ablation_abdec/train_mean_teacher.py \
  --workers 4 --work-dir work_dirs/dadou_ablation_C

"$SEG_PYTHON" paper_code/ablation_abdec/evaluate_all_groups.py
"$SEG_PYTHON" paper_code/ablation_abdec/paired_bootstrap.py

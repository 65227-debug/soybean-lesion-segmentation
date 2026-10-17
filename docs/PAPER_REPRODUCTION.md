# Final G reproduction

Run from the repository root after following `PAPER_INSTALLATION.md` and `PAPER_DATA.md`. Original data and trained checkpoints are not included. The following are command recipes, not commands that have been rerun to build this code release.

## 1. Teacher, independent leaf model, independent semantic model

```bash
python tools/dadou/prepare_dadou_dataset.py
python tools/train.py configs/dadou/deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py
python tools/train.py configs/dadou/unet-s5-d16_fcn_1xb4-60e_dadou-leaf-512x512.py
python tools/dadou_semantic/prepare_semantic_dataset.py --verify-sha256
python tools/dadou_semantic/train_efficientnet_layercam.py \
  --work-dir work_dirs/dadou_efficientnet_b0_semantic
```

The teacher and leaf work directories retain the best validation checkpoints and epoch-level metrics. Obtain the teacher's referenced pretrained checkpoint separately. The U-Net is independent and trained from scratch. See `configs/dadou/README.md` for inference-checkpoint conversion and curve-export commands. Replace filenames containing an original best epoch with the validation-selected checkpoint from your run.

## 2. Corrected three evidence maps and frozen gate

```bash
python tools/dadou/generate_three_base_maps.py \
  --manifest data/dadou_semantic/train_manifest.csv \
  --teacher-config configs/dadou/deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py \
  --teacher-checkpoint work_dirs/dadou_deeplabv3plus_r50_teacher/best_mIoU_epoch_23_inference.pth \
  --leaf-config configs/dadou/unet-s5-d16_fcn_1xb4-60e_dadou-leaf-512x512.py \
  --leaf-checkpoint work_dirs/dadou_unet_leaf/best_IoU_leaf_epoch_46_inference.pth \
  --semantic-checkpoint work_dirs/dadou_efficientnet_b0_semantic/best_macro_f1.pth \
  --output-dir data/dadou/05_三张基础图_修正版

python paper_code/corrected_f_training/generate_corrected_pseudo_labels.py \
  --base-maps data/dadou/05_三张基础图_修正版 \
  --config paper_code/corrected_f_training/selected_gating_config.json \
  --output-dir data/dadou_student_corrected_f
python scripts/prepare_supervised_manifests.py
```

The original best-epoch filenames above are provenance-specific. Use your own validation-frozen checkpoints for a new run. The gate thresholds and weighting equations are loaded from the selected JSON, not copied from a historical default dataclass. No threshold optimization is part of this recipe.

## 3. Final G chain for a single seed

```bash
SEED=20260803
RUN=work_dirs/reproduction_seed_${SEED}
python paper_code/ablation_abdec/train_ablation_student.py \
  --mode A --seed "$SEED" --work-dir "$RUN/A"
python paper_code/corrected_f_training/train_corrected_f.py \
  --pseudo-root data/dadou_student_corrected_f --seed "$SEED" \
  --skip-test --work-dir "$RUN/G0"
python paper_code/corrected_f_training/train_distilled_student.py \
  --checkpoint-a "$RUN/A/best_validation_mIoU.pth" \
  --checkpoint-f "$RUN/G0/best_validation_mIoU.pth" \
  --seed "$SEED" --work-dir "$RUN/G1_distilled"
python paper_code/corrected_f_training/calibrate_single_student.py \
  --checkpoint "$RUN/G1_distilled/best_validation_mIoU.pth" \
  --output-dir "$RUN/G_final_calibrated" \
  --healthy-biases=-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0 \
  --lesion-biases=0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
  --selection-metric mIoU
python paper_code/corrected_f_training/evaluate_frozen_student.py \
  --checkpoint "$RUN/G_final_calibrated/best_calibrated_student.pth" \
  --output-dir "$RUN/G_final_test"
```

A and G0 use 60×98 updates each; G1 uses 18×50. Model selection and logit calibration use validation only. The final checkpoint is one ordinary MobileNetV3 student. Calibration folds class biases into its classifier.

## 4. Baselines and supplementary experiments

The reference seeds are 20260803, 20260804, 20260805, 20260806 and 20260807. The historical multi-seed shell script starts at 20260804 because the original 20260803 already existed; a fresh checkout must also run 20260803 explicitly and provide the source paths expected by the aggregation scripts. Those scripts do not bundle or fabricate missing original results.

```bash
python paper_code/strong_ssl_baselines/train_unimatch.py
python paper_code/strong_ssl_baselines/train_stpp.py --seed 20260803
python paper_code/strong_ssl_baselines/train_u2pl.py
bash paper_code/submission_supplement_experiments/run_multiseed_g_stpp.sh
bash paper_code/submission_supplement_experiments/run_multiseed_g0cal.sh
bash paper_code/submission_supplement_experiments/run_label_fraction_curves.sh
```

Baselines need the full image-only weak manifest described in the data guide. Supplementary aggregators require the outputs at their documented original paths; adapt their path maps if you use a different run root. The historical 100% fraction endpoint reuses `work_dirs/dadou_ablation_A` and the original final-G output, so supply those results or change the endpoint mapping to your freshly trained runs.

UniMatch and U²PL read seed 20260803 from `aligned_protocol.json`; their current CLI does not have a `--seed` flag. For another seed, copy the JSON, change only its seed, and pass that file with `--config`. ST++ has a direct `--seed` override.

Training curves and losses are preserved by the original trainers as `epoch_metrics.csv` or `training_history.csv`. Summaries and per-image test metrics are saved alongside them. `analyze_multiseed_boundary.py` calculates lesion Boundary F1 with a 2-pixel tolerance and component recall with sizes <64, 64–255 and ≥256 pixels; a component is detected when predicted lesion overlap reaches 10% of its area. Keep those definitions attached to the reported numbers.

External scripts whose names contain `repaired` use an explicitly exploratory protocol after observing the external test domain. They must not be reported as a new untouched external test. Existing G3/G4 and other developmental configurations are retained as historical code and are not substituted for this final-G recipe.

## 5. Code-only verification

```bash
python scripts/check_release.py
```

This checks source syntax, JSON readability, protected directories/extensions, frozen gate provenance and a small in-memory synthetic gating case. It does not train a model or require dataset images.

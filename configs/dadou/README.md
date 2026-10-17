# Soybean lesion segmentation teacher

This experiment trains a 512 x 512 DeepLabV3+ model with a ResNet-50
backbone and three output classes: background, healthy leaf tissue, and
lesion.

The preparation script preserves the original LabelMe files. It intersects
lesion polygons with leaf polygons and logs every affected sample in
`data/dadou_mmseg/conversion_manifest.csv`. Samples without a leaf annotation
are excluded. The test split is configured for later use but is not evaluated
during training.

Training augmentation is online and stochastic: scale jitter, crop,
horizontal/vertical flip, rotation, and photometric distortion. A 4x repeat
wrapper presents four independently augmented views of each source image per
epoch without duplicating source files.

Commands (use the existing `seg` conda environment):

```bash
python tools/dadou/prepare_dadou_dataset.py
python tools/train.py \
  configs/dadou/deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py
python tools/dadou/plot_training_curves.py \
  work_dirs/dadou_deeplabv3plus_r50_teacher/epoch_metrics.csv
python tools/dadou/export_inference_checkpoint.py \
  work_dirs/dadou_deeplabv3plus_r50_teacher/best_mIoU_epoch_23.pth
python tools/dadou/infer_teacher.py \
  configs/dadou/deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py \
  work_dirs/dadou_deeplabv3plus_r50_teacher/best_mIoU_epoch_23_inference.pth \
  data/dadou_mmseg/val/images --output-dir outputs/dadou_teacher_val
```

The work directory retains the best checkpoint, the full JSON log,
`epoch_metrics.csv`, and a paper-ready convergence figure.
The inference command exports the three-class mask, a color overlay, and a
16-bit lesion probability PNG (`value / 65535 = P_lesion`) for downstream
dual-evidence gating.

The independent leaf model uses a binary U-Net trained from scratch. Its leaf
target is the union of healthy tissue and lesion pixels. It does not share
weights with the lesion teacher:

```bash
python tools/train.py \
  configs/dadou/unet-s5-d16_fcn_1xb4-60e_dadou-leaf-512x512.py
python tools/dadou/plot_training_curves.py \
  work_dirs/dadou_unet_leaf/epoch_metrics.csv --target-class leaf \
  --selection-metric val_IoU_leaf
python tools/dadou/export_inference_checkpoint.py \
  work_dirs/dadou_unet_leaf/best_IoU_leaf_epoch_46.pth --task leaf
python tools/dadou/infer_leaf.py \
  configs/dadou/unet-s5-d16_fcn_1xb4-60e_dadou-leaf-512x512.py \
  work_dirs/dadou_unet_leaf/best_IoU_leaf_epoch_46_inference.pth \
  data/dadou_mmseg/val/images --output-dir outputs/dadou_unet_leaf_val
python tools/dadou/evaluate_leaf_gate.py \
  --teacher-dir outputs/dadou_teacher_val \
  --leaf-dir outputs/dadou_unet_leaf_val \
  --gt-dir data/dadou_mmseg/val/masks \
  --output-dir outputs/dadou_leaf_gate_ablation
python tools/dadou/evaluate_leaf_gate.py \
  --teacher-dir outputs/dadou_teacher_val \
  --leaf-dir outputs/dadou_unet_leaf_val \
  --gt-dir data/dadou_mmseg/val/masks \
  --leaf-threshold 0.15 \
  --output-dir outputs/dadou_leaf_gate_ablation_val_tuned
```

Freeze any gate threshold on validation before the held-out test split is evaluated.

# Independent soybean disease semantic model

This module trains a two-class EfficientNet-B0 classifier for Frogeye leaf
spot and Target spot, then exports a multi-layer LayerCAM response. It shares
no weights with the lesion teacher or leaf U-Net.

The preparation step is deliberately strict. A weak image is excluded if its
ASDID `scene_id` occurs in any exact train/validation/test row, if its SHA-256
matches a supervised image, or if its 64-bit perceptual hash is within two
bits of a supervised image. Original files are retained and selected files are
represented by symbolic links.

```bash
python \
  tools/dadou_semantic/prepare_semantic_dataset.py --verify-sha256

python \
  tools/dadou_semantic/train_efficientnet_layercam.py \
  --work-dir work_dirs/dadou_efficientnet_b0_semantic

python \
  tools/dadou_semantic/export_multilayer_layercam.py \
  work_dirs/dadou_efficientnet_b0_semantic/best_macro_f1.pth \
  --manifest data/dadou_semantic/val_manifest.csv \
  --output-dir outputs/dadou_semantic_layercam_val

python \
  tools/dadou_semantic/evaluate_layercam_localization.py
```

LayerCAM uses EfficientNet feature stages 3, 5, and 8. Each response is
independently normalized, resized, and averaged. The saved 16-bit maps follow
`response = uint16_pixel / 65535.0`.

Only the 24-image exact validation split is evaluated. The exact test manifest
is retained for provenance but is not linked, loaded, or evaluated.

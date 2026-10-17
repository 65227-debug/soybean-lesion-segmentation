# Data contracts (datasets available separately)

Keep all local data under the ignored `data/` directory. File paths in generated manifests must point to your own machine. Existing manifests from another machine generally need regeneration.

The datasets are available as separate [release downloads](../README.md#dataset-downloads), not Git source files. The Soybean lesion segmentation dataset contains 172 pixel-labeled images and 1,395 image-level/weakly labeled images, totaling 1,567 distinct originals. The separate Soybean original-pairs dataset contains 617 original image/mask pairs. Neither Soybean package assigns a fixed train/validation/test split or includes pseudo-label caches.

## Downloaded data and split preparation

The pixel-labeled data use the following directories, without split subdirectories:

```text
images/
masks/
annotations_json/
```

The original-pairs package contains `images/` and `masks/`. Image-level/weakly labeled images have no pixel masks. Configure your own train/validation/test split before constructing the local training directories and manifests. Keep source images and acquisition scenes disjoint across splits, and check for duplicate or near-duplicate images.

## Pixel-labeled segmentation data

The teacher/student conversion uses the following semantic classes:

| Value | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Visible healthy leaf tissue |
| 2 | Lesion |
| 255 | Ignore, used in pseudo labels |

For LabelMe input, `tools/dadou/prepare_dadou_dataset.py` expects `leaf` and `lesion` polygons. It clips lesions to the union of the leaf polygons. The dataset-specific input subfolder names are defined in that script's `SPLIT_FOLDERS`; prepare those folders locally or adapt the mapping to your own split before conversion. Input folders contain `images/` and `annotations_json/`.

The converted directory must contain:

```text
data/dadou_mmseg/
  train/images/  train/masks/
  val/images/    val/masks/
  test/images/   test/masks/
```

Configure the split sizes and data paths for your own data. The independent leaf model merges labels 1 and 2 as leaf; its merge transform is in `projects/dadou_teacher/dadou_dataset.py`.

Generate the student's aligned 512×512 supervised manifests without loading historical F pseudo labels:

```bash
python scripts/prepare_supervised_manifests.py \
  --source-root data/dadou_mmseg --output-root data/dadou_student
```

It writes `supervised_train_manifest.csv`, `supervised_val_manifest.csv`, and `supervised_test_manifest.csv`. Each row has fields `sample_id,sample_type,split,image_path,mask_path,weight_path,source_image_path`. `weight_path` is empty for ground-truth supervision. The manifests are generated locally and excluded from Git.

## Weak image semantics and leakage checks

`tools/dadou_semantic/prepare_semantic_dataset.py` supports `frogeye` and `target_spot`. Supply your local source images and metadata through its CLI. The input metadata fields used by this pipeline include `role`, `scene_id`, `sha256`, `category_label`, and `filename`. Its protected roles are `exact_train`, `exact_val`, and `exact_test`. Scene overlap, SHA-256 duplicates and near-duplicate pHash matches are checked before selection. This information must come from real data provenance, not fabricated scene identifiers.

The semantic preparation step generates the classification image directories plus `train_manifest.csv` and `val_manifest.csv` under `data/dadou_semantic/`. Its manifests retain local paths and label indices required by the classifier and base-map exporter. Use `--help` and the preparation script for the complete source-specific schema.

## Final G pseudo labels

The corrected base-map generator exports the three-channel teacher probabilities, leaf probabilities and LayerCAM arrays, plus `三张基础图生成清单.csv`. Generate the **corrected** evidence set with `--output-dir data/dadou/05_三张基础图_修正版` and explicit teacher/leaf configuration paths (see the reproduction guide). The default historical base-map directory is not the final G directory.

`generate_corrected_pseudo_labels.py` reads that corrected evidence and the frozen selected gate. It writes images, uint8 class masks, uint16 continuous weights and `pseudo_train_manifest.csv` under `data/dadou_student_corrected_f/`. Its manifest columns are `sample_id,sample_type,split,class_name,scene_id,image_path,mask_path,weight_path,source_image_path`. Weight decoding is `uint16 / 65535.0`.

## Strong baselines and label-fraction experiments

The baseline configuration uses `data/dadou_abdec/pseudo_B_manifest.csv` as an image-only unlabeled input list. UniMatch/ST++/U²PL read only `sample_id`, `image_path`, and optional `source_image_path`; they discard weak semantics, masks and weight columns. If starting directly from the weak pool, supply an image-only manifest through a copied aligned configuration and document that change. Include all weak inputs, not only the gate-accepted subset.

The label-fraction builder additionally needs local metadata with `annotation_filename`, `disease_class`, `scene_id`, and `scene_cluster_id`. It keeps each scene cluster indivisible and creates nested, class-stratified subsets. It does not reconstruct missing provenance from filenames.

Neither this code release nor its ZIP includes any of the real CSVs, image hashes, sample IDs or annotations described above.

# Images and manual pixel labels

## Soybean manual annotations v2.1.0

The [manual annotation package](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-soybean-manual-v2.1.0) contains 172 original soybean images, their manual LabelMe polygon annotations and corresponding pixel masks. Matching images, masks and annotation files share the same sample ID.

```text
images/
masks/
annotations_json/
metadata_samples.csv
```

The included README describes the polygon-to-mask conversion. `metadata_samples.csv` links each image to its annotation and mask and records its source name and original filename. Source records comprise 94 ASDID images and 78 images from the annotation project's existing soybean collection. ASDID is available at https://doi.org/10.5061/dryad.41ns1rnj3.

| Value | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Healthy leaf tissue |
| 2 | Lesion |

The binary leaf model merges healthy leaf tissue and lesions into the foreground leaf class.

## MH-SoyaHealthVision subset v1.1.0

The [MH-SoyaHealthVision package](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-mh-soyahealthvision-v1.1.0) contains 160 selected original images, 160 aligned 512 × 512 images, 80 manual LabelMe annotations and 80 corresponding manual pixel masks. The manually annotated and unlabelled groups contain 80 separate images each, with 20 images per disease category in each group.

The package retains its own directory layout. Its README, manifests and file inventory provide image-to-annotation mappings and checksums. Pixel labels use the same 0/1/2 encoding. Source dataset: MH-SoyaHealthVision, Mendeley Data, version 1, https://doi.org/10.17632/hkbgh5s3b7.1. Source images are licensed under CC BY 4.0; see the included source notice for attribution.

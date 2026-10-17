# Soybean lesion segmentation: dual-evidence semi-supervised learning

Research code for **semi-supervised soybean lesion segmentation with leaf spatial constraints and disease-semantic soft evidence**.

This repository contains the complete local MMSegmentation **1.2.2 Python package**, model configurations, custom dataset/metric hooks, and experiment scripts. Datasets, annotations, dataset manifests, model weights, training outputs, and paper drafts are not distributed. Provide your own permitted data and obtain the referenced pretrained weights separately.

中文说明见 [README_zh-CN.md](README_zh-CN.md).

## Models and labels

| Component | Model | Role |
| --- | --- | --- |
| Lesion teacher | DeepLabV3+ / ResNet-50 | Background / healthy leaf / lesion probabilities |
| Independent leaf model | U-Net | Leaf probability and trustworthy interior |
| Disease-semantic model | EfficientNet-B0 + multi-layer LayerCAM | Frogeye / target-spot image semantics |
| Student | DeepLabV3+ / MobileNetV3-Large | Three-class lesion segmentation at 512×512 |

Pixel labels are `0=background`, `1=healthy_leaf`, `2=lesion`, `255=ignore`. Pseudo-label weights are float values in `[0,1]`; ignore pixels have zero weight. See the frozen configuration in `paper_code/corrected_f_training/selected_gating_config.json`.

## Start here

1. [Installation and recorded environment](docs/PAPER_INSTALLATION.md).
2. [Data contracts and preparation](docs/PAPER_DATA.md).
3. [Final G reproduction and supplementary experiments](docs/PAPER_REPRODUCTION.md).
4. [Code map and third-party attribution](THIRD_PARTY_NOTICES.md).

The paper's final G chain is **A supervised student → G0 corrected gate training → G1 distillation → validation-only class-logit calibration → frozen test**. The publication entry point is the frozen plan in `paper_code/submission_supplement_experiments/frozen_experiment_plan.json`. Earlier F/G3/G4 trials remain in the code as historical development variants; they are not interchangeable with the final G.

## Code map

| Directory | Contents |
| --- | --- |
| `mmseg/`, `configs/`, `tools/`, `projects/` | MMSegmentation package and project extensions |
| `configs/dadou/`, `projects/dadou_teacher/`, `tools/dadou/` | Lesion teacher, independent leaf U-Net, probability export |
| `tools/dadou_semantic/` | EfficientNet-B0, scene/duplicate exclusion, LayerCAM |
| `paper_code/corrected_f_training/` | Corrected gate, continuous weights, G0/G1/calibration/frozen evaluation |
| `paper_code/step9_student_training/` | Student model and shared training utilities |
| `paper_code/ablation_abdec/` | Supervised A and ablations |
| `paper_code/strong_ssl_baselines/` | Protocol-aligned UniMatch, ST++ and U²PL adaptations |
| `paper_code/submission_supplement_experiments/` | Five seeds, label-fraction curves, confidence intervals and integrity audits |
| `paper_code/external_test/`, `paper_code/external_validation/` | External-data preparation and evaluation, including explicitly marked exploratory variants |

## Validation and interpretation

The study used 108 labeled training, 27 validation, 37 internal-test and 1,390 weakly labeled input images. The corrected gate retained 1,354 weak images. The fixed five-seed analysis found final G at 83.57±0.63% mIoU and ST++ at 83.92±0.51%; G−ST++ was −0.35 pp (95% CI −1.34 to 0.64). G improved lesion Boundary F1 and small-component recall under the documented definitions. These are recorded study results, not expected scores on arbitrary replacement data.

Final G and the baselines share the student architecture and data split. Final G uses more total training updates across its multi-stage chain; do not describe the full comparison as equal total training compute. The strong baselines are independent **protocol-aligned adaptations**, not exact reproductions of the original VOC/Cityscapes headline results.

This code-only release has syntax, configuration, synthetic gating, model-forward and file-inclusion checks. It does not redistribute the original private data, so exact numerical reproduction requires the original authorized splits and checkpoints produced by training.

## Attribution and citation

Retain the [Apache-2.0 license](LICENSE) and [MMSegmentation citation](CITATION.cff). MMSegmentation is developed by OpenMMLab and its contributors. The soybean project extensions and paper-specific scripts are identified above. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for method sources and pinned reference commits. Paper authors, DOI and final publication metadata have not been invented; add those when finalized.

To cite this code in a manuscript, use the public repository URL and the exact Git commit or release tag, in addition to the published paper and upstream software/method references.

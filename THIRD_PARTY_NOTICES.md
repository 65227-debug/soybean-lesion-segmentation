# Third-party source and method attribution

## OpenMMLab MMSegmentation

This code release vendors the local MMSegmentation 1.2.2 source tree used by the experiments. Upstream: <https://github.com/open-mmlab/mmsegmentation>. The original Apache-2.0 license, copyright headers, `CITATION.cff`, model/config citations and text documentation are retained. Original top-level README files are preserved in `docs/upstream/`. The source project was not a Git checkout, so an exact upstream base commit is not asserted.

Soybean-specific extensions are located in `configs/dadou/`, `projects/dadou_teacher/`, `tools/dadou/`, `tools/dadou_semantic/`, and `paper_code/`. Mechanical local-path substitutions are listed in `CODE_RELEASE_MANIFEST.json`. This release uses the repository's Apache-2.0 license; third-party notices remain attached to their respective source files and references.

## Semi-supervised method references

The implementations in `paper_code/strong_ssl_baselines/` are independent protocol-aligned adaptations. Their consulted reference repositories and pinned commits are recorded verbatim in `paper_code/strong_ssl_baselines/OFFICIAL_REFERENCES.json`:

- UniMatch: <https://github.com/LiheYoung/UniMatch>, MIT.
- ST++: <https://github.com/LiheYoung/ST-PlusPlus>, MIT.
- U²PL: <https://github.com/haochen-wang409/U2PL>, Apache-2.0.

The official repositories are not bundled. Use the official project citations alongside this implementation's citation. Backbone, DeepLabV3+, U-Net, EfficientNet and LayerCAM method references should also be cited in the manuscript. No authorship of those methods is claimed here.

## Data and checkpoints

Public datasets retain their original terms; locally obtained images may have different redistribution rights. No dataset, annotation, sample-level manifest, pretrained model or trained checkpoint is redistributed by this code-only repository.

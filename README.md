# Soybean Lesion Segmentation

Model architectures and image inference for soybean leaf and lesion segmentation, based on MMSegmentation 1.2.2.

中文说明：[README_zh-CN.md](README_zh-CN.md)。

## Models

| Component | Architecture | Configuration |
| --- | --- | --- |
| Lesion teacher | DeepLabV3+ / ResNet-50 | `configs/soybean/teacher.py` |
| Leaf segmentation | U-Net | `configs/soybean/leaf.py` |
| Lesion student | DeepLabV3+ / MobileNetV3-Large | `configs/soybean/student.py` |
| Disease semantics | EfficientNet-B0 / LayerCAM | `tools/dadou_semantic/export_multilayer_layercam.py` |

Pixel labels: `0=background`, `1=healthy_leaf`, `2=lesion`; the leaf model uses `0=background`, `1=leaf`.

## Inference

Follow [installation](docs/PAPER_INSTALLATION.md), then supply the matching model checkpoint:

```bash
python tools/dadou/infer_teacher.py configs/soybean/student.py model.pth image.jpg --output-dir outputs/student
python tools/dadou/infer_teacher.py configs/soybean/teacher.py model.pth images/ --output-dir outputs/teacher
python tools/dadou/infer_leaf.py configs/soybean/leaf.py model.pth images/ --output-dir outputs/leaf
```

The probability exporters write class masks, overlays and uint16 probability maps. LayerCAM usage is available through `python tools/dadou_semantic/export_multilayer_layercam.py --help`.

## Downloads

- [Model and inference source v1.2.0](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/v1.2.0).
- [Soybean manual annotations v2.1.0](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-soybean-manual-v2.1.0): 172 original images with manual LabelMe annotations and corresponding pixel masks.
- [MH-SoyaHealthVision subset v1.1.0](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-mh-soyahealthvision-v1.1.0): 160 original images, including 80 with manual pixel annotations.

Dataset layouts, source citations and label encoding are described in [PAPER_DATA.md](docs/PAPER_DATA.md).

## License

Retain the [Apache-2.0 license](LICENSE), [MMSegmentation citation](CITATION.cff) and [third-party notices](THIRD_PARTY_NOTICES.md).

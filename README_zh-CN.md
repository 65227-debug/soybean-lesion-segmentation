# 大豆病斑分割

基于MMSegmentation 1.2.2的模型结构、图像推理与概率图导出。

## 模型

- 病斑教师：DeepLabV3+–ResNet-50，`configs/soybean/teacher.py`。
- 叶片分割：U-Net，`configs/soybean/leaf.py`。
- 病斑学生：DeepLabV3+–MobileNetV3-Large，`configs/soybean/student.py`。
- 病种语义：EfficientNet-B0与多层LayerCAM。

三分类标签：背景0、健康叶片1、病斑2；叶片模型标签：背景0、叶片1。

## 推理

按[环境安装说明](docs/PAPER_INSTALLATION.md)准备依赖，并提供对应的模型文件：

```bash
python tools/dadou/infer_teacher.py configs/soybean/student.py model.pth image.jpg --output-dir outputs/student
python tools/dadou/infer_teacher.py configs/soybean/teacher.py model.pth images/ --output-dir outputs/teacher
python tools/dadou/infer_leaf.py configs/soybean/leaf.py model.pth images/ --output-dir outputs/leaf
```

概率图工具输出类别掩码、叠加图和16位概率图。多层LayerCAM用法见`python tools/dadou_semantic/export_multilayer_layercam.py --help`。

## 下载

- [模型与推理源码 v1.2.0](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/v1.2.0)。
- [大豆人工标注数据 v2.1.0](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-soybean-manual-v2.1.0)：172张真实原图、人工LabelMe标注和对应像素掩膜。
- [MH-SoyaHealthVision子集 v1.1.0](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-mh-soyahealthvision-v1.1.0)：160张真实原图，其中80张提供人工像素标注。

数据目录、原始数据来源和标签编码见[数据说明](docs/PAPER_DATA.md)。

## 许可

保留[Apache-2.0许可证](LICENSE)、[MMSegmentation引用](CITATION.cff)与[第三方归属说明](THIRD_PARTY_NOTICES.md)。

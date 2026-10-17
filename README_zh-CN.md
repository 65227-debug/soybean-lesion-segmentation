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
- [Soybean病斑分割数据集](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-soybean-v2.0.0)：1567张不同原图，其中172张带像素级标注。
- [Soybean原图与掩码](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-soybean-original-pairs-v2.0.0)：617对。
- [MH-SoyaHealthVision子集](https://github.com/65227-debug/soybean-lesion-segmentation/releases/tag/dataset-mh-soyahealthvision-v1.0.0)。

数据目录与标签编码见[数据说明](docs/PAPER_DATA.md)。

## 许可

保留[Apache-2.0许可证](LICENSE)、[MMSegmentation引用](CITATION.cff)与[第三方归属说明](THIRD_PARTY_NOTICES.md)。

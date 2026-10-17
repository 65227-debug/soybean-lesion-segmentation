# 融合叶片空间约束与病种语义软证据的大豆病斑分割

本仓库用于公开论文实验代码，包含本项目使用的完整 MMSegmentation 1.2.2 Python 源码、配置、教师/叶片/病种语义模型、三张基础图、修正版三态门控、连续像素权重、学生蒸馏、消融及强半监督基线。

图像、标注和划分清单作为独立的数据集发布包提供，不放入 Git 源码历史。模型权重、训练结果及论文文件未包含。`data/`、`work_dirs/` 等本地目录已加入 `.gitignore`。

## 数据集下载

- [Dadou 实验数据集 v1.0.0](https://github.com/652276002-debug/soybean-lesion-segmentation/releases/tag/dataset-dadou-v1.0.0)：精标数据、弱标签图像、人工标注、固定划分及已使用的伪标签训练数据。下载全部 5 个 ZIP 分卷，按发布页说明合并后解压。
- [MH-SoyaHealthVision 实验子集 v1.0.0](https://github.com/652276002-debug/soybean-lesion-segmentation/releases/tag/dataset-mh-soyahealthvision-v1.0.0)：80 张人工像素标注测试图像和 80 张独立无标注适配图像。

包内附掩码、相对路径清单、使用说明和校验信息；数据文件的权利说明与软件许可证分别保留。

## 复现入口

- 安装和已记录环境：[docs/PAPER_INSTALLATION.md](docs/PAPER_INSTALLATION.md)。
- 数据结构、标签和清单字段：[docs/PAPER_DATA.md](docs/PAPER_DATA.md)。
- 最终 G、强基线、多种子及标签比例实验：[docs/PAPER_REPRODUCTION.md](docs/PAPER_REPRODUCTION.md)。
- 代码归属与引用：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

模型：病斑教师使用 DeepLabV3+–ResNet-50，独立叶片分割使用 U-Net，独立病种语义使用 EfficientNet-B0 + 多层 LayerCAM，学生使用 DeepLabV3+–MobileNetV3-Large。分割输入为 512×512，标签为背景0、健康叶片1、病斑2、忽略255。

投稿最终 G 的主链为：纯监督 A → 修正版门控训练 G0 → 蒸馏 G1 → 仅验证集校准 → 冻结测试。以 `paper_code/submission_supplement_experiments/frozen_experiment_plan.json` 为准；旧 F、G3/G4 等开发试验保留作历史代码，不应当作最终 G 混用。

发布副本只做路径可移植化及发布说明补充。冻结门控参数、训练配置及蒸馏配置与实验原文件逐字节核验一致；记录见 `CODE_RELEASE_MANIFEST.json`。尚未提供数据的使用者需先准备清单和本地预训练文件，不能直接获得原论文的数值结果。

## 论文表述

当前手稿采用108张训练图像、36张验证图像和36张测试图像。最终 G 的 mIoU 为85.57±0.63%，相比 ST++ 提升1.65个百分点。完整训练链的总计算量并不相等。UniMatch/ST++/U²PL 为协议对齐适配，保留官方引用和许可证来源。

保留 OpenMMLab 的 Apache-2.0 许可证、版权声明和 `CITATION.cff`。论文作者、题目终稿和 DOI 确认后可另外补充，不覆盖上游贡献者信息。

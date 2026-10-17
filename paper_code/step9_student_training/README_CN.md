# 第九步：半监督学生模型训练

本文件夹用于复现论文第九步：使用固定教师生成的双证据门控伪标签，训练三类像素输出的轻量学生模型。

## 1. 模型和数据

- 学生模型：DeepLabV3+ + MobileNetV3-Large。
- 输入尺寸：512×512。
- 输出类别：背景、健康叶片组织、病斑。
- 初始化：MobileNetV3-Large 使用 ImageNet 预训练权重，DeepLabV3+ 解码头随机初始化。
- 人工精标训练集：108 张。
- 双证据门控伪标签训练集：784 张。
- 验证集：27 张。
- 测试集：37 张，仅在验证集确定最佳权重后加载一次。

监督与伪标签图像每轮都进行同步在线增强：水平翻转、垂直翻转、±15°旋转，以及亮度、对比度和饱和度扰动。图像、掩膜和像素置信权重使用相同的几何变换；掩膜始终采用最近邻插值。

## 2. 损失函数

监督损失为交叉熵与软 Dice 损失之和：

```text
L_sup = L_CE + L_Dice
```

伪标签损失为逐像素置信权重归一化的交叉熵：

```text
L_pseudo = sum(W_i * CE_i) / (sum(W_i) + epsilon)
```

总损失为：

```text
L = L_sup + lambda_u(t) * L_pseudo
```

`lambda_u` 在第 1 轮为 0，第 10 轮升至 0.2，第 30 轮升至 0.5，第 60 轮最终升至 0.7。本次正式实验完整执行 60 轮，模型选择指标只使用验证集 mIoU。

## 3. 文件说明

- `mmseg_student_config.py`：MMSegmentation 模型结构和推理配置。
- `training_config.json`：训练超参数、类别权重、增强和半监督权重计划。
- `prepare_student_dataset.py`：整理精标/伪标签、置信权重并生成清单。
- `train_student.py`：联合训练、验证选模、早停、最终测试和论文曲线导出。
- `run_step9.sh`：从数据整理到正式训练的一键复现入口。

## 4. 复现命令

在 mmsegmentation 项目根目录执行：

```bash
bash paper_code/step9_student_training/run_step9.sh
```

若 Python 环境不同，可显式指定：

```bash
PYTHON_BIN=/path/to/python \
RUN_DIR=work_dirs/your_run_name \
bash paper_code/step9_student_training/run_step9.sh
```

正式训练前可进行一次不读取测试集的单轮检查：

```bash
python \
  paper_code/step9_student_training/train_student.py \
  --smoke-test \
  --work-dir work_dirs/dadou_student_mobilenetv3_smoke
```

## 5. 本次正式实验结果

- 完成轮数：60。
- 最佳轮次：第 7 轮。
- 最佳验证 mIoU：76.47%。
- 最佳验证病斑 IoU：50.86%。
- 测试 mIoU：81.48%。
- 测试 mDice：89.13%。
- 测试病斑 IoU：62.42%。
- 测试病斑 Dice：76.87%。
- 测试病斑 Precision：87.54%。
- 测试病斑 Recall：68.51%。
- 可训练参数总量：7.337M。

正式产物位于 `work_dirs/dadou_student_deeplabv3plus_mobilenetv3_full60`。其中 `best_validation_mIoU.pth` 是论文与推理应使用的权重，`epoch_metrics.csv` 保存每一轮的全部损失和验证指标，`training_curves.png/.pdf` 可直接用于论文绘图或重新排版，`training_summary.json` 保存最终机器可读结果。

## 6. 论文实验注意事项

测试结果高于验证结果说明两个子集难度或类别分布存在差异，不能据此反向调整阈值或超参数。若要证明伪标签带来的提升，需要保持 MobileNetV3 学生结构、训练/验证/测试划分、增强和随机种子完全一致，再补做一个“只用 108 张人工精标图”的监督基线；本次实验本身不能与结构不同的教师模型直接归因比较。

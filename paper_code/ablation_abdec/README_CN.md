# A/B/D/E/C 消融训练复现说明

## 组别定义

| 组别 | 训练方式 |
|---|---|
| A | 仅使用 108 张人工精标图像进行监督训练 |
| B | 固定分割教师 + 置信度阈值伪标签 |
| C | EMA Mean Teacher，一致性学习基线 |
| D | 分割置信度 + 叶片空间门控 |
| E | 分割置信度 + 病种语义 LayerCAM 门控 |
| F | 既有完整双证据门控模型，用于统一对比 |

所有组使用相同的 DeepLabV3+ + MobileNetV3-Large、ImageNet 预训练初始化、
512×512 输入、随机种子、增强、优化器、60 个 epoch 和每个 epoch 98 次参数更新。
最佳权重仅按 36 张验证图像的 mIoU 选择。A/B/C/D/E 全部训练完成并冻结后，
才统一加载 36 张测试图像；测试集不参与训练、早停、阈值或模型选择。

## 复现

从 mmsegmentation 项目根目录运行：

```bash
bash paper_code/ablation_abdec/run_abdec.sh
```

也可单独运行准备、训练或统一评测脚本。正式训练使用 `workers=4`，避免本机多进程
数据加载在高负载时出现段错误。完整阈值、损失权重和增强参数见
`ablation_config.json`。

## 输出位置

- `work_dirs/dadou_ablation_A` 至 `work_dirs/dadou_ablation_E`：最佳/末轮权重、
  逐 epoch CSV、PNG/PDF 收敛图、有效配置和训练摘要。
- `work_dirs/dadou_ablation_comparison`：统一测试 CSV、逐图指标、预测掩膜、
  叠加图、对比图和成对 bootstrap 区间。
- `data/dadou_abdec`：B/D/E 使用的统一弱图像、伪标签、像素权重和清单。

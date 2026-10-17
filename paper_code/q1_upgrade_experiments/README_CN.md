# 一区提升补充实验

本目录只对冻结实验产物做补充分析，不修改阈值、不重新选模，也不读取外部冻结测试标签进行训练。

## 脚本

- `analyze_severity_boundary.py`：在固定 36 张内部测试集上计算 A--G 的连续严重度、Bland--Altman、一致性指标、病斑边界指标和分尺度连通域召回。
- `plot_pseudo_quality_tradeoff.py`：把既有 20 张验证图门控网格绘制成 Precision--Recall、覆盖率--F1 和阈值敏感性图；不会改动已选阈值。
- `benchmark_efficiency.py`：在同一 RTX 5090 D v2、batch size 1、512×512、AMP FP16 条件下统一测教师与最终 G 的参数量、FLOPs、延迟、FPS、checkpoint 大小和峰值显存。
- `analyze_incremental_stages.py`：复用验证集冻结的 F、G0、G1 和 G2（最终 G）权重，补“严格门控→连续门控→蒸馏→校准”逐项消融及逐图配对 Bootstrap。
- `run_q1_upgrade_experiments.sh`：依次运行上述三项。

## 重要口径

严重度定义为 `100 × 病斑像素 /（健康叶片像素 + 病斑像素）`。它来自同一张像素掩膜，不等价于田间人工目测或独立仪器测量，因此论文中应写成“基于分割掩膜的连续严重度估计”。

边界 F1 的容差半径为 2 px；Boundary IoU 使用宽度为 3 px 的叶内边界带。病斑连通域采用 8 邻域，小/中/大阈值固定为 `<64`、`64--255` 和 `>=256` 像素；预测病斑覆盖真值连通域至少 10% 时记为检出。该分尺度结果属于补充性分析，必须与阈值一并报告。

FLOPs 使用项目自带 `tools/analysis_tools/get_flops.py`；其不支持算子警告保存在原始日志中，论文应说明 FLOPs 为追踪估计值。测速只计网络前向、logit 上采样和 argmax，不含磁盘读取与预处理。

## 运行

```bash
bash paper_code/q1_upgrade_experiments/run_q1_upgrade_experiments.sh
```

默认输出：`paper_materials/一区提升实验补充_20260826/`。

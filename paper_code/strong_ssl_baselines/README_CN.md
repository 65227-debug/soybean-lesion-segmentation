# 强半监督基线的协议对齐复现

本目录用于在大豆三分类分割任务上复现 UniMatch、ST++ 与 U²PL。它不是对原论文 VOC/Cityscapes 数值的逐字节复刻，而是将方法核心机制适配到与最终 G 相同的实验协议：DeepLabV3+ + MobileNetV3-Large、512×512、108 张像素级标注、1390 张纯未标注图、5880 次优化、验证集选模。

训练代码只从未标注清单读取 `sample_id`、`image_path` 和 `source_image_path`。即使原 CSV 中存在 `class_name`、伪标签或权重路径，加载函数也会主动丢弃这些列，因此不会使用病种弱标签、LayerCAM、叶片模型或 G 的伪标签。

## 当前执行顺序

1. `verify_protocol.py`：校验 108/36/36/1390 数量、文件可读性及按源文件 SHA-256 的跨划分重复。
2. `train_unimatch.py`：双强增强、CutMix、特征扰动和置信度门控。
3. `train_stpp.py`：监督初训、跨检查点稳定性筛选、可靠集再训练、全未标注集再训练。
4. `train_u2pl.py`：EMA 教师、基于熵的可靠/不可靠像素划分与类别队列对比学习。

统一配置见 `aligned_protocol.json`，官方仓库、提交号和许可证见 `OFFICIAL_REFERENCES.json`。

## 命令

```bash
python paper_code/strong_ssl_baselines/verify_protocol.py
python paper_code/strong_ssl_baselines/train_unimatch.py --smoke-test --work-dir work_dirs/dadou_unimatch_aligned_smoke
python paper_code/strong_ssl_baselines/train_unimatch.py --work-dir work_dirs/dadou_unimatch_aligned_seed20260803
python paper_code/strong_ssl_baselines/train_stpp.py --work-dir work_dirs/dadou_stpp_aligned_seed20260803
python paper_code/strong_ssl_baselines/train_u2pl.py --work-dir work_dirs/dadou_u2pl_aligned_seed20260803
python paper_code/strong_ssl_baselines/summarize_reproductions.py
```

正式训练完成后才加载“验证 mIoU 最优”冻结权重评估 36 张测试集；测试结果不参与阈值、轮数或模型选择。

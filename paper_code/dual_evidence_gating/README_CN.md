# 双证据门控：第六、七、八步复现代码

本文件夹提供论文方法中以下三个步骤的完整可复现实现：

1. 第六步：分割教师置信度、叶片空间证据和病种语义证据共同生成候选伪标签；
2. 第七步：为可信病斑、可信健康组织和可信背景计算连续像素权重；
3. 第八步：根据整图质量指标决定保留或排除一张弱标签图。

代码只使用验证集选择参数。36 张最终测试集不用于修改本目录中的参数。

## 标签与权重定义

最终伪标签使用以下编码：

| 数值 | 含义 |
| ---: | --- |
| 0 | 可信背景 |
| 1 | 可信健康叶片组织 |
| 2 | 可信病斑 |
| 255 | 不确定像素，训练时使用 `ignore_index=255` |

病斑像素必须同时满足：

```text
Pseg >= tau_p 且 Asem >= tau_s 且位于腐蚀后的叶片内部
```

健康组织像素必须同时满足：

```text
Pseg <= tau_n 且 Asem <= tau_sn 且位于腐蚀后的叶片内部
```

可信背景使用 `Pleaf <= 0.10`，其余像素设为 255。像素权重为：

```text
病斑：Pseg * Asem
健康组织：(1-Pseg) * (1-Asem)
背景：(1-Pleaf) * (1-Pseg)
忽略：0
```

## 实际选择的参数

`selected_config.json` 保存了参数、选择指标和是否使用测试集等完整记录。最终值为：

```text
tau_p  = 0.80
tau_n  = 0.15
tau_s  = 0.40
tau_sn = 0.30
叶片阈值 = 0.50
背景叶片概率上限 = 0.10
腐蚀半径 = 5 像素（512×512尺度）
语义—分割最小重叠率 = 0.02
病种分类最小置信度 = 0.60
最少可信病斑像素 = 16
```

## 文件说明

- `gating_core.py`：门控公式、权重公式、整图筛选及标签着色。
- `prepare_validation_manifest.py`：只选择验证集中语义模型支持的FLS/TLS图像。
- `tune_thresholds.py`：搜索四个像素门控阈值。
- `tune_image_filter.py`：选择整图语义—分割重叠阈值。
- `generate_gated_pseudo_labels.py`：批量生成第六至八步结果。
- `audit_outputs.py`：训练前完整性检查。
- `default_config.json`：阈值搜索起始配置。
- `selected_config.json`：本实验冻结后的最终配置及验证记录。
- `run_steps_6_8.sh`：从验证清单到最终审计的完整命令。

## 运行方法

在 mmsegmentation 项目根目录运行：

```bash
bash paper_code/dual_evidence_gating/run_steps_6_8.sh
```

如果第五步基础图已经存在，只需执行：

```bash
python paper_code/dual_evidence_gating/tune_thresholds.py
python paper_code/dual_evidence_gating/tune_image_filter.py
python paper_code/dual_evidence_gating/generate_gated_pseudo_labels.py
python paper_code/dual_evidence_gating/audit_outputs.py
```

学生训练应读取 `data/dadou/06_07_08_双证据门控/可用于学生训练.csv`，并同时加载伪标签和像素权重。不能把 `整图排除.csv` 中的图像加入伪标签损失。

## 适用范围

当前EfficientNet-B0病种语义模型只覆盖蛙眼叶斑病和靶斑病，因此本代码生成结果也只覆盖这两个病种。健康图片的专门规则已经在核心代码中实现，但当前1390张弱标签池没有健康类；RAB不能在缺少相应语义模型类别时直接套用本次参数。


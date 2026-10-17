> Release note: this file retains historical development notes, including G3/G4. For the paper's final G, use the root README and docs/PAPER_REPRODUCTION.md; the frozen submission chain is A -> G0 -> G1 -> validation calibration.

# 修正版 F：双证据门控学生训练

本目录对应对原 F 实验的可复现修正，不覆盖原 A–F 结果。

主要修正：

1. 监督图像与伪标签图像分开前向；BatchNorm 只由监督图像更新。
2. 前 15 轮为纯监督预热，随后才逐步加入伪标签损失。
3. 使用三分类分割教师概率与叶片概率共同确认背景和健康组织。
4. LayerCAM 对病斑作为低阈值门控和软权重，不再用于强制判定健康组织。
5. 伪标签损失按类别分别归一化，避免病斑被背景像素数量淹没。
6. 根据 `scene_id` 进行逆场景频率采样，减少连续帧和重复场景偏置。
7. 所有阈值只在像素级验证集上选择，测试集在最佳验证权重冻结后才加载。

运行入口与实际产物路径记录在同目录的 `run_corrected_f.sh`。

`calibrate_single_student.py` 只使用验证集选择健康组织和病斑的类别偏置，
并把偏置折叠进分类头权重。校准后仍是单模型、单次前向，不增加参数量或推理开销。

`select_logit_ensemble.py` 用验证集检查 A 与修正版 F 的软预测是否互补。
其结果只作为后续蒸馏教师，不作为最终移动端学生模型报告。

`train_distilled_student.py` 用双证据有效区域上的软目标把上述互补信息蒸馏回
单个 DeepLabV3+--MobileNetV3 学生，训练过程不读取测试集。
`distillation_v2_config.json` 使用RAB精标过采样、弱图按病种/场景均衡、门控外病斑边界低权重蒸馏，
并使用偏召回的病斑Tversky项。
`strong_teacher_distillation_config.json` 直接使用已训练的
DeepLabV3+--ResNet50强分割教师提供软边界，同时在RAB精标图上蒸馏。

`evaluate_frozen_student.py` 只用于验证集选定并冻结权重后的单次测试评估，
保存总体指标、逐图指标、预测掩膜和叠加图。

`finalize_results.py` 汇总 A--G 测试表、配对 bootstrap、训练收敛图和论文结果说明。

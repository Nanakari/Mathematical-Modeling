# BERT 后编码器优化实验报告（2026-09-26）

本轮完成 13 次本地 GPU 训练、两个独立 FP16 推理包及各 120 个验证场景。结论是保留 pooled 作为主方案：新增模块没有在两个随机种子上稳定改善主要指标。content_gate 改善了平均 Macro-F1、Neutral Recall 和 MAE，但牺牲了平均 Accuracy 与缺失场景 Accuracy，作为研究备选保留，不替换默认模型。

所有结果来自固定 validation；本轮未访问 official test。不能与历史 test 分数或不同划分、精度、集成结果直接比较。本轮没有获得可复现的主指标提升。

## 实验设置与方案取舍

- 固定 BERT-Mini 架构、预训练来源及微调配方；“固定编码器”指不更换 backbone，不是冻结其梯度。保留 BERT 时序输出、两层轻量 AV CNN、统一 128 维空间、mask-aware pooling、AV direct branch。
- train 3395、valid 728，valid 三类数为 206/184/338。数据检查显示三分类标签等于回归值符号加一，Neutral 为严格零值。因此不引入任意 ±0.5 中性带，也不直接用回归值接近零推断 Neutral 概率。
- 本地 GTX 1660 Ti 6GB，训练 FP32，导出 FP16；batch 32，最多 15 epoch，early stopping patience 4；BERT LR 2e-5，其他模块 LR 5e-4，weight decay 0.01，回归权重 0.2。种子为 1729、2718，验证缺失种子 101。
- 基线已有连续缺失增强：完整输入概率 0.6，缺失率 0.1/0.3/0.5，覆盖模态组合及首/中/尾/随机位置。补零位置不作为真实观测参与池化。
- 不重复此前无稳定收益的重型 Transformer、Conformer、cross-attention、普通 attention pooling 或旧 KD。保留轻量 AV 编码器，先隔离融合和预测机制的贡献。

## 实现及消融关系

1. pooled → content_gate：池化后的三模态表示产生三路 softmax 标量权重，仅在可用模态间归一化，再进入原有融合头。新增 1173 参数，门控零初始化与 pooled 初始行为一致；共享参数初始化和 RNG 对齐。
2. content_gate → availability_gate：相同参数结构，启用额外三路 availability 与三路 observed density。文本统计排除 CLS/SEP，密度以固定 50 个对齐位置为分母。全缺失时门控输出为零，避免 NaN。
3. 以 availability_gate 为固定锚点，分别比较 CLS、CLS+masked mean、focal loss（gamma=1）、联合分类回归头、缺失强度课程。没有把所有模块一次堆叠。
4. 联合头计算 mu=p_neg*m_neg+p_pos*m_pos，其中 m_neg∈[-3,0]、m_pos∈[0,3]、Neutral 分量为零；这是条件幅值混合，不是强迫回归均值近零就分类为 Neutral。consistency=0.05 单独与相同联合头、consistency=0 比较，约束原直接回归输出和 mu 的 SmoothL1 一致性。
5. 缺失课程仅把原有缺失率乘数从首轮 0.5 提升到第五轮 1.0，其他增强配方不变。

原默认配置仍为 pooled。上述功能均为可选开关，未覆盖历史结果或 temporal_v4。本轮 focal 是 Neutral 相关分类损失的一个候选，不代表已经穷尽 weighted CE / logit adjustment；没有再次展开大范围超参数搜索。

## 种子 1729：九组消融

下表缺失 Acc 为既有 v4_32 协议的 31 个缺失场景等权平均；完整输入单独统计。MAE 越低越好，其余指标越高越好。

| 方案 | Acc | Macro-F1 | Neutral R | MAE | Pearson | 缺失 Acc | Neutral→Positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| availability_gate | 0.6291 | 0.5702 | 0.2391 | 0.6193 | 0.6086 | 0.6090 | 99 |
| content_gate | 0.6291 | 0.5692 | 0.2337 | 0.6192 | 0.6079 | 0.6081 | 100 |
| pooled | 0.6264 | 0.5603 | 0.2174 | 0.6256 | 0.6062 | 0.6078 | 108 |
| availability_cls | 0.6126 | 0.5540 | 0.2717 | 0.6420 | 0.6011 | 0.5972 | 106 |
| availability_cls_mean | 0.6195 | 0.5741 | 0.2826 | 0.6215 | 0.6090 | 0.6015 | 88 |
| availability_focal_gamma1 | 0.6264 | 0.5617 | 0.2174 | 0.6158 | 0.6084 | 0.6040 | 102 |
| availability_joint_mix | 0.6305 | 0.5790 | 0.2717 | 0.6354 | 0.6002 | 0.6068 | 91 |
| availability_joint_mix_consistency005 | 0.6305 | 0.5754 | 0.2554 | 0.6359 | 0.6022 | 0.6077 | 96 |
| availability_severity_ramp | 0.6250 | 0.5670 | 0.2391 | 0.6258 | 0.6036 | 0.6030 | 95 |

CLS/hybrid 没有改善 Accuracy；focal 的 MAE 略好，但分类和缺失 Accuracy 不如其 availability 锚点；联合头首种子 Accuracy 达到 0.6305，但 MAE 恶化。一致性损失没有进一步提高 Accuracy，课程方案也未胜出。不能据此把这些模块串联成所谓最优模型。

## 双种子复核：四组方案

以下是两个独立训练结果的算术平均，**不是 ensemble 推理结果**。

| 方案 | Acc | Macro-F1 | Neutral R | MAE | Pearson | 缺失 Acc |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | 0.6223 | 0.5688 | 0.2690 | 0.6270 | 0.6078 | 0.6042 |
| content_gate | 0.6195 | 0.5748 | 0.3207 | 0.6181 | 0.6079 | 0.5951 |
| availability_gate | 0.6181 | 0.5652 | 0.2582 | 0.6328 | 0.6062 | 0.6016 |
| availability_joint_mix | 0.6181 | 0.5682 | 0.2717 | 0.6406 | 0.5995 | 0.6010 |

pooled 的平均 Accuracy 与缺失 Accuracy 最好。content_gate 相比 pooled：Accuracy 下降约 0.0027，Macro-F1 增加约 0.0059，Neutral Recall 增加约 0.0516，MAE 降低约 0.0089，但缺失 Accuracy 下降约 0.0091。availability 的首种子收益未复现，联合头在第二种子分类和回归均表现较差。

依据预先确定的 Accuracy 主指标，并参考回归与缺失表现，保留 pooled。最终冻结其 seed 1729 单模型；seed 2718 仅用于稳定性复核。未宣称统计显著性，两个 seed 仍不足以精确估计收益分布。

各类召回和误分如下，防止仅观察 Neutral 而忽略其他类别的损失：

| 方案 | seed | Negative R | Neutral R | Positive R | Neutral→Positive /184 |
| --- | --- | --- | --- | --- | --- |
| availability_gate | 1729 | 0.7087 | 0.2391 | 0.7929 | 99 |
| content_gate | 1729 | 0.7136 | 0.2337 | 0.7929 | 100 |
| pooled | 1729 | 0.6650 | 0.2174 | 0.8254 | 108 |
| availability_gate | 2718 | 0.7087 | 0.2772 | 0.7249 | 81 |
| availability_joint_mix | 2718 | 0.6990 | 0.2717 | 0.7308 | 82 |
| content_gate | 2718 | 0.5874 | 0.4076 | 0.7337 | 84 |
| pooled | 2718 | 0.7039 | 0.3207 | 0.7278 | 75 |
| availability_joint_mix | 1729 | 0.7087 | 0.2717 | 0.7781 | 91 |

## 导出包扩展鲁棒性评估

两份 FP16 包均在同一 728 条 valid、相同指纹与缺失随机种子上使用 CUDA 重载评估。每份 120 场景 = 1 完整 + 112 局部缺失 + 7 整模态缺失。

112 局部场景为 7 模态子集 × 4 位置 × 4 缺失率（0.1/0.3/0.5/0.7）。下表仅对这些局部场景平均，不能与上面的 31 场景均值混用：

| FP16 包 | Acc | Macro-F1 | Neutral R | MAE | Pearson |
| --- | --- | --- | --- | --- | --- |
| pooled | 0.5986 | 0.5423 | 0.2541 | 0.6535 | 0.5533 |
| content_gate | 0.5967 | 0.5469 | 0.2723 | 0.6489 | 0.5554 |

整模态缺失另行报告。原局部缺失规划器会保留至少一个观测，因此这七项通过直接清空所选模态的观测掩码实现，不是把局部缺失率简单设成 1。已验证不修改原样本、标签及未选模态。

| 完全缺失的模态 | pooled Acc | content_gate Acc |
| --- | --- | --- |
| text | 0.3668 | 0.3530 |
| audio | 0.6223 | 0.6291 |
| vision | 0.6250 | 0.6140 |
| text+audio | 0.3448 | 0.3187 |
| text+vision | 0.3338 | 0.3269 |
| audio+vision | 0.6181 | 0.6154 |
| text+audio+vision | 0.2830 | 0.2830 |

文本完全缺失使 pooled Accuracy 降至 0.3668，而单独缺音频或视觉仍约 0.62，反映目前预测强依赖文本。整模态缺失是额外压力测试，不等同于训练中的局部连续缺失。三模态全缺失时输入没有内容信息、预测为常数，Pearson 无定义，JSON 保留 null，没有错误地填零。

据此，后续最值得验证的假设是：为 AV 增加独立监督，并在训练中显式加入整模态缺失，以提高非文本分支的可用性。应先分别与当前基线做单因素比较，再考虑组合。本轮没有执行这些新实验，不把该假设当作已证实的提升。

## 单模型推理包与验证

| 包 | 完整目录 bytes | ZIP bytes | 完整 Acc | 31缺失 Acc |
| --- | --- | --- | --- | --- |
| pooled | 24168353 | 21574513 | 0.6264 | 0.6079 |
| content_gate | 24170880 | 21576531 | 0.6291 | 0.6080 |

主包 pooled 约 24.17 MB，严格小于 50,000,000 bytes。参数量 11,442,760；门控方案 11,443,933，联合头方案 11,444,191。各 FP32 checkpoint 的实际大小、SHA256、参数量、完整与缺失指标见 experiment_metrics.csv。

包包含模型权重、tokenizer、scaler、推理运行代码和依赖声明；通用 Python/PyTorch 环境依赖按 requirements 安装，不计作随附模型文件。两个包是独立替代方案，不应合并作为 ensemble 提交。FP16 指导出权重精度，本轮训练为 FP32。导出后已进行独立离线进程重载、ZIP 内容/CRC、包哈希及包级验证，FP16 与训练 checkpoint 指标有少量数值差异，因此分别留存结果。

pooled 冻结记录：freeze_pooled.json，status=frozen。未调用 final-test。

- pooled ZIP SHA256：`5e4f23a498ec5e672bab058dce3172a4d93b0e4c3e9feca7953aeef32b5010a6`
- content_gate ZIP SHA256：`b6e7b7f498f267021915984157ef9723c9e4af5241096a577abfbb0fc0c57042`

核心源码五文件冻结哈希与 13 次训练 sidecar 均一致；13 个 checkpoint 实际哈希、valid 数量及 official_test_evaluated=false 已核验。新增损失单元测试 5 项通过；模型预检覆盖门控初始化等价、共享初始化/RNG、mask/density、缺失文本池化、全缺失有限值及联合回归范围。120 场景结果的包哈希、样本指纹、场景数量与冻结记录已再次交叉核验。

## 交付文件

- [完整实验指标 CSV](experiment_metrics.csv)
- [主模型 pooled FP16 ZIP](packages/pooled.zip)
- [备选 content_gate FP16 ZIP](packages/content_gate.zip)
- [冻结记录](freeze_pooled.json)
- [推理包清单](packages/selected_package_manifest.json)
- [pooled 120 场景明细](packages/pooled_systematic_120_cuda.json)
- [content_gate 120 场景明细](packages/content_gate_systematic_120_cuda.json)
- [源码哈希](source_hashes.json)

验证集被用于本轮模型选择，因此这些结果不构成新的独立泛化测试。最终保留基线是依据实验的取舍，并非未完成执行；所有候选代码、配置、日志和失败消融均保留以便复查。

# 问题 2 小规模完整流程结果

使用官方训练集中的 512 条样本和验证集中的 128 条样本。标准化仅拟合这批训练样本，未评价官方测试集。

依据预先配置的验证评分选出的模型是 **augmented_concat**。评分为完整输入与指定缺失情景下 MAE + (1 − Macro-F1) 的平均值；权重可替换。

| 设置 | 最佳轮次 | 选择评分 | 完整输入 Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|---:|
| clean_concat | 8 | 1.4875 | 0.4844 | 0.3845 | 0.8655 | 0.2913 |
| augmented_concat | 8 | 1.4759 | 0.4922 | 0.3895 | 0.8578 | 0.2988 |
| augmented_gated | 1 | 1.5433 | 0.4844 | 0.3068 | 0.8409 | 0.2000 |

作为参考，使用训练集多数类和训练集强度中位数作恒定预测，在同一验证子集上的 Accuracy=0.4922、Macro-F1=0.2199、MAE=0.8529。常数回归输出的 Pearson 无定义。此参考不参与选模。

## 结果与解释边界

每个模型均在同一份 85 个验证情景上评价。metrics.csv 保存完整指标，validation_predictions.csv 保存逐样本预测。缺失区间、实际缺失比例和种子保存在压缩场景库中。

clean_concat 与 augmented_concat 的差别是缺失增强；augmented_concat 与 augmented_gated 的差别是融合结构及其参数量。三者共用样本、标准化器、编码器配置和验证区间。

验证集同时用于选轮次、选模型与诊断，所以这些结果不是独立留出集上的无偏性能估计。只有一个训练种子，不能据此声称显著提高。当前文本编码器为随机初始化的词元嵌入，尚未采用已确认词表的预训练 BERT。

error_cases.csv 按强度绝对误差列出前 20 条样本，并记录分类错误、分类与回归符号不一致、原始模态可用位置数。error_by_class.csv 给出分类别误差。这些是错误定位线索，不是因果归因。

附件 3 aligned 的 30 条样本已生成预测。attachment3_DRAFT_predictions.csv 使用文件名追踪键及暂定列名；无标签，不能据其计算性能。正式提交格式与文本编码器仍需确定。

## 文件说明

- selected_model.pt：选中模型的推理参数，不含优化器。
- 各实验 best.pt / last.pt：最佳推理参数 / 最后一轮模型及优化器状态。
- standardizer.json：训练集标准化参数。
- config.json / split_ids.json / report.json：配置、样本划分记录、环境与运行摘要。
- diagnostics.png / missingness_curves.png / position_modality_heatmap.png：选模、混淆矩阵、回归散点、缺失性能曲线及模态与位置对比。

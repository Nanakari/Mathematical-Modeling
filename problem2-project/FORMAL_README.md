# 问题 2 完整解答与正式实验复现

本版本替代前面的随机词元小规模实验，使用完整训练/验证划分、已核验词表的冻结 BERT-Tiny、稀疏词元词组、带掩码的音视频时序统计和正则化双任务预测。所有原始数据保持不变。

## 输出入口

- outputs/formal/问题2_完整解答.md：问题分析、假设、数学模型、优化、实验、消融、不确定性、专项预测与局限。
- outputs/formal/robustness_summary.csv：独立测试完整输入和缺失条件汇总。
- outputs/formal/condition_summary.csv：按模态、比例、位置汇总，随机位置先对三个种子平均。
- outputs/formal/附件3_问题2_预测结果.csv：全部 30 条专项结果；source_file 保留文件名。
- outputs/formal/frozen_selection.json：测试评价之前冻结的参数、数据 ID 和模型哈希。
- outputs/formal/bootstrap_confidence_intervals.csv：按原始视频分组的 500 次成对 bootstrap 区间。

## 环境

已在项目 .venv 中运行。CPU 执行，不依赖 CUDA。推荐严格复现安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1 matplotlib==3.10.0 joblib==1.4.2 threadpoolctl==3.5.0 transformers==4.46.3 tokenizers==0.20.3 safetensors==0.8.0
.\.venv\Scripts\python.exe -m pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

精确实测版本以 outputs/formal/environment.json 为准；若本说明某项与环境清单不同，使用环境清单。预训练文件位于 models/bert_tiny，固定模型版本、来源和 SHA256 位于 provenance.json。随复现包附带该轻量模型，不需要推理时联网下载。

目录结构应为 problem2 和 E题 同级。赛题数据位于 E题/E题数据；复现包不包含原始赛题数据。不要加载来源不可信的 pickle/joblib 模型文件。

## 执行

在 problem2 目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_formal.py
.\.venv\Scripts\python.exe formal_predict.py
```

第二条命令重新读取冻结参数、训练特征映射和预训练模型，校验哈希，并生成重载预测。它不读取标签、不重新拟合标准化或词表，也不检索原始完整样本补回缺失。

已核验 30 条重载预测的类别全部一致，强度最大绝对差约 1.04×10⁻⁷，概率最大绝对差约 3.17×10⁻⁸；浮点编码的批次/缓存路径可能引起微小差异，核验采用 10⁻⁶ 的绝对容差，不声称逐字节一致。

要从头复现实验，应复制项目到新的实验目录或将 configs/formal_protocol.json 的 output 改为新目录；训练及评价函数各自拒绝覆盖已经冻结/完成的结果。

```powershell
.\.venv\Scripts\python.exe formal_run.py train
.\.venv\Scripts\python.exe formal_run.py evaluate
.\.venv\Scripts\python.exe formal_analysis.py
```

默认目录为 outputs/formal。新实验可对 formal_run.py 指定 `--output outputs/formal_new`，分析和预测相应指定 `--run-dir outputs/formal_new`。保留旧的冻结实验作为独立结果，不能根据测试表现反复调参却继续声称测试独立。

## 组件接口

完整性核验可运行 `python formal_verify.py`。本地实验保留了逐样本预测和掩码日志，可加 `--full-logs` 重算全部指标并检查区间嵌套。为满足附件 50 MB 限制，复现包不包含大体积逐样本日志；从头运行实验可重新生成。核验不重新训练或选择模型。

| 组件 | 位置 | 替换约束 |
|---|---|---|
| Dataset 与原始掩码 | aligned_dataset.py | 返回词元、音视频、extent 与 observed 掩码；不伪造无标签目标 |
| 连续缺失区间 | pipeline.py / formal_run.scenario | 保留原序列位置，记录实际比例，确保训练/验证/测试同口径 |
| 文本语义 | formal_features.SemanticEncoder | encode(samples) 返回固定维度语义向量，编码前遮蔽输入，缓存键包含实际掩码 |
| 稀疏词组 | formal_features.token_ngrams / FeatureMap.lexical | 不跨缺失位置生成 bigram，只在训练集 fit |
| 音视频统计 | formal_features.temporal_summary | 返回统计值及逐统计量有效性；缺失统计不能参与均值方差估计 |
| 融合 | formal_features.combine | 返回同一稀疏矩阵接口；控制模态、语义和可用比例消融 |
| 分类/回归 | formal_model.DualModel | fit(blocks,classes,intensities,weights)，predict(blocks) 返回三类概率和强度 |
| 集成 | formal_model.ensemble | 同一顺序平均，不能根据专项数据选择种子 |
| 导出 | formal_predict.py | 根据官方模板替换列名/ID 映射，数值结果保持一致 |

配置集中了搜索网格、缺失比例、训练/评价随机种子、预训练修订版本和 bootstrap 规则。更换语义编码器或统计特征必须重新拟合 FeatureMap 与监督模型，不能直接沿用旧系数。

## 实验设计

全量训练 3395 条；验证 728 条；选择完成后测试 727 条。训练使用完整输入加三份局部缺失增强，每条原始样本的四个版本各权重 1/4。预处理只用完整训练样本拟合。三个增强种子分别训练后平均预测。

18 组参数配置由六个固定验证情景的 MAE+(1−Macro-F1) 均值选择。消融使用与主模型相同的超参数和种子17，应对照 proposed_seed17，而不是对照三种子集成结果来归因模块贡献。

独立测试共有169个情景：完整输入1个，首/中/尾各28个，随机位置3个种子各28个。随机重复先按条件平均，所有112个缺失条件等权汇总。bootstrap 按 video_id 重采样，保留同一视频的多个片段相关性。

## 尚需如实注明的事项

共同 extent 来自原始 attention，全零音视频行视作不可用，是经审计支持的处理假设；缺失长度不是精确秒数。tokenizer 已在全部训练和验证序列验证一致。专项不存在真实标签，也缺少原始样本 ID；结果通过 source_file 和文件内索引追踪，正式模板如另有要求只更换导出层。

未经结果支持，不保证融合必然超过文本单模态，也不保证某项改动显著有效。完整解答会按实测数值报告成功与不足。正式预测不使用问题 1 或问题 3 的输出。

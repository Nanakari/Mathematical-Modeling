# 问题 2

**BERT-Mini与时序门控融合：** [TEMPORAL_README.md](TEMPORAL_README.md)。比较Tiny/Mini纯文本、掩码音视频时序编码及MAG式门控融合，每种结构两个固定种子，结果位于 `outputs/temporal_v4`。以实际报告判断收益，保留旧版模型。

**Accuracy选模与BERT-Tiny微调：** [ACCURACY_README.md](ACCURACY_README.md)。新增类别权重对照、以完整验证Accuracy选模，以及同结构冻结/微调实验。独立结果位于 `outputs/accuracy_v3`；目标是否达到及其他指标取舍见该目录报告，不自动替换第一轮正式模型。

**第二轮优化入口：** [REFINEMENT_README.md](REFINEMENT_README.md)，涵盖回归改进、中性识别、分任务融合与增强权重调整。采用训练内视频分组交叉验证，冻结后报告官方验证结果；本轮不重新评价官方测试集。结果位于 `outputs/refinement_v2`，与以下第一轮正式实验分别保存。

第二轮未验证出稳定收益，保留为实验候选；默认模型与专项预测仍使用第一轮正式版本。

**正式全量实验入口：** [FORMAL_README.md](FORMAL_README.md)。已完成 3395 条训练、728 条验证、727 条独立测试，包含 3 个训练种子、169 个测试情景、5 项消融、视频分组 bootstrap 和附件 3 的 30 条预测。完整解答位于 [问题2_完整解答.md](outputs/formal/问题2_完整解答.md)，公开仓库保留文字解答与聚合指标；未上传含逐样本测试目标的本地归档。各组件的替换接口和实测局限见正式说明。

## 历史预实验记录

以下保留前期准备过程及其当时状态；正式版本已完成统一预训练文本编码与全量实验，以以上入口为准。

本目录实现了数据体检、局部连续缺失模拟、轻量模型、评价与小规模实验，用于在团队共同规则确定前完成可复用的准备工作。当前版本是工程预实验，不是最终竞赛方案。

**完整流程入口：** `workflow.py` 与 `configs/small.json`。已完成 512 条训练、128 条验证样本的三组对照训练，每组评价 85 个缺失情景；生成图表、错误表和附件 3 的 30 条草稿预测。先阅读 `docs/完整流程与替换接口.md`，本次结果见 `outputs/end_to_end_small/RESULTS.md`。模型重新加载后的专项预测与原输出完全一致。

```powershell
.\.venv\Scripts\python.exe workflow.py run --config configs/small.json --output outputs/my_new_run
.\.venv\Scripts\python.exe workflow.py predict --config configs/small.json
.\.venv\Scripts\python.exe workflow.py package --config configs/small.json
```

后续已补充 `aligned_dataset.py`：附件 2 和附件 3 统一读取 text_bert 的 Dataset。用法见 `docs/Dataset使用说明.md`。4850 条训练/验证/测试样本与 30 条专项样本均已校验。旧连续 text 基线保持原样；统一文本编码器的训练尚未执行。

最新补充：`pipeline.py`、`scenarios.py`、`torch_training.py` 已连接标准化、词元连续缺失、固定验证场景及 CPU 训练循环；使用随机词元嵌入验证，不是最终 BERT 编码方案。独立 `.venv` 已安装 CPU PyTorch。运行及边界见 `docs/训练基础设施说明.md`。完整测试套件 15 项通过，包括实际 DataLoader 批处理和断点恢复一致性。

## 已完成

- 附件 2 对齐、非对齐两个版本的全量只读检查。
- 附件 3 全部 60 个文件的字段与接口检查。
- 区分有效位置、原始可用位置和人工遮蔽位置的连续缺失模拟器。
- 单模态及多模态组合、首部/中部/尾部/随机区间、多区间、对齐同步遮蔽。
- 基于 NumPy 的可训练投影、掩码均值池化、拼接、三分类与强度双输出模型；解析梯度通过数值核验。
- 可运行的均值池化＋逻辑回归/岭回归基线，支持普通训练和缺失增强训练。
- Accuracy、Macro-F1、Weighted-F1、各类别 F1、MAE、Pearson，逐样本预测和类别概率。
- 85 个输入情景的小规模运行、曲线和运行记录。
- 实验计划、论文方法草稿和待确认事项。

## 运行

已使用 `D:\anaconda3\python.exe` 成功运行。无需 PyTorch，无需下载模型或其他数据。

在 PowerShell 中执行：

```powershell
Set-Location 'problem2-project'
python -m unittest -v test_p2.py
python audit.py
python run_pilot.py
```

`audit.py` 默认逐个读取两套特征。非对齐文件约 2.90 GB，需留出额外内存；若内存不足，可以先执行 `python audit.py --versions aligned_50.pkl`。原始 pickle 仅限加载可信赛题来源。

`run_pilot.py` 默认从官方训练集和验证集分别按固定种子选取 256、96 条样本，索引选择不使用标签；不会重新划分数据。若以后决定运行全量预实验：

```powershell
python run_pilot.py --train-limit 3395 --valid-limit 728 --output outputs/full_pilot
```

这只是样本量扩大，仍需先解决下面列出的口径问题，不会自动成为正式实验。

## 文件入口

| 文件 | 用途 |
|---|---|
| `audit.py` | 数据与附件 3 接口检查 |
| `p2.py` | 读取、候选掩码、缺失模拟、指标、轻量双头模型 |
| `test_p2.py` | 不变性、梯度、可复现性等测试 |
| `run_pilot.py` | 基线拟合、缺失实验、预测日志与绘图 |
| `config.json` | 暂定特征版本、种子、缺失比例和位置等 |
| `docs/数据检查与待确认事项.md` | 实际数据发现及影响 |
| `docs/实验计划.md` | 正式实验设计草案 |
| `docs/论文方法草稿.md` | 可扩展的方法部分 |
| `outputs/*_audit.json` | 两个版本全量检查结果 |
| `outputs/attachment3_schema.json` | 每个专项文件的字段、缺失项、SHA256 |
| `outputs/pilot/metrics.csv` | 两种训练方式共 170 行评价结果 |
| `outputs/pilot/validation_predictions.csv` | 16320 行逐样本预测及类别概率 |
| `outputs/pilot/mask_records.jsonl` | 每个样本的遮蔽区间与实际比例 |
| `outputs/pilot/run_manifest.json` | 样本 ID、环境版本、参数与限制 |
| `outputs/pilot/pilot_models.pkl` | 两个基线的标准化器和模型参数 |
| `outputs/pilot/pilot_curves.png` | 小规模性能曲线 |

## 当前配置的边界

1. 对齐版采用 `text_bert[:,1,:]` 作为三模态共同有效范围的候选掩码，尚未证明其特殊词元与音视频位置完全一一对应。有效范围内的全零行暂作不可用，不能据此声称它们都是人工缺失。
2. 非对齐版音视频依照长度字段构造候选有效范围。视觉长度字段与实际非零位置不完全一致，详见检查报告；当前已完成的预实验仅使用对齐版。
3. 主实验模拟器遮蔽比例小于 1，保留至少一个未遮蔽的有效位置；对于原先有可用信息的模态，还保留至少一个可用位置。必要时缩短遮蔽区间，实际比例写入日志。原本完全无可用信息的模态保持原状，程序不删除这类样本。
4. 分类输出由分类器决定，回归输出截断到 [-3,3]；两者暂未强制一致，不用临时阈值合并。0 为中性，数字编码已核实为 0=负向、1=中性、2=正向。
5. 标准化仅拟合当前训练输入；验证集不参与拟合。两种基线各自拟合自己的标准化器，因此当前增强对比也包含标准化统计量改变的影响，正式消融需固定这项因素。
6. 轻量 NumPy 模型用于结构与梯度验证，预实验实际训练的是逻辑回归和岭回归。`hidden_dim` 和两个损失权重预留给 NumPy 模型，不控制基线；基线参数完整记录在 manifest 中。
7. 不支持的掩码策略会报错，不会仅更改配置名称而静默沿用旧规则。其他策略需要实现后再验证。
8. 附件 3 尚不兼容连续 `text` 特征接口，未生成专项提交预测。没有用标签或匹配回原始完整样本补回缺失内容。
9. 原始数据没有修改，没有删除样本；官方 test 只做只读结构与一致性检查，没有计算模型性能或据其调参。

## 已验证结果

6 项单元测试通过。预实验完成 256 条训练、96 条验证、85 种输入情景，生成两种训练方式的结果。当前单种子、小样本和均值池化基线只证明流程可运行，不能支持最终鲁棒性结论；也不能将序列比例解释成真实缺失秒数。

源代码与模型较小，但日志、原始数据、审计结果不应直接全部加入竞赛附件。最终按赛题 50 MB 限制另行整理提交包。

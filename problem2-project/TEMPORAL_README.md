# BERT-Mini 时序编码与门控融合

本轮输出至 `outputs/temporal_v4`，保留所有已有结果。以完整输入的负向、中性、正向三分类验证Accuracy为主要目标，同时报告Macro-F1、MAE、Pearson、中性召回率和局部缺失表现。是否达到0.70以实际报告为准。

## 运行

在problem2目录使用原有 `.venv`，依赖沿用 `requirements-formal.txt`：

```powershell
.\.venv\Scripts\python.exe temporal_prepare.py
.\.venv\Scripts\python.exe -m unittest -v test_temporal.py
.\.venv\Scripts\python.exe temporal_run.py run
.\.venv\Scripts\python.exe temporal_run.py predict
.\.venv\Scripts\python.exe temporal_verify.py
```

`temporal_prepare.py`只下载Google通用预训练BERT-Mini的固定修订版，转换为safetensors并记录源文件及转换文件的SHA256。已存在并且通过哈希核验时不会重新下载。Mini词表与已核验Tiny词表逐字节相同，并再次检查官方train、valid全部词元序列。

`run`拒绝覆盖已完成的实验。被中断时重跑相同命令：已完成候选校验哈希后跳过；未完成候选加载最近完整轮次的模型、优化器、Torch随机状态、最佳记录与早停计数，继续下一轮。每轮视图抽样由种子和轮次确定，确保恢复后相同。保存使用同目录临时文件后替换，避免中断留下半个检查点。恢复等价性有单元测试。

如需改协议、模型代码或重做实验，复制 `configs/temporal_protocol.json` 并更换output，然后使用 `--config configs/新配置.json`。预测和报告也支持该参数。单独生成报告用 `temporal_run.py report`，不重新选模。

## 对照结构

| 名称 | 文本编码器 | 音视频 | 融合 |
|---|---|---|---|
| tiny_text | 2层128维BERT-Tiny | 不使用 | 纯文本 |
| mini_text | 4层256维BERT-Mini | 不使用 | 纯文本 |
| mini_temporal | 同上 | 每模态两层掩码时序卷积 | 文本和音视频池化后拼接 |
| mini_mag | 同上 | 相同掩码时序卷积 | 逐位置门控残差，再池化与拼接 |

所有新结构各用种子1729、2027训练，最多10轮，连续3轮未改善停止。分类等权、交叉熵加0.2倍SmoothL1回归损失、batch_size=64、dropout=0.2、AdamW weight_decay=0.01、编码器学习率5e-5、头部学习率5e-4、梯度裁剪范数1；4线程CPU运行。其余细节以配置文件为准。

每个原始训练样本保留完整版本及三个局部连续缺失版本（10%、30%、50%）。每轮抽取一份，完整输入概率0.6，三个缺失版本共享0.4；保持每个原始样本每轮出现一次。同一seed与epoch的各结构使用同样采样及顺序。

各Mini模型的公共参数按相同构造顺序初始化。时序卷积隐维32、卷积核3，并携带位置与观测指示；每层对缺失位置清零，最终只池化观测位置。门控由对应位置的文本和音视频表示决定；将残差范数限制在文本表示范数的0.1倍。音视频有单独汇总支路，文本缺失时不会被一并删除。模型维数从编码器配置读取，没有把Tiny的128维写死。

这是借鉴[MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer)思路的独立实现：门控在BERT输出后作用，未复现原论文的内部注入方式，不应写成原版MAG-BERT复现成绩。

## 数据与选择协议

仅在官方train的3395条样本上拟合参数和音视频标准化。valid的728条样本用于选轮次与结构；官方test不评价、不参与选择。原始赛题数据保持不变。归一化只使用实际观测的训练位置；音视频与文本掩码独立，补零不会被视为观测。

每个种子按完整验证Accuracy保存最佳轮次，同分时依次比较Macro-F1与MAE。每个结构固定平均两个种子的概率和裁剪后的强度，不挑表现较好的种子。按集成完整验证Accuracy选择结构；若不能优于accuracy_v3，则继续保留v3候选。结构冻结后才跑32个诊断情景：完整输入1个、30%缺失下7种模态组合×4种位置，以及全模态随机10%、50%、70%。缺失诊断不参与重选。

对照v3预测复用其已保存的相同样本、相同掩码场景结果，并核验样本ID、目标、文件哈希。v3是单种子统计融合模型，新结构是双种子时序模型，因此不能把新旧差异全部归因于某一个模块；严格的结构比较应看本轮四组及各自单种子结果。

验证集已在之前多轮使用，当前结果仍是选模分数，不能称为新的独立测试成绩。两个种子只提供有限的随机性检查。按video_id的bootstrap是选模后的描述性区间，不校正选择偏差。

`tiny_text`和`mini_text`是纯文本消融对照。如果全体候选中最高分来自纯文本，说明本轮融合尚未证明分类收益；其候选CSV只用于研究比较，不能把它描述为题目要求的多模态模型改进。原正式多模态模型始终保留，是否升级应同时依据题目要求及分类、回归、缺失表现。

## 输出文件

- `时序融合优化报告.md`：结果、对照、错误分析与局限。
- `candidate_summary.json`、`candidates/*_history.json`：每种子结果及每轮指标。
- `architecture_comparison.json`：单种子均值、样本标准差、双种子集成结果。
- `frozen_selection.json`、`model_manifest.json`：冻结选择与参数哈希。
- `validation_metrics.json`、`validation_predictions.joblib`：5组模型×32情景，含v3参照。
- `validation_errors.csv`、`validation_diagnostics.png`：最终候选的验证错误记录。
- `附件3_时序融合候选预测.csv`与`附件3_时序融合重载预测.csv`：30条专项结果及独立重载核验。
- `verification.json`：完整指标、选择、哈希与预测核验。

当前实验目录不是直接提交的压缩包。候选和准备缓存用于本地审计、恢复；仅选定的双种子Mini全精度参数也可能超过50MB。最终参赛打包需核验压缩精度/量化/蒸馏后的结果，并与问题1、3一起核算总大小，不能直接提交全部实验文件。

本轮尚未加入完整—缺失蒸馏。如果基础表示和融合没有验证收益，不应直接叠加蒸馏后宣称改进有效。

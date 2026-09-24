# 类别权重对照 Accuracy选模与BERT-Tiny微调

本轮独立输出至 `outputs/accuracy_v3`，不替换 `outputs/formal` 中的正式结果。目标是保留中性类别的三分类完整输入验证 Accuracy ≥ 0.70；是否达到以实际报告为准。专项附件3没有标签，不能计算专项Accuracy。

## 入口和运行

在problem2目录，沿用已安装的 `.venv` 和 `requirements-formal.txt`：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_accuracy.py
.\.venv\Scripts\python.exe accuracy_run.py run
.\.venv\Scripts\python.exe accuracy_run.py predict
.\.venv\Scripts\python.exe accuracy_verify.py
```

CPU运行，4个计算线程。预训练权重来自原有 `models/bert_tiny`，无需下载。`run` 拒绝覆盖已经完成的实验；训练意外中断时可以重跑相同命令，跳过已完成且哈希匹配的神经网络候选，未完成候选从相同初始化重新训练。线性对照会重新计算。协议或源码变化必须使用新输出目录，不能混合候选结果。

从头复现时复制 `configs/accuracy_protocol.json`，仅将output改为例如 `outputs/accuracy_reproduction`，运行：

```powershell
.\.venv\Scripts\python.exe accuracy_run.py run --config configs/accuracy_reproduction.json
```

预测和报告阶段同样使用相应 `--config`。报告可通过 `accuracy_run.py report` 重新生成，不重新训练或选模。推理依赖原正式模型作为对照/可能的保留候选，须保留 `outputs/formal`、`models/bert_tiny` 和赛题数据原目录。当前不是独立移动即可运行的压缩提交包。

## 数据与选模边界

- 仅用附件2官方train的3395条样本监督训练，728条valid用于类别权重、C、学习率、训练轮次及最终候选选择。
- 复用原正式版仅在训练集拟合的特征映射，核验文件哈希及train/valid样本ID完全一致，检查video_id不跨train/valid。
- 不评价官方test，不以test标签选参。官方单个pickle包含所有划分，加载后仅提取train和valid。
- 验证集已经在之前实验中使用，本轮又用于多候选与轮次选择，其分数是选模结果，不是新独立测试成绩。分组bootstrap未校正选模偏差。
- 选模优先完整输入Accuracy，其次Macro-F1，再次更小MAE；仍完全相同则保留较早的候选/轮次。缺失诊断在选择冻结后执行，不据其结果重选模型。

## 对照设计

线性组沿用原BERT-Tiny冻结特征、TF-IDF、音视频统计、可用比例融合，以及每样本原始加三个缺失版本各1/4权重。分类类别权重为训练类别频率的负gamma次方，gamma为0、0.5、1，整体归一化使训练样本平均权重为1。C为0.5、2、8，共9组；回归保留旧seed17模型，隔离分类变化。gamma=1、C=0.5应复现旧seed17分类。

神经组使用可见词元的BERT均值池化、LayerNorm，与音视频统计的64维投影拼接，接分类与回归线性头。音视频沿用训练拟合的掩码统计和标准化。分类使用加权交叉熵，辅助回归使用0.2倍SmoothL1；强度仅在推理时裁剪至[-3,3]。原始token和observed掩码在编码前共同决定输入，禁止补回缺失文本。

3组冻结控制和6组微调候选使用同一初始化种子1729、同一结构、头部学习率5e-4和每轮同一采样顺序。微调编码器学习率为2e-5或5e-5，类别权重gamma为0、0.5、1；冻结时编码器保持eval状态。AdamW、weight_decay=0.01、dropout=0.2、batch_size=64、梯度范数上限1，最多12轮，连续4轮未改善停止。每轮每个原始样本抽取一个版本，完整概率0.6，三个固定缺失版本共享0.4。只运行一个初始化种子，不声称跨种子稳定。

冻结与微调神经组可用于分析编码器更新效果；线性与神经组还改变了预测头、辅助损失和增强采样，不能把它们的差异都归因于微调。

## 输出

- `Accuracy优化报告.md`：完整指标、类别权重与微调对照、验证混淆矩阵和错误分析。
- `linear_search.json`、`neural_search.json`：全部候选及其选择指标。
- `candidates/*_history.json`：每轮训练损失和验证指标；对应pt文件是该候选的最佳验证轮次。
- `frozen_selection.json`、`model_manifest.json`：冻结选择及可校验哈希。
- `validation_metrics.json`、`validation_predictions.joblib`：3组模型×32个诊断情景。
- `validation_errors.csv`：完整验证集逐样本错误。
- `附件3_Accuracy候选预测.csv`：30条专项候选结果；`predict`生成重载预测文件。
- `verification.json`：重算指标、参数更新、类别权重控制及重载结果检查。

候选保留以便审计，但所有候选参数总大小不适合直接作为竞赛附件。最终打包应只包含选定模型、必要预训练资产、源码和结果，再与问题1、3共同核算50MB总上限。

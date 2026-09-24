# 问题2第二轮优化与复现

本轮实现回归改进、中性识别、分任务融合、增强权重调整。结果入口为 `outputs/refinement_v2/问题2_第二轮优化报告.md`。旧正式实验保留在 `outputs/formal`。新旧版本的验证分数与旧测试分数不能跨数据划分直接比较。

**使用状态：V2为实验候选，未替代旧正式模型。** 三折选模分数小幅改善，但官方验证未复现提升，成对置信区间跨0。新版专项预测仅供研究对照，原默认预测仍为 `outputs/formal/附件3_问题2_预测结果.csv`。

## 模型与选参

沿用已核验的 aligned Dataset 和冻结 BERT-Tiny。全部3395条训练样本按原始视频分成三折，每折重新拟合词表、IDF与标准化。冻结编码器的纯前向计算可以缓存，但不共享拟合后的折内特征映射。训练集内32个固定评价情景，完整输入权重0.4，其他31个情景共享0.6。

分类比较24个基础模型与4个中性决策权重，共96个候选；回归比较60个基础模型与9个线性校准组合，共540个候选。含有候选选择的OOF分数属于选模分数。分类和回归分别按Macro-F1和MAE选择参数，不用测试集反馈。

新的分类输出列 `decision_score_0/1/2` 为中性权重调整后的归一化决策分数，不声称是校准后的概率。极性与强度仍由独立分支输出，不强制将中性强度置零或修改符号。

官方验证集728条只在冻结选择后用于报告对照；本轮不按验证结果再换配置。此前已使用过该验证集，且此前测试分析启发了本轮方向，因此本轮不能被描述为全新独立测试。官方测试集未在本轮重新评价。

## 执行

环境与第一轮相同，见 `requirements-formal.txt` 和 `outputs/formal/environment.json`。需要原始赛题目录 `E题` 与 `problem2` 同级。附带本地预训练模型，推理无需下载。

在 problem2 目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_refinement.py
.\.venv\Scripts\python.exe refinement_predict.py
.\.venv\Scripts\python.exe refinement_verify.py
```

从头复现时，复制 `configs/refinement_protocol.json` 并修改其中output为新的目录，例如outputs/refinement_reproduction。然后：

```powershell
.\.venv\Scripts\python.exe refinement_run.py select --config configs/refinement_reproduction.json
.\.venv\Scripts\python.exe refinement_run.py validate --config configs/refinement_reproduction.json
.\.venv\Scripts\python.exe refinement_analysis.py --run-dir outputs/refinement_reproduction
.\.venv\Scripts\python.exe refinement_predict.py --run-dir outputs/refinement_reproduction
.\.venv\Scripts\python.exe refinement_verify.py --run-dir outputs/refinement_reproduction
```

选择阶段可以在相同配置、相同源码下恢复已经完成的折。修改模型实现或协议后应使用新输出目录。验证阶段拒绝覆盖已完成的评价。训练只拟合官方训练集，最终没有将验证标签加入训练。

## 替换接口

| 模块 | 接口 | 约束 |
|---|---|---|
| 模态融合 | refinement_model.FUSIONS | 输出顺序固定；分类与回归分别记录所用配置 |
| 增强质量分配 | view_weights | 每条原始样本的完整与增强版本权重和为1 |
| 回归估计器 | fit_head(task='regression') | 支持稀疏矩阵与样本权重；不收敛时不接受结果 |
| 中性规则 | adjusted_probabilities | 不修改原始概率；只调整决策权重 |
| 强度校准 | calibrated_intensity | 在训练内OOF选择参数；输出范围[-3,3] |
| 分组与场景 | refinement_run.select / cases | 同视频不跨折；预处理每折独立拟合 |
| 专项导出 | refinement_predict.py | 只重载冻结模型；保留文件名与样本索引 |

消融保持其他参数不变，用于分析选定模型中的模块作用，不等同于给每个消融模型各自寻找最优参数。若改动未被选择或未改善指标，报告保留该结果。

复现包包含源码、选定模型、固定预训练权重、搜索记录和验证报告。大体积每折所有候选预测仅保留本地；执行select阶段可重新生成。原始赛题数据不进入压缩包。`refinement_package.py` 检查压缩包完整性并限制大小在50 MB内。

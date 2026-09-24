# 复杂场景下多模态情感识别

本仓库收录多模态情感预测项目的建模代码、实验说明、汇总结果和部分选定模型。当前主要完成 E 题问题2（局部模态缺失下的情感预测）实验；问题1的100条原始视频特征提取与核验、问题3的可解释预测交付尚未完成，因此本仓库不代表整道 E 题的完整参赛解答。

## 目录

- `problem2-project/`：数据集接口、局部连续缺失模拟、训练与评价代码、BERT-Tiny/BERT-Mini实验、测试和方法说明。
- `problem2-project/outputs/formal/`：第一轮正式问题2的解答、汇总指标、附件3预测和基准模型参数。
- `problem2-project/outputs/accuracy_v3/`：类别权重对照、Accuracy选模、BERT-Tiny微调结果及选定候选模型。
- `problem2-project/outputs/temporal_v4/`：BERT-Mini时序与门控融合的汇总实验结果，以及冻结模型后的官方测试集汇总评估。
- `problem2-project/outputs/refinement_v2/`：第二轮优化报告和汇总结果。

## 数据和运行

依据竞赛数据分发规则，本仓库不包含附件1至附件4的原始视频、特征pickle或标签文件。请从官方渠道获取赛题数据，放入 `E题/E题数据/`，再按 `problem2-project/*_README.md` 和 `problem2-project/configs/` 中的说明安装依赖、选择协议并运行。代码使用相对于仓库根目录的 `E题/E题数据/` 路径定位输入。

公开版本保留主要报告、汇总指标和选定的Accuracy候选模型；出于数据保护和体积考虑，不包含逐样本官方测试真值/预测文件、完整测试日志、全部候选检查点或本地虚拟环境。尤其 `outputs/temporal_v4/official_test/` 只公开聚合指标与报告，不发布逐样本标签和ID。

完整问题2的文字解答见 [`problem2-project/outputs/formal/问题2_完整解答.md`](problem2-project/outputs/formal/问题2_完整解答.md)。BERT-Tiny准确率实验见 [`problem2-project/outputs/accuracy_v3/Accuracy优化报告.md`](problem2-project/outputs/accuracy_v3/Accuracy优化报告.md)，时序融合实验见 [`problem2-project/outputs/temporal_v4/时序融合优化报告.md`](problem2-project/outputs/temporal_v4/时序融合优化报告.md)。各报告区分验证选择分数、测试集聚合结果和模型局限。

本仓库公开且不附带额外软件许可。所用预训练模型和竞赛数据各自的许可、引用及使用条件仍适用；来源与版本记录见 `problem2/models/bert_tiny/`、实验配置和报告。

# MiniLM-L6-H384 FP16 推理模型

这是问题 2 的单模型完整离线推理包，使用 MiniLM-L6-H384 文本编码器与轻量音视频时序编码器。压缩包内含模型权重、tokenizer、标准化参数和推理代码，不含训练、验证或测试数据及预测明细。

- ZIP：`minilm_l6_h384_fp16.zip`
- ZIP 大小：42,976,635 bytes；展开大小：47,282,274 bytes（小于 50,000,000 bytes）
- 参数量：23,001,800；权重为 FP16
- ZIP SHA256：`bb8db3ea8673fca87bfc0dce0f28d9dd563df966ae513894b221f1cab7b061b2`
- 验证集（728 条）Accuracy：0.64698，Macro-F1：0.62462，Neutral Recall：0.48370，MAE：0.60969，Pearson：0.64354
- 31 个缺失场景平均 Accuracy：0.61047
- 包已通过独立离线重载与推理一致性验证；场景数据见 `validation_v4_32.json`

预训练 backbone 来自 [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)，该项目说明许可为 MIT；本模型采用其每隔一层抽取后的六层版本。参见上游模型卡与包内来源说明。

`config.json` 保存训练配置。此候选于 2026-09-25 冻结；其后曾做 official test 评估。这里仅列出固定验证集指标，未收录 official test 分数、数据或预测文件。

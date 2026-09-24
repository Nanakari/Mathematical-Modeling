# aligned Dataset 使用说明

建议问题 2 优先采用 aligned，并在训练、验证与专项推理中统一使用 text_bert 输入同一个文本编码器。aligned_dataset.py 已提供两类附件共享的 Dataset 接口；它没有训练文本编码器，也不意味着以前用连续 text 训练的基线可以直接用于专项推理。

## 接口

返回 input_ids、attention_mask、token_type_ids（int64，长度 50），audio（float32，50×74）、vision（float32，50×35），各模态布尔可用掩码、候选有效范围 extent_mask、sample_id，以及 has_label。有标签数据额外返回 class_label 和 regression_label；专项数据不伪造标签。

类实现 __len__ 和 __getitem__，返回 NumPy 数组，可按 PyTorch map-style 数据协议交给 DataLoader 的默认 collate。后续已在项目独立 `.venv` 安装 CPU PyTorch，并通过实际 DataLoader 批处理、最后不足一批、无标签推理等集成测试，详见训练基础设施说明。

```python
from pathlib import Path
from torch.utils.data import DataLoader
from aligned_dataset import AlignedDataset

root = Path(r'E题/E题数据')
train = AlignedDataset.from_attachment2(
    root / '附件2-数据集特征文件' / 'aligned_50.pkl', 'train')
valid = AlignedDataset.from_attachment2(
    root / '附件2-数据集特征文件' / 'aligned_50.pkl', 'valid')
special = AlignedDataset.from_attachment3(
    root / '附件3-模态缺失特征样本' / '对齐版本')

train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False, num_workers=0)
special_loader = DataLoader(special, batch_size=16, shuffle=False, num_workers=0)
```

Windows 初期使用 num_workers=0，避免大数组随进程复制和脚本入口问题。from_attachment2 每次构造会暂时载入完整 pickle；内存紧张时一次读取原始字典，用 `AlignedDataset(AlignedDataset._select(data['train']), labeled=True)` 等方式共享已读数据，构建各划分后释放原始字典。

## 数据处理顺序

1. 检查数值、整数词元、形状、标签映射与 attention 是否前缀连续。
2. 在标准化前确定音视频全零行的可用性。全零只表示当前处理策略下不可用，不判断缺失原因。
3. 从原始 attention 保存候选有效范围。对现有全部文件检查后，未发现音视频非零位置超出该范围；这支持该候选策略，但仍不能证明全部特殊词元和音视频时序的物理对应关系。
4. 如需增强，在候选范围内生成连续缺失区间，保存独立的 missing mask，更新相应模态 observed mask。不要因文本遮蔽而缩短其他模态的时间范围。
5. 文本在编码前移除被遮蔽位置的信息，再经统一文本编码器处理。不要只把完整文本的缓存 BERT 输出部分清零，就声称其他位置完全不含被遮蔽文本的信息。
6. 音视频标准化统计量仅从训练集可用位置拟合；转换后再次把不可用位置归零。文本整数 ID 不做数值标准化。
7. 编码后的文本在池化和融合时仍使用 text_pool_mask。attention_mask 为 0 不保证该位置输出隐状态就是零。

## 特殊词元与文本缺失

实际 token 序列常以 101 开始、102 结束，且存在 ID 100。但这些数值本身不足以证明完整 tokenizer 身份或缺失生成规则。不要直接认定 ID 100 都是缺失；可能只是词表中的未知词标记。

默认 text_pool_mask 不擅自删除特殊 ID。确认 tokenizer 后，可传入 `special_token_ids=(101, 102)` 排除池化中的对应特殊词元；它们通常仍保留在编码器的 attention 内。

若以后出现 attention 中间断裂、词元与 attention 不一致、有效范围外音视频非零，Dataset 会报错。此时需独立长度/有效范围元数据，不能静默用 attention 的和截断三模态。

具体 BERT 编码器必须先确认词表与提供的 token IDs 相容。即使两个编码器都输出 768 维，也不能把其输出当成同一特征空间。不要采用训练用提供的 text、推理临时换任意 BERT 生成 text 的混合方案。

## 当前验证

3 项新增测试通过；附件 2 全部 4850 条 aligned 样本及附件 3 全部 30 条 aligned 样本成功读取和逐样本校验。检查结果见 outputs/aligned_dataset_validation.json。专项 sample_id 是文件名加文件内索引，用于追踪，尚不是经确认的官方提交编号。

## 官方接口参考

- PyTorch map-style Dataset 与默认批处理：https://docs.pytorch.org/docs/stable/data.html
- BERT 的 input_ids、attention_mask、token_type_ids：https://huggingface.co/docs/transformers/model_doc/bert

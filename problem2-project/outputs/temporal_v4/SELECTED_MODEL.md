# Selected model: Mini-MAG two-seed ensemble

The selected model is `mini_mag`, a BERT-Mini encoder with mask-aware audio/video temporal encoders and MAG-inspired gated residual fusion. The reported ensemble averages predictions from both fixed seeds; retain both checkpoints to reproduce it.

- `candidates/mini_mag_seed1729.pt` — best epoch 4; SHA256 `c54b981da4730d961f64e0a2a4160068ee0a2043a8da172908a23770f7aaf81a`
- `candidates/mini_mag_seed2027.pt` — best epoch 3; SHA256 `d38309449b541b027c3ee7f16d344c1f5a1370ae555f32c7fa63c91c3803b590`
- `models/bert_mini/` — shared BERT-Mini config, tokenizer vocabulary, and base weights used to construct the encoder before loading each fine-tuned checkpoint.

Each fine-tuned checkpoint is 45,449,255 bytes. The complete model weights include both checkpoints and the 44,690,024-byte base BERT-Mini weights. The inference implementation is in `temporal_model.py` and `temporal_run.py`; running it also requires the competition data and the dependencies described in `TEMPORAL_README.md`.

The frozen selection reached validation Accuracy 0.6264. A later one-time evaluation reported official test Accuracy 0.6547; the test score was not used to select the model. Detailed metrics and limits are in `时序融合优化报告.md` and `official_test/官方测试集评估报告.md`.

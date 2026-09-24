"""Masked local BERT-Tiny fine-tuning; all supervision uses official train."""
import numpy as np
import torch
from torch import nn
from transformers import AutoModel


def class_weights(labels, gamma):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=3)
    if (counts == 0).any():
        raise ValueError('All three training classes are required')
    weights = (len(labels) / (3.0 * counts)) ** gamma
    # Expected weight per original training sample is one.
    return weights / np.average(weights, weights=counts)


def tensors(samples, mapper):
    # Use only train-fitted AV statistics. BERT is evaluated inside the network.
    blocks = mapper.blocks(samples, np.zeros((len(samples), 128), dtype=np.float32))
    attention = np.stack([s['attention_mask'].astype(bool) & s['text_observed'] for s in samples])
    ids = np.stack([np.where(a, s['input_ids'], 0) for a, s in zip(attention, samples)])
    segments = np.stack([np.where(a, s['token_type_ids'], 0) for a, s in zip(attention, samples)])
    pool = np.stack([s['text_pool_mask'] & a for s, a in zip(samples, attention)])
    av = np.concatenate([blocks['audio'], blocks['vision'], blocks['fractions']], axis=1)
    return {'ids': torch.tensor(ids, dtype=torch.long),
            'attention': torch.tensor(attention), 'segments': torch.tensor(segments, dtype=torch.long),
            'pool': torch.tensor(pool), 'av': torch.tensor(av, dtype=torch.float32)}


class TinyDual(nn.Module):
    def __init__(self, model_dir, *, frozen=False, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        self.frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.text_norm = nn.LayerNorm(self.encoder.config.hidden_size)
        self.av = nn.Sequential(nn.Linear(657, 64), nn.GELU(), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size + 64, 3)
        self.regressor = nn.Linear(self.encoder.config.hidden_size + 64, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, ids, attention, segments, pool, av):
        # Defense in depth: unseen token values never reach embedding lookup.
        ids = torch.where(attention, ids, 0)
        segments = torch.where(attention, segments, 0)
        safe = attention.clone()
        safe[~safe.any(1), 0] = True
        h = self.encoder(input_ids=ids, attention_mask=safe.long(), token_type_ids=segments).last_hidden_state
        observed = pool & attention
        pooled = torch.where(observed.unsqueeze(-1), h, 0).sum(1) / observed.sum(1).clamp_min(1).unsqueeze(-1)
        pooled = self.text_norm(pooled)
        pooled = torch.where(observed.any(1).unsqueeze(-1), pooled, 0)
        representation = self.dropout(torch.cat([pooled, self.av(av)], dim=1))
        return self.classifier(representation), self.regressor(representation).squeeze(-1)


@torch.inference_mode()
def predict(model, batch, batch_size=128):
    model.eval()
    probs, values = [], []
    for start in range(0, len(batch['ids']), batch_size):
        logits, value = model(**{k: v[start:start + batch_size] for k, v in batch.items()})
        probs.append(logits.softmax(-1).numpy())
        values.append(value.clamp(-3, 3).numpy())
    return np.concatenate(probs), np.concatenate(values)

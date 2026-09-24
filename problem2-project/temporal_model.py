"""Mask-aware temporal CNN and MAG-inspired post-BERT residual fusion.

This is a small adaptation, not a reproduction of the original MAG-BERT.
Audio/video masks are independent of text attention, preserving evidence when
text is missing. Original aligned sequence positions are never compacted.
"""
import numpy as np
import torch
from torch import nn
from transformers import AutoModel


def pool_observed(x, mask):
    return torch.where(mask.unsqueeze(-1), x, 0).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)


def tensors(samples, standardizer):
    items = [standardizer.transform(s) for s in samples]
    attention = np.stack([s['attention_mask'].astype(bool) & s['text_observed'] for s in items])
    result = {
        'ids': torch.tensor(np.stack([np.where(a, s['input_ids'], 0) for a, s in zip(attention, items)]), dtype=torch.long),
        'attention': torch.tensor(attention),
        'segments': torch.tensor(np.stack([np.where(a, s['token_type_ids'], 0) for a, s in zip(attention, items)]), dtype=torch.long),
        'pool': torch.tensor(np.stack([s['text_pool_mask'] & a for s, a in zip(items, attention)])),
    }
    for modality in ['audio', 'vision']:
        result[modality] = torch.tensor(np.clip(np.stack([s[modality] for s in items]), -5, 5), dtype=torch.float32)
        result[modality + '_mask'] = torch.tensor(np.stack([s[modality + '_observed'] for s in items]))
    return result


class MaskedTemporal(nn.Module):
    def __init__(self, dimension, width, dropout):
        super().__init__()
        self.project = nn.Linear(dimension + 2, width)
        self.layers = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        values = torch.where(mask.unsqueeze(-1), values, 0)
        position = torch.linspace(0, 1, values.shape[1], device=values.device).expand(*mask.shape)
        x = self.project(torch.cat([values, mask.unsqueeze(-1).float(), position.unsqueeze(-1)], dim=-1))
        x = torch.where(mask.unsqueeze(-1), x, 0)
        for conv in self.layers:
            x = self.dropout(torch.nn.functional.gelu(conv(x.transpose(1, 2)).transpose(1, 2)))
            x = torch.where(mask.unsqueeze(-1), x, 0)
        return x


class TemporalDual(nn.Module):
    def __init__(self, model_dir, stage, width=32, dropout=.2, beta=.1):
        super().__init__()
        if stage not in ['tiny_text', 'mini_text', 'mini_temporal', 'mini_mag']:
            raise ValueError('Unknown architecture')
        self.stage, self.beta = stage, beta
        self.encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        dimension = self.encoder.config.hidden_size
        # Construct in the same order for common-parameter initialization across
        # the three Mini variants. Unused branches do not participate in loss.
        self.audio_encoder = MaskedTemporal(74, width, dropout)
        self.vision_encoder = MaskedTemporal(35, width, dropout)
        self.audio_shift = nn.Linear(width, dimension, bias=False)
        self.vision_shift = nn.Linear(width, dimension, bias=False)
        self.audio_gate = nn.Linear(dimension + width, dimension)
        self.vision_gate = nn.Linear(dimension + width, dimension)
        nn.init.constant_(self.audio_gate.bias, -2.)
        nn.init.constant_(self.vision_gate.bias, -2.)
        self.text_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dimension + 2 * width, 3)
        self.regressor = nn.Linear(dimension + 2 * width, 1)
        self.width = width

    def forward(self, ids, attention, segments, pool, audio, vision, audio_mask, vision_mask):
        ids = torch.where(attention, ids, 0)
        segments = torch.where(attention, segments, 0)
        safe = attention.clone(); safe[~safe.any(1), 0] = True
        h = self.encoder(input_ids=ids, token_type_ids=segments, attention_mask=safe.long()).last_hidden_state
        text_mask = pool & attention
        h = torch.where(text_mask.unsqueeze(-1), h, 0)
        if self.stage.endswith('text'):
            av = h.new_zeros((len(h), self.width * 2))
        else:
            a = self.audio_encoder(audio, audio_mask)
            v = self.vision_encoder(vision, vision_mask)
            av = torch.cat([pool_observed(a, audio_mask), pool_observed(v, vision_mask)], dim=-1)
            if self.stage == 'mini_mag':
                ga = torch.sigmoid(self.audio_gate(torch.cat([h, a], dim=-1)))
                gv = torch.sigmoid(self.vision_gate(torch.cat([h, v], dim=-1)))
                shift = (ga * self.audio_shift(a) * audio_mask.unsqueeze(-1)
                         + gv * self.vision_shift(v) * vision_mask.unsqueeze(-1))
                ratio = (self.beta * h.norm(dim=-1) / shift.norm(dim=-1).clamp_min(1e-6)).clamp(max=1)
                h = h + ratio.unsqueeze(-1) * shift
        text = self.text_norm(pool_observed(h, text_mask))
        text = torch.where(text_mask.any(1).unsqueeze(-1), text, 0)
        representation = self.dropout(torch.cat([text, av], dim=-1))
        return self.classifier(representation), self.regressor(representation).squeeze(-1)


@torch.inference_mode()
def predict(model, batch, batch_size=128):
    model.eval(); ps, zs = [], []
    for start in range(0, len(batch['ids']), batch_size):
        logits, z = model(**{k: v[start:start + batch_size] for k, v in batch.items()})
        ps.append(logits.softmax(-1).numpy()); zs.append(z.clamp(-3, 3).numpy())
    return np.concatenate(ps), np.concatenate(zs)

"""Replaceable encoders, fusion and objectives behind stable interfaces."""
import copy
import importlib
import torch
from torch import nn


def construct(spec, registry):
    cfg = dict(spec)
    name = cfg.pop("name")
    # Custom components use an explicit module:factory entry point.
    if ":" in name:
        module, factory = name.split(":", 1)
        return getattr(importlib.import_module(module), factory)(**cfg)
    if name not in registry:
        raise ValueError(f"Unknown component {name}; supported: {list(registry)}")
    return registry[name](**cfg)


class TokenEmbedding(nn.Module):
    def __init__(self, hidden, vocab_size=32768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)

    def forward(self, b, mask):
        ids = torch.where(mask, b["input_ids"], 0)
        return self.embedding(ids)


class LocalBert(nn.Module):
    """Optional offline adapter. Tokenizer compatibility must be confirmed."""
    def __init__(self, hidden, model_path, tokenizer_verified=False, freeze=True):
        super().__init__()
        if not tokenizer_verified:
            raise ValueError("Confirm supplied token IDs/tokenizer before enabling local BERT")
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.freeze = freeze
        if freeze:
            self.encoder.requires_grad_(False)
        self.project = nn.Linear(self.encoder.config.hidden_size, hidden)

    def forward(self, b, mask):
        attention = b["attention_mask"].bool() & b["text_observed"]
        ids = torch.where(attention, b["input_ids"], 0)
        # Avoid undefined all-masked attention; artificial position is removed
        # from all downstream pooling by the original masks.
        safe_attention = attention.clone()
        safe_attention[~attention.any(1), 0] = True
        if self.freeze:
            self.encoder.eval()
        h = self.encoder(input_ids=ids, attention_mask=safe_attention.long(),
                         token_type_ids=torch.where(attention,b["token_type_ids"],0)).last_hidden_state
        return self.project(h)


class LinearSequence(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden)

    def forward(self, x, mask):
        x = torch.where(mask.unsqueeze(-1), x, 0.)
        h = torch.tanh(self.projection(x))
        return torch.where(mask.unsqueeze(-1), h, 0.)


class TemporalSequence(LinearSequence):
    def __init__(self, input_dim, hidden, kernel_size=3):
        super().__init__(input_dim, hidden)
        if kernel_size % 2 != 1:
            raise ValueError("Use odd kernel size to preserve sequence length")
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2)

    def forward(self, x, mask):
        h = super().forward(x, mask)
        h = torch.tanh(h + self.conv(h.transpose(1,2)).transpose(1,2))
        return torch.where(mask.unsqueeze(-1), h, 0.)


class ConcatFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.output_dim = hidden*3

    def forward(self, representations, fractions):
        return representations.flatten(1)


class GatedFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden+1, hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.output_dim = hidden

    def forward(self, representations, fractions):
        present = fractions > 0
        scores = self.score(torch.cat([representations,fractions.unsqueeze(-1)],-1)).squeeze(-1)
        weights = scores.masked_fill(~present,-1e4).softmax(-1)*present
        weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return (representations*weights.unsqueeze(-1)).sum(1)


TEXT_ENCODERS = {"embedding": TokenEmbedding, "local_bert": LocalBert}
SEQUENCE_ENCODERS = {"linear": LinearSequence, "temporal_conv": TemporalSequence}
FUSIONS = {"concat": ConcatFusion, "gated": GatedFusion}


class ModularModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        hidden = config["hidden"]
        self.text = construct({**config["text"],"hidden":hidden},TEXT_ENCODERS)
        self.audio = construct({**config["sequence"],"input_dim":74,"hidden":hidden},SEQUENCE_ENCODERS)
        self.vision = construct({**config["sequence"],"input_dim":35,"hidden":hidden},SEQUENCE_ENCODERS)
        self.fusion = construct({**config["fusion"],"hidden":hidden},FUSIONS)
        self.head = nn.Sequential(nn.Linear(self.fusion.output_dim,hidden),nn.Tanh())
        self.classifier = nn.Linear(hidden,3)
        self.regressor = nn.Linear(hidden,1)

    def forward(self,b):
        masks = [b["text_pool_mask"] & b["text_observed"] & b["attention_mask"].bool(),
                 b["audio_observed"],b["vision_observed"]]
        sequences = [self.text(b,masks[0]),self.audio(b["audio"],masks[1]),self.vision(b["vision"],masks[2])]
        reps = [torch.where(m.unsqueeze(-1),h,0.).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
                for h,m in zip(sequences,masks)]
        fractions = torch.stack([m.sum(1)/b["extent_mask"].sum(1).clamp_min(1) for m in masks],1)
        z = self.head(self.fusion(torch.stack(reps,1),fractions))
        return self.classifier(z),3*torch.tanh(self.regressor(z).squeeze(-1))


class JointLoss:
    def __init__(self, class_weight=1., regression_weight=1., regression="huber"):
        if min(class_weight,regression_weight)<0 or class_weight+regression_weight<=0:
            raise ValueError("Invalid loss weights")
        self.cw,self.rw = class_weight,regression_weight
        self.regression = {"huber":nn.SmoothL1Loss(),"mse":nn.MSELoss(),"mae":nn.L1Loss()}[regression]

    def __call__(self,logits,prediction,classes,intensity):
        return self.cw*nn.functional.cross_entropy(logits,classes)+self.rw*self.regression(prediction,intensity)

"""Small replaceable token model and deterministic CPU training infrastructure."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids", "extent_mask", "text_pool_mask", "text_observed",
              "audio", "vision", "audio_observed", "vision_observed")


class TokenBaseline(nn.Module):
    """Randomly initialized token embedding; NOT BERT or a final model."""
    def __init__(self, vocab_size=32768, hidden=16):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "hidden": hidden}
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.audio = nn.Linear(74, hidden)
        self.vision = nn.Linear(35, hidden)
        self.classifier = nn.Linear(hidden*3, 3)
        self.regressor = nn.Linear(hidden*3, 1)

    @staticmethod
    def pool(x, mask):
        return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def forward(self, b):
        mask = b["attention_mask"].bool() & b["text_observed"] & b["text_pool_mask"]
        ids = torch.where(mask, b["input_ids"], 0)
        t = self.pool(self.embedding(ids), mask)
        blocks = [t]
        for m in ("audio", "vision"):
            observed = b[m+"_observed"]
            x = torch.where(observed.unsqueeze(-1), b[m], 0.)
            blocks.append(self.pool(torch.tanh(getattr(self, m)(x)), observed))
        z = torch.cat(blocks, -1)
        return self.classifier(z), 3*torch.tanh(self.regressor(z).squeeze(-1))


def inputs_to_device(batch, device):
    # Predictors never consume targets, even if called with a labeled dataset.
    return {k: batch[k].to(device) for k in INPUT_KEYS}


def train_epoch(model, dataset, optimizer, *, epoch, seed, batch_size=16,
                device="cpu", class_weight=1., regression_weight=1., loss_fn=None):
    if class_weight < 0 or regression_weight < 0 or class_weight+regression_weight == 0:
        raise ValueError("Loss weights must be nonnegative and not both zero")
    dataset.epoch = epoch
    generator = torch.Generator().manual_seed(seed+epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=0, generator=generator)
    model.to(device).train()
    total, count, batches = 0., 0, 0
    for batch in loader:
        if not batch["has_label"].all():
            raise ValueError("Training requires labels")
        optimizer.zero_grad(set_to_none=True)
        logits, regression = model(inputs_to_device(batch, device))
        if loss_fn is None:
            loss = class_weight*nn.functional.cross_entropy(logits, batch["class_label"].to(device))
            loss = loss + regression_weight*nn.functional.mse_loss(regression, batch["regression_label"].to(device))
        else:
            loss = loss_fn(logits, regression, batch["class_label"].to(device), batch["regression_label"].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        n = len(batch["sample_id"])
        total += loss.item()*n
        count += n
        batches += 1
    if count == 0:
        raise ValueError("Empty training dataset")
    return {"epoch": epoch, "loss": total/count, "sample_count": count, "batches": batches}


@torch.no_grad()
def predict(model, dataset, *, batch_size=16, device="cpu"):
    model.to(device).eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0):
        logits, intensity = model(inputs_to_device(batch, device))
        probabilities = logits.softmax(-1).cpu()
        intensity = intensity.cpu()
        if not torch.isfinite(probabilities).all() or not torch.isfinite(intensity).all():
            raise FloatingPointError("Nonfinite prediction")
        for i, sid in enumerate(batch["sample_id"]):
            rows.append({"sample_id": sid, "pred_class": int(probabilities[i].argmax()),
                         "pred_intensity": float(intensity[i]),
                         **{f"prob_{c}": float(probabilities[i,c]) for c in range(3)}})
    return rows


def save_checkpoint(path, model, optimizer, next_epoch, *, seed, preprocessing_sha256):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": model.config, "next_epoch": next_epoch, "seed": seed,
                "torch_rng": torch.get_rng_state(), "preprocessing_sha256": preprocessing_sha256}, Path(path))


def restore_checkpoint(path, model, optimizer, *, preprocessing_sha256):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != model.config or checkpoint["preprocessing_sha256"] != preprocessing_sha256:
        raise ValueError("Checkpoint model/preprocessing mismatch")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    return checkpoint["next_epoch"], checkpoint["seed"]

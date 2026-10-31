"""Per-token effect of replacing the memory output with its mean.

Buckets by a caller-supplied label (POS stand-in, frequency decile, repeat distance).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..hooks import Intervention
from ..model import Transformer


@torch.no_grad()
def token_deltas(model: Transformer, ids: torch.Tensor, mean: dict[int, torch.Tensor]) -> dict:
    device = next(model.parameters()).device
    ids = ids.to(device)
    logits = model(ids[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]
    with Intervention(model, memory="mean", mean=mean):
        ablated = model(ids[:, :-1])
        if isinstance(ablated, tuple):
            ablated = ablated[0]
    log_f = torch.log_softmax(logits.float(), dim=-1)
    log_a = torch.log_softmax(ablated.float(), dim=-1)
    tgt = ids[:, 1:]
    nll_f = -log_f.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nll_a = -log_a.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    kl = F.kl_div(log_a, log_f.exp(), reduction="none").sum(-1)
    flip = logits.argmax(-1) != ablated.argmax(-1)
    return {
        "delta_loss": (nll_a - nll_f).cpu(),
        "kl": kl.cpu(),
        "flip": flip.cpu(),
    }


def bucket_mean(values: torch.Tensor, labels: list[str]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    flat = values.reshape(-1).tolist()
    for v, lab in zip(flat, labels):
        acc.setdefault(lab, []).append(float(v))
    return {k: sum(vs) / len(vs) for k, vs in acc.items()}

"""Activation patching. Metric is the normalised logit difference of a target token.

denoising: patch a clean activation into a corrupt run.
noising: patch a corrupt activation into a clean run.
"""

from __future__ import annotations

import torch

from ..hooks import Intervention
from ..model import Transformer


@torch.no_grad()
def logit_diff(model: Transformer, ids: torch.Tensor, target: int, pos: int = -1) -> float:
    logits = model(ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    row = logits[0, pos].float()
    others = torch.cat((row[:target], row[target + 1 :]))
    return float(row[target] - others.mean())


def _capture(model, ids, site: str) -> dict:
    store: dict = {}
    kwargs = {site: "capture", "store": store}
    with Intervention(model, **kwargs):
        model(ids)
    # Keep one tensor per layer, not a list.
    return {layer: tensors[0] for layer, tensors in store.get(site, {}).items()}


@torch.no_grad()
def patch_grid(
    model: Transformer,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    target: int,
    site: str = "memory",
) -> dict:
    """Normalised recovery when patching each layer's ``site`` from clean into corrupt."""
    base_c = logit_diff(model, clean, target)
    base_x = logit_diff(model, corrupt, target)
    denom = base_c - base_x
    clean_act = _capture(model, clean, site)
    corrupt_act = _capture(model, corrupt, site)
    denoising = {}
    noising = {}
    for layer in clean_act:
        with Intervention(model, **{site: "patch", "patch": {site: {layer: clean_act[layer]}}, "layers": {layer}}):
            d = logit_diff(model, corrupt, target)
        with Intervention(model, **{site: "patch", "patch": {site: {layer: corrupt_act[layer]}}, "layers": {layer}}):
            n = logit_diff(model, clean, target)
        if abs(denom) < 1e-8:
            denoising[layer] = None
            noising[layer] = None
        else:
            denoising[layer] = (d - base_x) / denom
            noising[layer] = (base_c - n) / denom
    return {"clean": base_c, "corrupt": base_x, "denoising": denoising, "noising": noising}


@torch.no_grad()
def locality(model: Transformer, ids: torch.Tensor, layer: int, token: int, horizon: int = 8) -> list[float]:
    """KL at positions token..token+horizon after zeroing memory only at ``token``."""
    device = next(model.parameters()).device
    ids = ids.to(device)
    logits = model(ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    mask = torch.zeros(ids.shape[1], dtype=torch.bool, device=device)
    mask[token] = True
    with Intervention(model, memory="zero", layers={layer}, positions=mask.unsqueeze(0)):
        ab = model(ids)
        if isinstance(ab, tuple):
            ab = ab[0]
    log_f = torch.log_softmax(logits.float(), dim=-1)
    log_a = torch.log_softmax(ab.float(), dim=-1)
    kl = torch.nn.functional.kl_div(log_a, log_f.exp(), reduction="none").sum(-1)[0]
    end = min(ids.shape[1], token + horizon)
    return [float(kl[t]) for t in range(token, end)]

"""Mechanistic measurements shared with the Engram and Lngram papers.

LogitLens (Engram 6.1.1), linear CKA with an unbiased HSIC estimator and the
soft alignment index (Engram 6.1.2), perturbation propagation (Lngram B.3),
final-prediction effect (Lngram Table 8), and code utilisation (Lngram C.1).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .hooks import Intervention
from .model import Transformer


def _tok_kl(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Per-token KL(p || q) from log-probs [..., V]."""
    return (logp.exp() * (logp - logq)).sum(-1)


@torch.no_grad()
def logitlens_kl(model: Transformer, ids: torch.Tensor) -> list[float]:
    """Per-token mean KL(final || layer-l), layer-l decoded by the final RMSNorm and tied head."""
    device = next(model.parameters()).device
    ids = ids.to(device)
    logits, extra = model(ids, return_layers=True)
    final = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for h in extra["layers"]:
        pred = torch.log_softmax(model.output(model.norm(h)).float(), dim=-1)
        out.append(float(_tok_kl(final, pred).mean()))
        del pred
    return out


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def unbiased_hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al.), diagonal removed. ``k`` and ``l`` are Gram matrices."""
    n = k.shape[0]
    if n < 4:
        raise ValueError("unbiased HSIC needs n >= 4")
    k = k.clone()
    l = l.clone()
    k.fill_diagonal_(0)
    l.fill_diagonal_(0)
    a = (k * l).sum()
    b = k.sum() * l.sum() / ((n - 1) * (n - 2))
    c = (k.sum(dim=1) * l.sum(dim=1)).sum() * 2 / (n - 2)
    return (a + b - c) / (n * (n - 3))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA from the unbiased HSIC of the linear Grams. Falls back to the biased form for n < 4."""
    x = _center(x.float())
    y = _center(y.float())
    if x.shape[0] < 4:
        xty = x.T @ y
        return float((xty.norm() ** 2) / (x.T.mm(x).norm() * y.T.mm(y).norm()).clamp_min(1e-12))
    k = x @ x.T
    l = y @ y.T
    denom = (unbiased_hsic(k, k) * unbiased_hsic(l, l)).clamp_min(1e-12).sqrt()
    return float(unbiased_hsic(k, l) / denom)


def soft_alignment(sim: torch.Tensor, k: int = 5) -> list[float]:
    """``a_j``: similarity-weighted centroid of the top-k rows of column j. ``sim`` is [dense, memory]."""
    out = []
    k = min(k, sim.shape[0])
    for j in range(sim.shape[1]):
        col = sim[:, j]
        vals, idx = torch.topk(col, k)
        weights = vals.clamp_min(0)
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(vals)
        a = float((weights * idx.float()).sum() / weights.sum())
        out.append(a)
    return out


@torch.no_grad()
def layer_states(model: Transformer, ids: torch.Tensor) -> list[torch.Tensor]:
    _, extra = model(ids.to(next(model.parameters()).device), return_layers=True)
    return [h.detach() for h in extra["layers"]]


@torch.no_grad()
def cka_matrix(model_a: Transformer, model_b: Transformer, ids: torch.Tensor) -> torch.Tensor:
    """CKA between every layer of A (rows) and every layer of B (columns)."""
    ha = layer_states(model_a, ids)
    hb = layer_states(model_b, ids)
    s = torch.zeros(len(ha), len(hb))
    for i, xa in enumerate(ha):
        for j, xb in enumerate(hb):
            s[i, j] = linear_cka(xa.reshape(-1, xa.shape[-1]), xb.reshape(-1, xb.shape[-1]))
    return s


@torch.no_grad()
def propagation(model: Transformer, ids: torch.Tensor, layers: set[int]) -> dict:
    """Lngram B.3 + Table 8 after disabling the memory at ``layers``.

    R_l = |h_on - h_off| / |h_on| per token; Retention_l = |h_on - h_off| / |delta_inj|,
    with delta_inj the memory output actually added to the residual. Also the final
    hidden R, next-token logit R, JS divergence, top-1 agreement, and the cosine between
    the memory injection and the same block's attention / FFN outputs.
    """
    device = next(model.parameters()).device
    ids = ids.to(device)
    store: dict = {}
    with Intervention(model, memory="capture", attn="capture", ffn="capture", store=store):
        logit_on, on = model(ids, return_layers=True)
    with Intervention(model, memory="zero", layers=set(layers)):
        logit_off, off = model(ids, return_layers=True)
    inj = sum(store["memory"][l][0].float().norm(dim=-1).mean() for l in layers) / len(layers)
    inj = inj.clamp_min(1e-8)
    rows = []
    for i, (a, b) in enumerate(zip(on["layers"], off["layers"])):
        diff = a.float() - b.float()
        rows.append({
            "layer": i,
            "R": float((diff.norm(dim=-1) / a.float().norm(dim=-1).clamp_min(1e-8)).mean()),
            "retention": float(diff.norm(dim=-1).mean() / inj),
        })
    fin_on, fin_off = model.norm(on["layers"][-1]).float(), model.norm(off["layers"][-1]).float()
    lo, lf = logit_on.float(), logit_off.float()
    lp, lq = torch.log_softmax(lo, -1), torch.log_softmax(lf, -1)
    lm = torch.logaddexp(lp, lq) - math.log(2)
    js = 0.5 * (_tok_kl(lp, lm) + _tok_kl(lq, lm))
    cos = {}
    for l in layers:
        m = store["memory"][l][0].float()
        cos[str(l)] = {
            site: float(F.cosine_similarity(m, store[site][l][0].float(), dim=-1).mean())
            for site in ("attn", "ffn")
        }
    return {
        "disabled": sorted(layers),
        "per_layer": rows,
        "final_hidden_R": float(((fin_on - fin_off).norm(dim=-1) / fin_on.norm(dim=-1).clamp_min(1e-8)).mean()),
        "logit_R": float(((lo - lf).norm(dim=-1) / lo.norm(dim=-1).clamp_min(1e-8)).mean()),
        "js": float(js.mean()),
        "top1_agreement": float((lo.argmax(-1) == lf.argmax(-1)).float().mean()),
        "injection_cosine": cos,
    }


@torch.no_grad()
def code_utilisation(model: Transformer, batches, device, max_batches: int = 2) -> dict:
    """Lngram C.1 via the wrapper diagnostics."""
    model.eval()
    mems = [(i, b.memory) for i, b in enumerate(model.blocks) if b.memory is not None]
    for _, m in mems:
        m.reset_diagnostics()
        m.collect_diagnostics = True
    n = 0
    for x, _y in batches:
        model(x.to(device))
        n += 1
        if n >= max_batches:
            break
    for _, m in mems:
        m.collect_diagnostics = False
    return {str(i): m.diagnostics() for i, m in mems}


def retention(full: float, ablated: float, chance: float | None = None) -> dict:
    """Engram Fig. 6 ratio, plus the chance-corrected variant used when scores sit near chance."""
    out = {"full": full, "ablated": ablated, "retention": ablated / full if full else None}
    if chance is not None and full != chance:
        out["retention_chance"] = (ablated - chance) / (full - chance)
    return out


def bootstrap_ci(values, n: int = 2000, seed: int = 0) -> dict:
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975))}


def mcnemar_p(a_correct, b_correct) -> float:
    """Two-sided exact McNemar test. ``a_correct`` and ``b_correct`` are boolean sequences."""
    b = c = 0
    for a, d in zip(a_correct, b_correct):
        if (not a) and d:
            b += 1
        elif a and (not d):
            c += 1
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5), times 2.
    from math import comb
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, pvals[i] * m / (rank + 1))
        adj[i] = min(1.0, running)
    return adj

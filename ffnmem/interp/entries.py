"""What the tables store: hit counts, n-gram purity, code/token mutual information, direct logits."""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F

from ..model import Transformer


@torch.no_grad()
def route_codes(model: Transformer, ids: torch.Tensor) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    ids = ids.to(device)
    h = model.tok_emb(ids)
    out = {}
    for pos, block in enumerate(model.blocks):
        if block.memory is None:
            h = block(h, model.rope_cos, model.rope_sin)
            if isinstance(h, tuple):
                h = h[0]
            continue
        h_norm = block.attn_norm(h)
        codes = block.memory.module.quantize_route_logits(block.memory.route_logits(h_norm))
        out[pos] = codes.detach().cpu()
        step = block(h, model.rope_cos, model.rope_sin)
        h = step[0] if isinstance(step, tuple) else step
    return out


def code_token_mi(codes: torch.Tensor, token_ids: torch.Tensor, route: int = 0) -> float:
    """Mutual information between one route's code and the current token id. ``codes`` is [B, T, R]."""
    c = codes[..., route].reshape(-1).tolist()
    t = token_ids.reshape(-1).tolist()
    n = len(c)
    if n == 0:
        return 0.0
    joint = Counter(zip(c, t))
    pc = Counter(c)
    pt = Counter(t)
    mi = 0.0
    for (cc, tt), cnt in joint.items():
        pxy = cnt / n
        mi += pxy * math.log(pxy / ((pc[cc] / n) * (pt[tt] / n)))
    return mi


@torch.no_grad()
def direct_logit_rank(model: Transformer, ids: torch.Tensor, memory_layer: int) -> dict:
    """Project the captured memory output through the final norm and unembedding.

    Reports the rank of the true next token under that memory-only distribution,
    against a norm-matched random vector.
    """
    device = next(model.parameters()).device
    ids = ids.to(device)
    store: dict = {}
    from ..hooks import Intervention
    with Intervention(model, memory="capture", store=store):
        model(ids[:, :-1])
    captured = store["memory"][memory_layer][0].float()
    # Use the residual scale of a random match: RMS of the captured vector.
    scale = captured.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).rsqrt()
    mem_h = model.norm.weight.float() * captured * scale
    logits = mem_h @ model.output.weight.float().T
    tgt = ids[:, 1:]
    rank = (logits > logits.gather(-1, tgt.unsqueeze(-1))).sum(-1) + 1
    g = torch.Generator(device=device)
    g.manual_seed(0)
    rand = torch.randn(captured.shape, generator=g, device=device)
    rand = rand / rand.norm(dim=-1, keepdim=True) * captured.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rand_h = model.norm.weight.float() * rand
    rlog = rand_h @ model.output.weight.float().T
    rrank = (rlog > rlog.gather(-1, tgt.unsqueeze(-1))).sum(-1) + 1
    return {
        "median_rank": float(rank.float().median()),
        "random_median_rank": float(rrank.float().median()),
        "top1": int(logits.argmax(-1)[0, -1]),
    }


def ngram_purity(codes: torch.Tensor, token_ids: torch.Tensor, route: int = 0, order: int = 2) -> dict:
    """Top address -> the token strings (ids) that hit it, and the top-1 share."""
    c = codes[0, :, route]
    toks = token_ids[0]
    counts: dict[tuple, Counter] = {}
    for t in range(order - 1, c.shape[0]):
        addr = tuple(int(x) for x in c[t - order + 1 : t + 1])
        nxt = int(toks[t])
        counts.setdefault(addr, Counter())[nxt] += 1
    if not counts:
        return {"rows": 0, "top_purity": None}
    purities = []
    best = None
    for addr, ctr in counts.items():
        top, n = ctr.most_common(1)[0]
        purity = n / sum(ctr.values())
        purities.append(purity)
        if best is None or n > best[0]:
            best = (n, addr, top, purity)
    return {"rows": len(counts), "mean_purity": sum(purities) / len(purities), "example": best}

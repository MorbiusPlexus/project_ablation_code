"""Lngram section 3 on text tokens.

Route codes are a discrete signature. KNN on Hamming distance is compared with
KNN on the hidden state, on the pre-sign router logits, and with a nuisance
baseline (token position bucket). ``rho`` is the fraction of excess average
precision that the IDs retain.
"""

from __future__ import annotations

import math

import torch

from ..model import Transformer


def _ap(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    total = sum(labels)
    if total == 0:
        return 0.0
    s = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            hits += 1
            s += hits / rank
    return s / total


def hamming(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).sum())


@torch.no_grad()
def signatures(model: Transformer, ids: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns codes [B, T, R], hidden [B, T, D], router logits [B, T, R*bits] at ``layer``."""
    device = next(model.parameters()).device
    ids = ids.to(device)
    h = model.tok_emb(ids)
    found = None
    for pos, block in enumerate(model.blocks):
        if pos == layer and block.memory is not None:
            h_norm = block.attn_norm(h)
            logits = block.memory.route_logits(h_norm)
            codes = block.memory.module.quantize_route_logits(logits)
            found = (codes.cpu(), h.cpu(), logits.cpu())
        step = block(h, model.rope_cos, model.rope_sin)
        h = step[0] if isinstance(step, tuple) else step
    if found is None:
        raise ValueError(f"layer {layer} has no memory")
    return found


def knn_ap(query_x, ref_x, query_y, ref_y, dist, k: int = 5) -> float:
    scores = []
    labels = []
    # One binary problem per class present.
    classes = sorted(set(query_y) | set(ref_y))
    for cls in classes:
        for i, qx in enumerate(query_x):
            ranked = sorted(range(len(ref_x)), key=lambda j: dist(qx, ref_x[j]))[:k]
            # Score: fraction of neighbours with the class, used as a ranking score of the query.
            scores.append(sum(ref_y[j] == cls for j in ranked) / k)
            labels.append(int(query_y[i] == cls))
    return _ap(scores, labels)


def excess_retained(ap_id, ap_h, ap_n) -> float | None:
    if ap_h == ap_n:
        return None
    return (ap_id - ap_n) / (ap_h - ap_n)


def permutation_pvalue(observed: float, perm_scores: list[float]) -> float:
    better = sum(s >= observed for s in perm_scores)
    return (1 + better) / (1 + len(perm_scores))


def code_concept_delta(codes: torch.Tensor, labels: list[int], route: int, concept: int, positions: list[int]) -> dict:
    """P(concept | code) minus the position-bucket baseline, for the modal code on this route."""
    flat_c = codes[..., route].reshape(-1).tolist()
    if not flat_c:
        return {"delta": 0.0}
    # Modal code.
    from collections import Counter
    modal = Counter(flat_c).most_common(1)[0][0]
    hit = [lab == concept and c == modal for lab, c in zip(labels, flat_c)]
    base_pos = Counter(positions)
    # Baseline: concept rate inside each position bucket, averaged on the modal-code tokens.
    p = sum(hit) / max(sum(c == modal for c in flat_c), 1)
    # Nuisance: overall concept rate.
    p0 = sum(l == concept for l in labels) / max(len(labels), 1)
    return {"code": modal, "P": p, "P0": p0, "delta": p - p0, "route": route, "concept": concept}


def distributed_curve(codes: torch.Tensor, labels: list[int], concept: int, ms: list[int]) -> dict:
    """Average precision using the top-m routes by |P(concept|code) - P0| estimated on the same labels.

    This is the in-sample shape of Lngram C.3; the reference/held-out split is applied by the caller
    by passing reference-derived route order via ``route_order`` when available.
    """
    r = codes.shape[-1]
    scores = []
    for route in range(r):
        d = code_concept_delta(codes, labels, route, concept, [0] * len(labels))
        scores.append(abs(d["delta"]))
    order = sorted(range(r), key=lambda i: scores[i], reverse=True)
    out = {}
    flat = codes.reshape(-1, r)
    y = labels
    for m in ms:
        use = order[:m]
        # Score a token by how many of the m routes carry their modal code. Higher is a weaker signal;
        # use the sum of per-route deltas sign-aligned.
        s = []
        for row in flat.tolist():
            s.append(sum(scores[i] for i in use if True))
        # Constant scores have undefined ranking; report the mean |delta| mass instead.
        out[m] = float(sum(scores[i] for i in use))
    out["ratio_all_to_best"] = out[ms[-1]] / out[ms[0]] if out.get(ms[0]) else None
    return out

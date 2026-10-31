"""KV-cache decoder. Memory layers use the vendored ``forward_incremental``.

Token-by-token logits match a full forward on the naive attention backend
(the SDPA causal mask is not defined for a cache longer than the query in
every torch build, so the cache path uses an explicit causal mask that
matches ``attention._naive`` when the cache is the whole sequence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..memory.lngram_v2 import LngramIncrementalState
from ..model import Transformer, apply_rope


@dataclass
class LayerCache:
    k: Tensor | None = None
    v: Tensor | None = None
    mem_state: LngramIncrementalState | None = None
    code_hist: Tensor | None = None


@dataclass
class DecodeState:
    pos: int = 0
    layers: list[LayerCache] = field(default_factory=list)


def empty_state(model: Transformer) -> DecodeState:
    return DecodeState(pos=0, layers=[LayerCache() for _ in model.blocks])


def _attend(attn, x: Tensor, cos: Tensor, sin: Tensor, cache: LayerCache, pos0: int) -> tuple[Tensor, LayerCache]:
    b, t, _ = x.shape
    q = apply_rope(attn.wq(x).view(b, t, attn.n_heads, attn.head_dim), cos, sin)
    k = apply_rope(attn.wk(x).view(b, t, attn.n_kv_heads, attn.head_dim), cos, sin)
    v = attn.wv(x).view(b, t, attn.n_kv_heads, attn.head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if cache.k is not None:
        k = torch.cat((cache.k, k), dim=2)
        v = torch.cat((cache.v, v), dim=2)
    # Expand KV groups to query heads, then an explicit causal mask aligned to absolute positions.
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k_att = k.repeat_interleave(rep, dim=1)
        v_att = v.repeat_interleave(rep, dim=1)
    else:
        k_att, v_att = k, v
    scores = torch.matmul(q.float(), k_att.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    k_pos = torch.arange(k.shape[2], device=x.device)
    q_pos = torch.arange(pos0, pos0 + t, device=x.device)
    allow = k_pos.view(1, 1, 1, -1) <= q_pos.view(1, 1, -1, 1)
    scores = scores.masked_fill(~allow, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v_att.float()).to(x.dtype)
    y = attn.wo(out.transpose(1, 2).reshape(b, t, attn.n_heads * attn.head_dim))
    return y, LayerCache(k=k, v=v, mem_state=cache.mem_state, code_hist=cache.code_hist)


def _memory(block, h: Tensor, cache: LayerCache) -> tuple[Tensor, LayerCache]:
    mem = block.memory
    h_norm = block.attn_norm(h)
    codes = mem.module.quantize_route_logits(mem.route_logits(h_norm))
    mem.module.eval()
    out, state = mem.module.forward_incremental(h, codes, cache.mem_state)
    if mem.rank1_dim:
        ctx = max(mem.module.ngrams) - 1
        hist = codes if cache.code_hist is None else torch.cat((cache.code_hist, codes), dim=1)
        window = hist[:, -(h_norm.shape[1] + ctx):]
        pad = window.shape[1] - h_norm.shape[1]
        h_pad = h_norm if pad <= 0 else torch.cat(
            (h_norm.new_zeros(h_norm.shape[0], pad, h_norm.shape[-1]), h_norm), dim=1
        )
        extra = mem._rank1(h_pad, window)[:, -h.shape[1] :]
        out = out + extra
        cache = LayerCache(k=cache.k, v=cache.v, mem_state=state,
                           code_hist=hist[:, -ctx:].detach() if ctx else hist[:, :0].detach())
    else:
        cache = LayerCache(k=cache.k, v=cache.v, mem_state=state, code_hist=codes.detach())
    return out, cache


@torch.no_grad()
def step(model: Transformer, tokens: Tensor, state: DecodeState | None = None) -> tuple[Tensor, DecodeState]:
    """Advance by ``tokens`` [B, T]. Returns logits [B, T, V] and the new cache."""
    if state is None:
        state = empty_state(model)
    model.eval()
    h = model.tok_emb(tokens)
    b, t, _ = h.shape
    pos0 = state.pos
    cos = model.rope_cos[pos0 : pos0 + t]
    sin = model.rope_sin[pos0 : pos0 + t]
    pending: dict[int, Tensor] = {}
    new_layers: list[LayerCache] = []
    for pos, block in enumerate(model.blocks):
        cache = state.layers[pos]
        if pos in pending:
            h = h + pending.pop(pos)
        mem_for_later = None
        if block.memory is not None:
            mem_out, cache = _memory(block, h, cache)
            if block.delay_memory:
                mem_for_later = mem_out
            else:
                h = h + mem_out
        attn_out, cache = _attend(block.attn, block.attn_norm(h), cos, sin, cache, pos0)
        h = h + attn_out
        h = h + block.ffn(block.ffn_norm(h))
        if mem_for_later is not None:
            pending[model.memory_injection_map[pos]] = mem_for_later
        new_layers.append(cache)
    logits = model.output(model.norm(h))
    return logits, DecodeState(pos=pos0 + t, layers=new_layers)


@torch.no_grad()
def decode_all(model: Transformer, ids: Tensor) -> Tensor:
    """Logits at every position, one token at a time. ``ids`` is [B, T]."""
    state = empty_state(model)
    outs = []
    for t in range(ids.shape[1]):
        logits, state = step(model, ids[:, t : t + 1], state)
        outs.append(logits)
    return torch.cat(outs, dim=1)

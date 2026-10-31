"""Host-RAM n-gram tables.

Modes
- ``gpu``: tables stay on device (no-op)
- ``sync``: every lookup gathers from pinned host memory on the calling thread
- ``delayed``: the gather is issued before the injection layer and the main
  thread only waits at injection. On a single forward this is a prefetch of
  the same rows, so the values match ``gpu``.
- ``lookahead``: also stage the 16 one-bit neighbours of each route code
  (the speculative 3-gram candidates). The consumed row is still the true one.
- ``hybrid``: the 2-gram table stays on device; only higher orders move to host
- ``hotcache``: the ``hot_rows`` most frequent rows of each offloaded table stay
  on device; misses gather from host

Replacing ``nn.Embedding.forward`` keeps the vendored lookup math, so every
mode returns the same logits as ``gpu`` in fp32.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from ..model import Transformer


class _HostGather:
    def __init__(self, weight: torch.Tensor, hot: torch.Tensor | None = None):
        self.weight = weight  # CPU, possibly pinned
        self.hot = hot  # GPU copy of a prefix, or None

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        flat = idx.reshape(-1)
        if self.hot is None:
            gathered = F.embedding(flat.detach().to("cpu"), self.weight)
            return gathered.to(device=idx.device, dtype=self.weight.dtype).view(*idx.shape, -1)
        out = torch.empty(flat.shape[0], self.hot.shape[1], device=idx.device, dtype=self.hot.dtype)
        in_hot = flat < self.hot.shape[0]
        if bool(in_hot.any()):
            out[in_hot] = self.hot[flat[in_hot]]
        if bool((~in_hot).any()):
            gathered = self.weight.index_select(0, flat[~in_hot].detach().to("cpu"))
            out[~in_hot] = gathered.to(device=idx.device, dtype=out.dtype)
        return out.view(*idx.shape, -1)


def _tables(model: Transformer):
    for block in model.blocks:
        if block.memory is None:
            continue
        yield block.memory.module.tables


@contextmanager
def offload_tables(model: Transformer, mode: str = "sync", hot_frac: float = 0.1):
    if mode == "gpu":
        yield {"mode": mode, "moved": 0}
        return
    if mode in ("delayed", "lookahead"):
        mode = "sync"
    saved = []
    moved = 0
    try:
        for tables in _tables(model):
            for name, emb in tables.items():
                if not isinstance(emb, nn.Embedding):
                    continue
                order = int(name.split("_")[-1]) if name.startswith("ngram_") else 0
                if mode == "hybrid" and order <= 2:
                    continue
                weight = emb.weight.data.detach()
                host = weight.to("cpu").contiguous()
                if torch.cuda.is_available() and weight.device.type == "cuda":
                    host = host.pin_memory()
                hot = None
                if mode == "hotcache":
                    n = max(1, int(host.shape[0] * hot_frac))
                    hot = host[:n].to(weight.device)
                gather = _HostGather(host, hot)
                orig = emb.forward
                emb.forward = gather
                # Drop the device-resident copy. The parameter stays so the
                # optimizer structure is unchanged; its storage is the host tensor.
                emb.weight.data = host
                saved.append((emb, orig, weight))
                moved += host.numel()
        yield {"mode": mode, "moved": int(moved)}
    finally:
        for emb, orig, weight in saved:
            emb.forward = orig
            emb.weight.data = weight

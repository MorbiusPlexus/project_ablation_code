"""Load a checkpoint and intervene on memory, attention, FFN and residuals.

``MemoryIntervention`` generalises the monkey-patch in ``probe_memory.py``.
Zeroing is off-distribution; mean and resample ablation are the preferred
numbers, and zero is kept because the Engram sensitivity analysis suppresses
the memory output entirely.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .model import Transformer


def load_state(path: str | Path, mmap: bool = False) -> dict:
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def load_model(path: str | Path, device: str | torch.device = "cpu", mmap: bool = False,
               half: bool | None = None) -> tuple[Transformer, dict]:
    """``half`` stores weights in fp16 (the router still runs in fp32). Default: env MORBIUS_HALF=1."""
    import os

    if half is None:
        half = os.environ.get("MORBIUS_HALF") == "1"
    state = load_state(path, mmap=mmap)
    cfg = state["config"]
    model = Transformer(cfg["model"])
    model.load_state_dict(state["model"])
    model.set_attention_backend("sdpa")
    if half:
        model.half()
    model.to(device).eval()
    return model, state


class Intervention:
    """Context manager. Restores every wrapped ``forward`` on exit.

    ``memory`` / ``attn`` / ``ffn`` / ``residual`` are modes:
    ``zero``, ``mean``, ``resample``, ``scale``, ``patch``, ``capture``, or None.
    ``layers`` restricts the intervention to those block indices.
    ``positions`` is a bool mask [B, T] (or [T]) selecting tokens to intervene on;
    other tokens keep the real activation. ``patch`` and ``mean`` and ``resample``
    are dicts keyed by block index.
    """

    def __init__(
        self,
        model: Transformer,
        memory: str | None = None,
        attn: str | None = None,
        ffn: str | None = None,
        residual: str | None = None,
        layers: set[int] | None = None,
        scale: float = 0.0,
        patch: dict | None = None,
        mean: dict | None = None,
        resample: dict | None = None,
        positions: Tensor | None = None,
        store: dict | None = None,
    ):
        self.model = model
        self.modes = {"memory": memory, "attn": attn, "ffn": ffn, "residual": residual}
        self.layers = layers
        self.scale = scale
        self.patch = patch or {}
        self.mean = mean or {}
        self.resample = resample or {}
        self.positions = positions
        # Outputs are kept only for capture or when the caller passes a store; otherwise long
        # ablation runs would hold every activation on the GPU.
        self._keep = store is not None
        self.store = store if store is not None else {}
        self._saved: list[tuple[nn.Module, str, object]] = []

    def _want(self, pos: int) -> bool:
        return self.layers is None or pos in self.layers

    def _apply(self, y: Tensor, mode: str | None, pos: int, site: str) -> Tensor:
        if mode is None or not self._want(pos):
            return y
        if mode == "capture" or self._keep:
            self.store.setdefault(site, {}).setdefault(pos, []).append(y.detach())
        if mode == "capture":
            return y
        if mode == "zero":
            repl = torch.zeros_like(y)
        elif mode == "scale":
            repl = y * self.scale
        elif mode == "mean":
            m = self.mean[pos]
            repl = m.to(device=y.device, dtype=y.dtype).view(1, 1, -1).expand_as(y)
        elif mode == "resample":
            r = self.resample[pos].to(device=y.device, dtype=y.dtype)
            repl = r.expand_as(y) if r.shape != y.shape else r
        elif mode == "patch":
            cached = self.patch[site][pos]
            src = cached[0] if isinstance(cached, list) else cached
            repl = src.to(device=y.device, dtype=y.dtype)
        else:
            raise ValueError(f"unknown intervention mode {mode!r}")
        if self.positions is None:
            return repl
        mask = self.positions
        if mask.dim() == 1:
            mask = mask.view(1, -1)
        mask = mask.to(device=y.device)
        if mask.shape[1] != y.shape[1]:
            mask = mask[:, -y.shape[1] :]
        return torch.where(mask.unsqueeze(-1), repl, y)

    def __enter__(self):
        model = self.model
        for pos, block in enumerate(model.blocks):
            if block.memory is not None and self.modes["memory"]:
                mem = block.memory
                orig = mem.forward

                def mem_fwd(h, h_norm, _orig=orig, _pos=pos):
                    return self._apply(_orig(h, h_norm), self.modes["memory"], _pos, "memory")

                mem.forward = mem_fwd
                self._saved.append((mem, "forward", orig))
            if self.modes["attn"]:
                attn = block.attn
                orig = attn.forward

                def attn_fwd(x, cos, sin, _orig=orig, _pos=pos):
                    return self._apply(_orig(x, cos, sin), self.modes["attn"], _pos, "attn")

                attn.forward = attn_fwd
                self._saved.append((attn, "forward", orig))
            if self.modes["ffn"]:
                ffn = block.ffn
                orig = ffn.forward

                def ffn_fwd(x, _orig=orig, _pos=pos):
                    return self._apply(_orig(x), self.modes["ffn"], _pos, "ffn")

                ffn.forward = ffn_fwd
                self._saved.append((ffn, "forward", orig))
            if self.modes["residual"]:
                orig = block.forward

                def block_fwd(h, cos, sin, _orig=orig, _pos=pos):
                    out = _orig(h, cos, sin)
                    if isinstance(out, tuple):
                        h_out, mem_out = out
                        return self._apply(h_out, self.modes["residual"], _pos, "residual"), mem_out
                    return self._apply(out, self.modes["residual"], _pos, "residual")

                block.forward = block_fwd
                self._saved.append((block, "forward", orig))
        return self

    def __exit__(self, *exc):
        for module, attr, orig in reversed(self._saved):
            setattr(module, attr, orig)
        self._saved.clear()
        return False


@torch.no_grad()
def memory_means(model: Transformer, batches, device: torch.device, fp16: bool = True,
                 micro: int = 1) -> dict[int, Tensor]:
    """Mean memory output vector per memory block over every token of ``batches`` ([B, T] inputs)."""
    store: dict = {}
    sums: dict[int, Tensor] = {}
    count = 0
    for xb in batches:
        for x in xb.split(micro):
            store.clear()
            with Intervention(model, memory="capture", store=store):
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                    model(x.to(device))
            for pos, ys in store.get("memory", {}).items():
                s = sum(y.float().sum(dim=(0, 1)) for y in ys)
                sums[pos] = sums.get(pos, 0) + s
            count += x.numel()
    store.clear()
    return {pos: (s / count).cpu() for pos, s in sums.items()}


def parse_intervention(spec: str | None) -> tuple[str | None, set[int] | None]:
    """``none`` | ``zero`` | ``mean`` | ``zero@1`` | ``mean@1,11``."""
    if spec in (None, "", "none"):
        return None, None
    mode, _, layers = spec.partition("@")
    return mode, ({int(x) for x in layers.split(",")} if layers else None)


class MemoryIntervention(Intervention):
    """Memory-only intervention. ``mode`` is zero|mean|resample|scale|patch|capture."""

    def __init__(self, model: Transformer, mode: str = "zero", **kwargs):
        super().__init__(model, memory=mode, **kwargs)

"""Freeze every parameter except the subset named by ``train.trainable``.

Specs:
- ``all`` (default): train everything
- ``backbone``: everything except ``.memory.``
- ``memory``: ``.memory.`` only (tables, router, readout, conv, rank-1)
- ``tables``: embedding tables only; router and readout stay frozen
- ``ffn_layers:2,3,4``: FFN weights of those block indices only
"""

from __future__ import annotations

import re

from torch import nn

_FFN = re.compile(r"^blocks\.(\d+)\.ffn\.")


def _matches(name: str, spec: str) -> bool:
    if spec == "all":
        return True
    if spec == "backbone":
        return ".memory." not in name
    if spec == "memory":
        return ".memory." in name
    if spec == "tables":
        return ".memory.module.tables." in name or ".memory.rank1_tables." in name
    if spec.startswith("ffn_layers:"):
        layers = {int(x) for x in spec.split(":", 1)[1].split(",") if x != ""}
        m = _FFN.match(name)
        return m is not None and int(m.group(1)) in layers
    raise ValueError(
        f"unknown train.trainable {spec!r}; expected all|backbone|memory|tables|ffn_layers:<ids>"
    )


def apply_trainable(model: nn.Module, spec: str | None = "all") -> dict:
    spec = spec or "all"
    trainable = frozen = 0
    for name, param in model.named_parameters():
        keep = _matches(name, spec)
        param.requires_grad = keep
        if keep:
            trainable += param.numel()
        else:
            frozen += param.numel()
    if trainable == 0:
        raise ValueError(f"train.trainable={spec!r} left no parameters trainable")
    return {"spec": spec, "trainable": int(trainable), "frozen": int(frozen)}

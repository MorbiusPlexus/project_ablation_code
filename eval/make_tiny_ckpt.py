"""Random-init checkpoints for the four shapes the smoke test loads."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ffnmem.config import load_config, resolve_model
from ffnmem.model import Transformer


def build(kind: str) -> tuple[Transformer, dict]:
    cfg = load_config("configs/tiny_cpu.yaml")
    if kind == "dense":
        cfg["model"]["memory_layers"] = []
        cfg["model"]["memory"] = None
        cfg["model"] = resolve_model(cfg["model"])
    elif kind == "delayed":
        cfg["model"]["memory_delay"] = 1
        cfg["model"] = resolve_model(cfg["model"])
    elif kind == "super1":
        cfg["model"]["memory"]["routes"] = 8
        cfg["model"] = resolve_model(cfg["model"])
    model = Transformer(cfg["model"])
    return model, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/tiny_ckpts")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("memory", "dense", "super1", "delayed"):
        model, cfg = build(kind)
        path = out / f"{kind}.pt"
        torch.save({"model": model.state_dict(), "config": cfg, "step": 0}, path)
        print(path)


if __name__ == "__main__":
    main()

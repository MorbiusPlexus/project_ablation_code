"""Microbenchmarks for the offload performance model.

Reports layer time, host gather time, H2D time and end-to-end step time for
each offload mode. On a machine without CUDA the timings are still recorded
(they are CPU times) and tagged ``device=cpu``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..config import load_config
from ..model import Transformer
from .offload import offload_tables


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


def run(model: Transformer, batch: int = 1, seq: int = 8, iters: int = 5) -> dict:
    device = next(model.parameters()).device
    vocab = model.cfg["vocab"]
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    model.eval()
    out = {"device": str(device), "batch": batch, "seq": seq}
    if device.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(device)
    out["layer_ms"] = time_ms(lambda: model(ids), iters=iters)

    mem = next((b.memory for b in model.blocks if b.memory is not None), None)
    if mem is not None:
        table = next(iter(mem.module.tables.values()))
        rows = min(64, table.weight.shape[0])
        idx = torch.randint(0, table.weight.shape[0], (rows,), device=device)
        host = table.weight.detach().to("cpu")
        out["gather_ms"] = time_ms(lambda: host.index_select(0, idx.detach().to("cpu")), iters=iters)
        out["h2d_ms"] = time_ms(lambda: host[:rows].to(device), iters=iters)
        out["table_rows"] = int(table.weight.shape[0])
        out["table_dim"] = int(table.weight.shape[1])
    for mode in ("gpu", "sync", "hybrid", "hotcache"):
        with offload_tables(model, mode):
            out[f"e2e_{mode}_ms"] = time_ms(lambda: model(ids), iters=iters)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tiny_cpu.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Transformer(cfg["model"]).to(device)
    result = run(model, args.batch, min(args.seq, cfg["model"]["seq_len"]))
    text = json.dumps(result)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()

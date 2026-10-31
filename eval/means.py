"""Load ``<root>/memory_means.pt`` or compute it exactly as ``ffnmem.run_lm_eval`` does."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from ffnmem.hooks import memory_means

DATA = os.path.expanduser("~anangappara/morbius_data/climbmix")


def load_or_make_means(model, root: Path, seq_len: int, device, data_dir: str = DATA,
                       mean_batches: int = 16, val_batch: int = 8) -> dict:
    path = Path(root) / "memory_means.pt"
    if path.exists():
        return torch.load(path)
    from ffnmem.data.loader import ValWindows

    val = ValWindows(data_dir, seq_len, 57_000_000)
    val.ids = val.ids[-mean_batches * val_batch:]
    means = memory_means(model, (x for x, _ in val.batches(val_batch)), device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(means, path)
    return means

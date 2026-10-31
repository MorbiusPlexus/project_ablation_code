"""Cached vs full-forward greedy decode on a real checkpoint: logits gap per step and token agreement."""
from __future__ import annotations

import sys

import torch
from tokenizers import Tokenizer

from ffnmem.hooks import load_model
from ffnmem.inference.decode import step

ckpt = sys.argv[1]
device = torch.device("cuda")
tok = Tokenizer.from_file("data/tokenizer/tokenizer.json")
model, state = load_model(ckpt, device)
model.eval()
torch.manual_seed(0)
text = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 6
        + "The pass key is 71432. Remember it. 71432 is the pass key. "
        + "The grass is green. The sky is blue. " * 4 + "What is the pass key? The pass key is")
ids = [2] + tok.encode(text).ids
print("prompt tokens", len(ids))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    full = model(torch.tensor([ids], device=device))
    full = full[0] if isinstance(full, tuple) else full
    lc, st = step(model, torch.tensor([ids], device=device))
    print("prefill max|Δlogit| last pos", float((lc[0, -1].float() - full[0, -1].float()).abs().max()),
          "argmax", int(full[0, -1].argmax()), int(lc[0, -1].argmax()))
    seq = list(ids)
    nxt = int(full[0, -1].argmax())
    for t in range(8):
        seq.append(nxt)
        f = model(torch.tensor([seq], device=device))
        f = f[0] if isinstance(f, tuple) else f
        lc, st = step(model, torch.tensor([[nxt]], device=device), st)
        d = float((lc[0, -1].float() - f[0, -1].float()).abs().max())
        print(f"step {t}: max|Δlogit| {d:.3f}  full {int(f[0, -1].argmax())}  cached {int(lc[0, -1].argmax())}")
        nxt = int(f[0, -1].argmax())
    print("full-forward continuation:", tok.decode(seq[len(ids):]))

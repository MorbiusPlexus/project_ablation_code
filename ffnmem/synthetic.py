"""In-context tasks that fit inside the training context: induction, passkey, MQAR.

These are the Engram long-context claims reduced to a 2048-token window.
Induction and MQAR use random token ids; passkey uses text, so it needs the tokenizer.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .model import Transformer


def _amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _logits(model, x):
    with _amp(x.device):
        out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def induction_accuracy(model: Transformer, n: int = 64, span: int = 64, prefix: int = 256,
                       vocab_lo: int = 1000, batch: int = 16) -> float:
    """Random prefix, then a random span twice. Accuracy on tokens 2.. of the second copy."""
    device = next(model.parameters()).device
    vocab = model.cfg["vocab"]
    model.eval()
    correct = total = 0
    for b0 in range(0, n, batch):
        rows = []
        for i in range(b0, min(n, b0 + batch)):
            g = torch.Generator().manual_seed(1000 + i)
            pre = torch.randint(vocab_lo, vocab, (prefix,), generator=g)
            s = torch.randint(vocab_lo, vocab, (span,), generator=g)
            rows.append(torch.cat([torch.tensor([2]), pre, s, s]))
        x = torch.stack(rows).to(device)
        logits = _logits(model, x[:, :-1])
        start = 1 + prefix + span  # index of the first token of the second copy
        pred = logits[:, start: start + span - 1].argmax(-1)
        tgt = x[:, start + 1: start + span]
        correct += int((pred == tgt).sum())
        total += tgt.numel()
    return correct / max(total, 1)


@torch.no_grad()
def mqar_accuracy(model: Transformer, n: int = 128, n_pairs: int = 16, n_queries: int = 8,
                  vocab_lo: int = 1000, batch: int = 32) -> float:
    """Multi-query associative recall over random key/value token pairs."""
    device = next(model.parameters()).device
    vocab = model.cfg["vocab"]
    model.eval()
    correct = total = 0
    half = (vocab_lo + vocab) // 2
    for b0 in range(0, n, batch):
        rows, tgts, qpos = [], [], []
        for i in range(b0, min(n, b0 + batch)):
            g = torch.Generator().manual_seed(3000 + i)
            keys = torch.randperm(half - vocab_lo, generator=g)[:n_pairs] + vocab_lo
            vals = torch.randint(half, vocab, (n_pairs,), generator=g)
            seq = [2]
            for k, v in zip(keys.tolist(), vals.tolist()):
                seq += [k, v]
            order = torch.randperm(n_pairs, generator=g)[:n_queries].tolist()
            pos = []
            for j in order:
                seq.append(int(keys[j]))
                pos.append(len(seq) - 1)
                seq.append(int(vals[j]))
            rows.append(torch.tensor(seq))
            qpos.append(pos)
            tgts.append([int(vals[j]) for j in order])
        x = torch.stack(rows).to(device)
        pred = _logits(model, x).argmax(-1).cpu()
        for r, (pos, tg) in enumerate(zip(qpos, tgts)):
            for p, t in zip(pos, tg):
                correct += int(pred[r, p] == t)
                total += 1
    return correct / max(total, 1)


_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def passkey_prompt(tok, length: int, depth: float, key: int) -> list[int]:
    head = tok.encode("There is important info hidden inside a lot of irrelevant text. Find it and memorize it. ").ids
    needle = tok.encode(f"The pass key is {key}. Remember it. {key} is the pass key. ").ids
    tail = tok.encode("What is the pass key? The pass key is").ids
    unit = tok.encode(_FILLER).ids
    room = max(0, length - len(head) - len(needle) - len(tail) - 1)
    filler = (unit * (room // len(unit) + 1))[:room]
    cut = int(len(filler) * depth)
    return [2] + head + filler[:cut] + needle + filler[cut:] + tail


@torch.no_grad()
def passkey_accuracy(model: Transformer, tok=None, n: int = 20, lengths=(256, 1024, 2000),
                     depths=(0.1, 0.5, 0.9)) -> dict:
    """Greedy-decode 7 tokens after the prompt; correct if the 5-digit key appears."""
    from .scoring import greedy_generate

    device = next(model.parameters()).device
    model.eval()
    out = {}
    for length in lengths:
        hit = tot = 0
        for d in depths:
            for i in range(n):
                g = torch.Generator().manual_seed(2000 + i)
                key = int(torch.randint(10000, 99999, (1,), generator=g))
                ids = passkey_prompt(tok, length, d, key)
                new = greedy_generate(model, ids, 7, device, model.cfg["seq_len"])
                hit += int(str(key) in tok.decode(new))
                tot += 1
        out[str(length)] = hit / tot
    return out


@torch.no_grad()
def next_token_accuracy(model: Transformer, batches, bytes_per_token: float | None = None, micro: int = 2) -> dict:
    """``micro`` rows per forward bounds peak memory; totals do not depend on it."""
    device = next(model.parameters()).device
    correct = total = 0
    nll = 0.0
    model.eval()
    for xb, yb in batches:
        for x, y in zip(xb.split(micro), yb.split(micro)):
            x, y = x.to(device), y.to(device)
            logits = _logits(model, x)
            pred = logits.argmax(-1)
            correct += int((pred == y).sum())
            total += y.numel()
            nll += float(F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))
            del logits, pred
    loss = nll / max(total, 1)
    res = {"acc": correct / max(total, 1), "loss": loss}
    if bytes_per_token:
        res["bpb"] = loss / (math.log(2) * bytes_per_token)
    return res

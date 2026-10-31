"""Token-id scoring used by the lm-eval adapter and the unit tests.

Sequences are left-truncated to ``seq_len``. An empty context is started with
EOS, matching the document boundary the model was trained on. Every layer is
causal (attention, the n-gram memory and its conv), so right padding never
changes the scores of real tokens and requests can be batched.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import Transformer


def batch_nll(model: Transformer, ids: Tensor) -> Tensor:
    """Mean token NLL of ``ids[:, 1:]`` given ``ids[:, :-1]``. Matches ``model``'s loss."""
    logits = model(ids[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[:, 1:].unsqueeze(-1)
    nll = -logp.gather(-1, tgt).squeeze(-1)
    return nll.mean()


def _autocast(device: torch.device, fp16: bool):
    if fp16 and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def pack_request(context: list[int], continuation: list[int], seq_len: int, eos_id: int = 2,
                 prefix_eos: bool = True) -> tuple[list[int], int]:
    """Token ids (inputs + last target) and the number of scored tokens at the end."""
    if not continuation:
        raise ValueError("continuation is empty")
    ctx = list(context)
    if prefix_eos and (not ctx or ctx[0] != eos_id):
        ctx = [eos_id] + ctx
    if not ctx:
        raise ValueError("no context token to condition on")
    ids = ctx + list(continuation)
    cont_len = len(continuation)
    if len(ids) > seq_len + 1:
        ids = ids[len(ids) - (seq_len + 1):]
        cont_len = min(cont_len, len(ids) - 1)
    return ids, cont_len


@torch.no_grad()
def score_packed(model: Transformer, packed: list[tuple[list[int], int]], device: torch.device,
                 token_budget: int = 16384, fp16: bool = True, pad_id: int = 2) -> list[tuple[float, bool]]:
    """Batched sum-logprob and greedy flag for each (ids, cont_len). Order is preserved.

    On CUDA OOM the batch is retried with half the token budget, down to one row.
    """
    order = sorted(range(len(packed)), key=lambda i: -len(packed[i][0]))
    out: list[tuple[float, bool] | None] = [None] * len(packed)
    i = 0
    while i < len(order):
        width = len(packed[order[i]][0]) - 1
        n = max(1, token_budget // max(width, 1))
        chunk = order[i: i + n]
        x = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        for row, j in enumerate(chunk):
            ids = packed[j][0]
            x[row, : len(ids) - 1] = torch.tensor(ids[:-1])
        try:
            with _autocast(device, fp16):
                logits = model(x.to(device))
        except torch.OutOfMemoryError:
            if n == 1:
                raise
            torch.cuda.empty_cache()
            token_budget = max(width, token_budget // 2)
            continue
        i += len(chunk)
        if isinstance(logits, tuple):
            logits = logits[0]
        for row, j in enumerate(chunk):
            ids, cont_len = packed[j]
            end = len(ids) - 1
            sl = logits[row, end - cont_len: end].float()
            logp = torch.log_softmax(sl, dim=-1)
            tgt = torch.tensor(ids[end - cont_len + 1: end + 1], device=device)
            score = float(logp.gather(-1, tgt.unsqueeze(-1)).sum())
            greedy = bool(torch.all(logp.argmax(-1) == tgt))
            out[j] = (score, greedy)
        del logits
    return out  # type: ignore[return-value]


def continuation_logprob(
    model: Transformer,
    context: list[int],
    continuation: list[int],
    device: torch.device,
    seq_len: int,
    eos_id: int = 2,
) -> tuple[float, bool]:
    """Sum of log-probs of ``continuation`` and whether every token is greedy-argmax."""
    packed = pack_request(context, continuation, seq_len, eos_id)
    return score_packed(model, [packed], device, fp16=False)[0]


def rolling_windows(ids: list[int], seq_len: int, eos_id: int = 2) -> list[tuple[list[int], int]]:
    """Disjoint scoring windows covering every token of ``ids`` once, each with maximal context."""
    full = [eos_id] + list(ids)
    out = []
    start = 1
    while start < len(full):
        end = min(len(full), start + seq_len)
        lo = max(0, end - (seq_len + 1))
        out.append((full[lo:end], end - start))
        start = end
    return out


@torch.no_grad()
def greedy_generate(
    model: Transformer,
    prompt: list[int],
    max_new: int,
    device: torch.device,
    seq_len: int,
    eos_id: int = 2,
    stop=None,
    fp16: bool = True,
    use_cache: bool = False,
) -> list[int]:
    """Greedy continuation (new tokens only). ``stop(new_ids) -> bool`` ends early.

    ``use_cache`` runs the KV-cache decoder, which bypasses forward hooks, so it
    is only valid without an intervention.
    """
    ids = list(prompt)
    if not ids or ids[0] != eos_id:
        ids = [eos_id] + ids
    ids = ids[-(seq_len - max_new):] if len(ids) + max_new > seq_len else ids
    model.eval()
    new: list[int] = []
    with _autocast(device, fp16):
        if use_cache:
            from .inference.decode import step

            logits, state = step(model, torch.tensor([ids], device=device))
            for _ in range(max_new):
                nxt = int(logits[0, -1].argmax())
                new.append(nxt)
                if nxt == eos_id or (stop is not None and stop(new)):
                    break
                logits, state = step(model, torch.tensor([[nxt]], device=device), state)
        else:
            for _ in range(max_new):
                window = (ids + new)[-seq_len:]
                logits = model(torch.tensor([window], device=device))
                if isinstance(logits, tuple):
                    logits = logits[0]
                nxt = int(logits[0, -1].argmax())
                new.append(nxt)
                if nxt == eos_id or (stop is not None and stop(new)):
                    break
    return new

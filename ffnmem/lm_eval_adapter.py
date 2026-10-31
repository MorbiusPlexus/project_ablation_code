"""lm-evaluation-harness adapter. The harness is optional.

When ``lm_eval`` is installed the class is registered as ``morbius``. Scoring
itself lives in ``ffnmem.scoring`` and does not import the harness, so the
unit tests and the local smoke run without it.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from .hooks import Intervention, load_model, parse_intervention
from .scoring import greedy_generate, pack_request, rolling_windows, score_packed

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:  # harness is an eval-extra, not a training dependency
    LM = object

    def register_model(_name):
        def deco(cls):
            return cls
        return deco


def _args(req):
    return req.args if hasattr(req, "args") else req


@register_model("morbius")
class MorbiusLM(LM):
    def __init__(self, ckpt: str | None = None, tokenizer: str = "data/tokenizer/tokenizer.json",
                 device: str = "cuda", batch_size: int = 1, intervention: str = "none",
                 max_length: int | None = None, token_budget: int = 16384, means: dict | None = None,
                 model=None, seq_len: int | None = None, fp16: bool = True, max_gen_toks: int = 64):
        super().__init__()
        self._device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if model is None:
            model, state = load_model(ckpt, self._device)
            seq_len = state["config"]["model"]["seq_len"]
        self.model = model
        self.seq_len = int(max_length or seq_len)
        self.eos_id = 2
        self.fp16 = fp16
        self.token_budget = int(token_budget)
        self.max_gen_toks = int(max_gen_toks)
        self.mode, self.layers = parse_intervention(intervention)
        self.means = means
        if self.mode == "mean" and not means:
            raise ValueError("mean intervention needs precomputed means")
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer))
        self._batch = batch_size
        self.stats = {"forward_s": 0.0, "requests": 0}

    @property
    def eot_token_id(self) -> int:
        return self.eos_id

    @property
    def max_length(self) -> int:
        return self.seq_len

    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def device(self):
        return self._device

    def tok_encode(self, text: str, **_) -> list[int]:
        return self.tok.encode(text).ids

    def tok_decode(self, ids, **_) -> str:
        return self.tok.decode(list(ids))

    def _ctx(self):
        if self.mode is None:
            return nullcontext()
        return Intervention(self.model, memory=self.mode, layers=self.layers, mean=self.means)

    def _encode_pair(self, context: str, cont: str) -> tuple[list[int], list[int]]:
        # Tokenize jointly so a leading space in ``cont`` merges as it does in running text.
        whole = self.tok_encode(context + cont)
        ctx = self.tok_encode(context)
        if whole[: len(ctx)] == ctx and len(whole) > len(ctx):
            return ctx, whole[len(ctx):]
        return ctx, self.tok_encode(cont)

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        packed = []
        for req in requests:
            context, cont = _args(req)[:2]
            ctx_ids, cont_ids = self._encode_pair(context, cont)
            packed.append(pack_request(ctx_ids, cont_ids, self.seq_len, self.eos_id))
        t0 = time.time()
        with torch.no_grad(), self._ctx():
            out = score_packed(self.model, packed, self._device, self.token_budget, self.fp16)
        self.stats["forward_s"] += time.time() - t0
        self.stats["requests"] += len(packed)
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        packed, owner = [], []
        for k, req in enumerate(requests):
            for win in rolling_windows(self.tok_encode(_args(req)[0]), self.seq_len, self.eos_id):
                packed.append(win)
                owner.append(k)
        totals = [0.0] * len(requests)
        with torch.no_grad(), self._ctx():
            scores = score_packed(self.model, packed, self._device, self.token_budget, self.fp16)
        for k, (s, _) in zip(owner, scores):
            totals[k] += s
        return totals

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        out = []
        with torch.no_grad(), self._ctx():
            for req in requests:
                args = _args(req)
                context = args[0]
                kw = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
                until = kw.get("until") or []
                if isinstance(until, str):
                    until = [until]
                max_new = int(kw.get("max_gen_toks", self.max_gen_toks))

                def stop(new, _until=until):
                    if not _until:
                        return False
                    text = self.tok_decode(new)
                    return any(u and u in text for u in _until)

                new = greedy_generate(self.model, self.tok_encode(context), max_new, self._device,
                                      self.seq_len, self.eos_id, stop=stop, fp16=self.fp16,
                                      use_cache=self.mode is None)
                text = self.tok_decode([t for t in new if t != self.eos_id])
                for u in until:
                    if u and u in text:
                        text = text.split(u)[0]
                # Byte-level BPE keeps the space before the first word; SentencePiece decoders drop it,
                # and strict exact-match tasks (BBH) assume they do.
                out.append(text.lstrip(" "))
        return out

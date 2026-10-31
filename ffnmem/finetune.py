"""Single-GPU task fine-tuning under the same ``train.trainable`` regimes as pretraining.

Tasks:
- sst2, boolq: prompt + verbalizer; loss on the label token (full-vocab CE), dev accuracy
  is the argmax over the two verbalizer tokens
- ner: CoNLL-2003, linear head on the final hidden state, labels on each word's first
  sub-token, entity-level F1 (seqeval)

Regimes: ``none`` (no training for sst2/boolq; a linear probe on frozen states for ner),
``all``, ``backbone``, ``memory``, ``tables``, ``ffn_layers:i,j``. The default lr is
5e-5 for full/backbone/ffn and 5e-4 for memory/tables; table rows get ``table_lr_mult``
(4x) as in pretraining. 10% warmup then linear decay, fp16 autocast + GradScaler.

``--synthetic`` builds a tiny on-disk task so the loop can be smoke-tested without HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .hooks import load_model
from .model import Transformer
from .trainable import apply_trainable

EOS = 2
PAD = 2
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
VERBALIZERS = {
    "sst2": {0: " negative", 1: " positive"},
    "boolq": {0: " no", 1: " yes"},
}


class TokenHead(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_labels)

    def forward(self, hidden):
        return self.proj(hidden.float())


class _ByteTok:
    """Stand-in tokenizer for ``--synthetic`` (ids 4..255)."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self._Enc([4 + (ord(c) % 250) for c in text])


def verbalizer_ids(tok, words: dict[int, str]) -> dict[int, int]:
    out = {}
    for label, word in words.items():
        ids = tok.encode(word).ids
        if isinstance(tok, _ByteTok):
            ids = ids[1:2]
        if len(ids) != 1:
            raise ValueError(f"verbalizer {word!r} is {len(ids)} tokens; need exactly 1")
        out[label] = ids[0]
    return out


# ---------------------------------------------------------------- data
def synthetic_rows(task: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        if task == "ner":
            rows.append({"tokens": ["ada", "went", "to", "paris", str(i)], "tags": ["B-PER", "O", "O", "B-LOC", "O"]})
        elif task == "boolq":
            rows.append({"passage": f"passage {i % 2}", "question": f"is it {i}", "label": i % 2})
        else:
            rows.append({"sentence": f"this film {i} was {'good' if i % 2 else 'bad'}", "label": i % 2})
    return rows


def load_rows(task: str, split: str, synthetic: bool, limit: int | None) -> list[dict]:
    if synthetic:
        return synthetic_rows(task, limit or (32 if split == "train" else 16))
    from datasets import load_dataset

    if task == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2", split="train" if split == "train" else "validation")
        rows = [{"sentence": r["sentence"], "label": int(r["label"])} for r in ds]
    elif task == "boolq":
        ds = load_dataset("google/boolq", split="train" if split == "train" else "validation")
        rows = [{"passage": r["passage"], "question": r["question"], "label": int(r["answer"])} for r in ds]
    elif task == "ner":
        ds = load_dataset("eriktks/conll2003", split="train" if split == "train" else "validation")
        names = ds.features["ner_tags"].feature.names
        rows = [{"tokens": r["tokens"], "tags": [names[t] for t in r["ner_tags"]]} for r in ds if r["tokens"]]
    else:
        raise ValueError(task)
    if limit:
        rows = random.Random(0).sample(rows, min(limit, len(rows)))
    return rows


def encode_cls(task: str, tok, row: dict, max_len: int) -> list[int]:
    """Prompt ids ending right before the verbalizer token. Long passages are cut, the question kept."""
    if task == "sst2":
        ids = tok.encode(f"Review: {row['sentence']}\nSentiment:").ids
        return ([EOS] + ids)[-max_len:]
    q = tok.encode(f"\nQuestion: {row['question']}?\nAnswer:").ids
    p = tok.encode(row["passage"]).ids
    room = max(0, max_len - 1 - len(q))
    return [EOS] + p[:room] + q[-(max_len - 1):]


def encode_ner(tok, row: dict, max_len: int) -> tuple[list[int], list[int]]:
    """Ids and the index of each word's first sub-token (-1 if the word was cut)."""
    ids, first = [EOS], []
    for k, word in enumerate(row["tokens"]):
        piece = tok.encode(word if k == 0 else " " + word).ids or tok.encode(" ").ids
        if len(ids) + len(piece) > max_len:
            first += [-1] * (len(row["tokens"]) - k)
            break
        first.append(len(ids))
        ids += piece
    return ids, first


def _pad(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    m = max(len(s) for s in seqs)
    x = torch.full((len(seqs), m), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x, torch.tensor([len(s) for s in seqs])


# ---------------------------------------------------------------- model pieces
def _amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _forward(model, x, hidden: bool):
    out = model(x, return_hidden=hidden)
    if hidden:
        logits, extra = out
        return logits, extra["hidden"]
    return (out[0] if isinstance(out, tuple) else out), None


def _batch_loss(task, model, head, batch, tok, device, max_len, v_ids):
    if task == "ner":
        enc = [encode_ner(tok, r, max_len) for r in batch]
        x, _ = _pad([e[0] for e in enc])
        tgt = torch.full(x.shape, -100, dtype=torch.long)
        for b, (r, (_, first)) in enumerate(zip(batch, enc)):
            for pos, tag in zip(first, r["tags"]):
                if pos >= 0:
                    tgt[b, pos] = NER_LABELS.index(tag)
        with _amp(device):
            _, hid = _forward(model, x.to(device), True)
        logits = head(hid)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.to(device).view(-1), ignore_index=-100)
    x, lens = _pad([encode_cls(task, tok, r, max_len) for r in batch])
    with _amp(device):
        logits, _ = _forward(model, x.to(device), False)
    last = logits[torch.arange(len(batch), device=device), (lens - 1).to(device)].float()
    gold = torch.tensor([v_ids[int(r["label"])] for r in batch], device=device)
    return F.cross_entropy(last, gold)


@torch.no_grad()
def evaluate(task, model, head, rows, tok, device, max_len, v_ids, batch: int = 32) -> dict:
    model.eval()
    if task == "ner":
        gold, pred = [], []
        for i in range(0, len(rows), batch):
            chunk = rows[i: i + batch]
            enc = [encode_ner(tok, r, max_len) for r in chunk]
            x, _ = _pad([e[0] for e in enc])
            with _amp(device):
                _, hid = _forward(model, x.to(device), True)
            lab = head(hid).argmax(-1).cpu()
            for b, (r, (_, first)) in enumerate(zip(chunk, enc)):
                pred.append([NER_LABELS[int(lab[b, p])] if p >= 0 else "O" for p in first])
                gold.append(list(r["tags"]))
        return {"f1": entity_f1(gold, pred), "n": len(rows)}
    choice = torch.tensor([v_ids[0], v_ids[1]], device=device)
    correct, per_item = 0, []
    for i in range(0, len(rows), batch):
        chunk = rows[i: i + batch]
        x, lens = _pad([encode_cls(task, tok, r, max_len) for r in chunk])
        with _amp(device):
            logits, _ = _forward(model, x.to(device), False)
        last = logits[torch.arange(len(chunk), device=device), (lens - 1).to(device)][:, choice]
        pick = last.argmax(-1).cpu().tolist()
        for r, p in zip(chunk, pick):
            per_item.append(int(p == int(r["label"])))
    correct = sum(per_item)
    return {"acc": correct / max(len(rows), 1), "n": len(rows), "items": per_item}


def entity_f1(gold: list[list[str]], pred: list[list[str]]) -> float:
    """Entity-level F1. Uses seqeval when it is installed, otherwise a span F1 of the same definition."""
    try:
        from seqeval.metrics import f1_score
        return float(f1_score(gold, pred))
    except ImportError:
        def spans(seq):
            found, cur = set(), None
            for i, tag in enumerate(seq):
                if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[0] != tag[2:])):
                    if cur:
                        found.add(cur)
                    cur = (tag[2:], i, i)
                elif tag.startswith("I-") and cur and cur[0] == tag[2:]:
                    cur = (cur[0], cur[1], i)
                else:
                    if cur:
                        found.add(cur)
                    cur = None
            if cur:
                found.add(cur)
            return found
        tp = fp = fn = 0
        for g, p in zip(gold, pred):
            gs, ps = spans(g), spans(p)
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def default_lr(regime: str) -> float:
    return 5e-4 if regime in ("memory", "tables") else 5e-5


def build_opt(model, head, lr: float, device, table_mult: float = 4.0):
    from .train import param_group_of

    groups = {"decay": [], "no_decay": [], "tables": [], "router": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[param_group_of(name, p)].append(p)
    spec = {"decay": (1.0, 0.1), "no_decay": (1.0, 0.0), "tables": (table_mult, 0.0), "router": (1.0, 0.0)}
    pg = [{"params": ps, "lr_mult": spec[g][0], "weight_decay": spec[g][1], "lr": lr * spec[g][0]}
          for g, ps in groups.items() if ps]
    if head is not None:
        pg.append({"params": list(head.parameters()), "lr_mult": 1e-3 / lr if lr < 1e-3 else 1.0,
                   "weight_decay": 0.0, "lr": max(lr, 1e-3)})
    return torch.optim.AdamW(pg, betas=(0.9, 0.95), eps=1e-8)


def run(args) -> dict:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.synthetic or args.ckpt is None:
        from .config import load_config
        cfg = load_config(args.config)
        model = Transformer(cfg["model"]).to(device)
        tok = _ByteTok()
    else:
        model, _ = load_model(args.ckpt, device)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    args.max_len = min(args.max_len, int(model.cfg["seq_len"]))
    regime = args.trainable
    if regime == "none":
        for p in model.parameters():
            p.requires_grad_(False)
        info = {"spec": "none", "trainable": 0, "frozen": sum(p.numel() for p in model.parameters())}
    else:
        info = apply_trainable(model, regime)
    lr = args.lr or default_lr(regime)
    train_rows = load_rows(args.task, "train", args.synthetic, args.limit)
    dev_rows = load_rows(args.task, "dev", args.synthetic, args.dev_limit)
    head = TokenHead(model.cfg["d_model"], len(NER_LABELS)).to(device) if args.task == "ner" else None
    v_ids = verbalizer_ids(tok, VERBALIZERS[args.task]) if args.task != "ner" else {}
    trains = regime != "none" or head is not None
    epochs = args.epochs if args.epochs is not None else (3 if args.task == "ner" else 1)
    steps_per_epoch = (len(train_rows) + args.batch - 1) // args.batch
    total = steps_per_epoch * epochs if trains else 0
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(0.1 * total))
    t0 = time.time()
    losses, step = [], 0
    if total:
        model.act_ckpt = regime not in ("none",) and device.type == "cuda"
        opt = build_opt(model, head, lr, device)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        params = [p for g in opt.param_groups for p in g["params"]]
        model.train()
        if regime == "none":
            model.eval()
        while step < total:
            random.shuffle(train_rows)
            for i in range(0, len(train_rows), args.batch):
                if step >= total:
                    break
                scale = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g in opt.param_groups:
                    g["lr"] = lr * g["lr_mult"] * scale
                rows = train_rows[i: i + args.batch]
                micro = max(1, -(-len(rows) // args.accum))
                loss = 0.0
                for j in range(0, len(rows), micro):
                    part = rows[j: j + micro]
                    l = _batch_loss(args.task, model, head, part, tok, device, args.max_len, v_ids) * (len(part) / len(rows))
                    scaler.scale(l).backward()
                    loss += float(l)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                losses.append(float(loss))
                step += 1
                if step % 100 == 0:
                    print(json.dumps({"step": step, "of": total, "loss": sum(losses[-100:]) / len(losses[-100:]),
                                      "s": round(time.time() - t0)}), flush=True)
        model.act_ckpt = False
        del opt, scaler, params
        model.zero_grad(set_to_none=True)
        if head is not None:
            head.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    train_s = time.time() - t0
    score = evaluate(args.task, model, head, dev_rows, tok, device, args.max_len, v_ids, batch=8)
    result = {
        "task": args.task, "ckpt": args.ckpt, "regime": regime, "trainable": info, "steps": step,
        "epochs": epochs, "lr": lr, "batch": args.batch, "accum": args.accum, "max_len": args.max_len, "seed": args.seed,
        "score": score, "train_seconds": round(train_s, 1), "final_loss": (sum(losses[-50:]) / len(losses[-50:])) if losses else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None,
        "synthetic": args.synthetic,
    }
    if args.save and regime not in ("none",):
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"trainable": sd, "regime": regime, "base": args.ckpt}, args.save)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "score"} | {"score": {k: v for k, v in score.items() if k != "items"}}))
    return result


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["sst2", "boolq", "ner"])
    p.add_argument("--ckpt")
    p.add_argument("--config", default="configs/tiny_cpu.yaml")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--trainable", default="all")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="split each batch into this many micro-batches")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="train rows (random subset)")
    p.add_argument("--dev_limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default=None, help="save trainable params here")
    p.add_argument("--out", default=None)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    main()

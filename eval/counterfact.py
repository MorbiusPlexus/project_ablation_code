"""CounterFact factual-recall probe (Meng et al. 2022) with memory ablation and localisation.

    python eval/counterfact.py --ckpt <ckpt> --name R2 --out_dir results --n 2000

Each item is a prompt such as "The mother tongue of Danielle Darrieux is" with the true
object (" French") and a counterfactual one (" English"). The model "knows" the fact when
log p(true) > log p(new), each summed over the object's tokens.

1. Recall accuracy and mean margin under none / zero / mean / zero@i / mean@i.
2. Localisation (memory models, items correct with memory intact): memory is zeroed or
   mean-replaced only at (a) the subject tokens, (b) the last subject token, (c) the last
   prompt token, (d) every prompt token except the subject, (e) every prompt token.
   The drop in margin says where memory contributes to recall.
Writes <out_dir>/<name>/counterfact.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import Tokenizer

from eval.means import DATA, load_or_make_means
from ffnmem.hooks import Intervention, load_model

BOS = 2


def load_items(n: int, seed: int, tok: Tokenizer) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx:
        rr = ds[i]["requested_rewrite"]
        tmpl, subj = rr["prompt"], rr["subject"]
        if "{}" not in tmpl:
            continue
        pre, post = tmpl.split("{}", 1)
        pre_ids = tok.encode(pre.rstrip(" ")).ids if pre.strip() else []
        subj_ids = tok.encode((" " if pre.strip() else "") + subj).ids
        post_ids = tok.encode(post).ids if post.strip() else []
        prompt = [BOS] + pre_ids + subj_ids + post_ids
        true = tok.encode(" " + rr["target_true"]["str"]).ids
        new = tok.encode(" " + rr["target_new"]["str"]).ids
        if not true or not new or true == new:
            continue
        s0 = 1 + len(pre_ids)
        items.append({"case_id": ds[i]["case_id"], "prompt": prompt, "true": true, "new": new,
                      "subj": (s0, s0 + len(subj_ids)), "relation": rr.get("relation_id")})
        if len(items) >= n:
            break
    return items


def positions(kind: str, it: dict, T: int) -> torch.Tensor:
    m = torch.zeros(T, dtype=torch.bool)
    s0, s1 = it["subj"]
    P = len(it["prompt"])
    if kind == "subject":
        m[s0:s1] = True
    elif kind == "subject_last":
        m[s1 - 1] = True
    elif kind == "last":
        m[P - 1] = True
    elif kind == "non_subject":
        m[:P] = True
        m[s0:s1] = False
    elif kind == "prompt":
        m[:P] = True
    return m


@torch.no_grad()
def margins(model, items: list[dict], device, mode=None, layers=None, means=None, kind=None, batch=32) -> list[float]:
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        seqs, spans = [], []
        for it in chunk:
            for obj in (it["true"], it["new"]):
                seqs.append(it["prompt"] + obj)
                spans.append((len(it["prompt"]), len(obj)))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), BOS, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
        mask = None
        if kind is not None:
            mask = torch.stack([positions(kind, it, T) for it in chunk for _ in (0, 1)])
        ctx = (Intervention(model, memory=mode, layers=layers, mean=means, positions=mask)
               if mode else nullcontext())
        with ctx, torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logp = torch.log_softmax(model(ids.to(device)).float(), -1)
        lp = []
        for j, (p0, L) in enumerate(spans):
            tgt = ids[j, p0:p0 + L].to(device)
            lp.append(logp[j, p0 - 1:p0 - 1 + L].gather(-1, tgt[:, None]).sum().item())
        out += [lp[2 * k] - lp[2 * k + 1] for k in range(len(chunk))]
        del logp
    return out


def summary(m: list[float]) -> dict:
    return {"acc": sum(x > 0 for x in m) / len(m), "margin": sum(m) / len(m), "n": len(m)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out_dir", default="results/eval")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_dir", default=DATA, help="ClimbMix shards for the mean vectors")
    a = p.parse_args()
    out_path = Path(a.out_dir) / a.name / "counterfact.json"
    if out_path.exists():
        print(f"exists {out_path}")
        return
    device = torch.device(a.device)
    tok = Tokenizer.from_file(a.tokenizer)
    model, state = load_model(a.ckpt, device)
    seq_len = int(state["config"]["model"]["seq_len"])
    del state
    mem_layers = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    t0 = time.time()
    items = load_items(a.n, a.seed, tok)
    base = margins(model, items, device)
    res = {"name": a.name, "n": len(items), "memory_layers": mem_layers, "recall": {"none": summary(base)},
           "per_item_none": base, "case_ids": [it["case_id"] for it in items]}
    if mem_layers:
        means = load_or_make_means(model, Path(a.out_dir) / a.name, seq_len, device, a.data_dir)
        specs = [("zero", None), ("mean", None)]
        if len(mem_layers) > 1:
            specs += [(m, {l}) for m in ("zero", "mean") for l in mem_layers]
        per = {}
        for mode, layers in specs:
            key = mode if layers is None else f"{mode}@{next(iter(layers))}"
            per[key] = margins(model, items, device, mode, layers, means)
            res["recall"][key] = summary(per[key])
            print(key, res["recall"][key], flush=True)
        res["per_item_zero"] = per["zero"]
        known = [it for it, m in zip(items, base) if m > 0]
        kb = [m for m in base if m > 0]
        loc = {}
        for mode in ("zero", "mean"):
            for kind in ("subject", "subject_last", "last", "non_subject", "prompt"):
                m = margins(model, known, device, mode, None, means, kind)
                loc[f"{mode}:{kind}"] = {"acc": sum(x > 0 for x in m) / len(m),
                                         "margin_drop": sum(b - x for b, x in zip(kb, m)) / len(m)}
                print(mode, kind, loc[f"{mode}:{kind}"], flush=True)
        res["localisation"] = {"n_known": len(known), "base_margin": sum(kb) / max(len(kb), 1), **loc}
    res["seconds"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["recall"]), flush=True)


if __name__ == "__main__":
    main()

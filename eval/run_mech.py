"""Mechanistic diagnostics of one checkpoint (Engram 6.1, Lngram B.3 / C.1 / Table 8).

    python eval/run_mech.py --ckpt <final>.pt --name R2 --parts logitlens,propagation,codes,fewnerd,delay

Writes ``<out_dir>/<name>/mech/<part>.json``; ``fewnerd`` saves per-layer hidden
states at the final token of each Few-NERD entity (``fewnerd_states.pt``) that
``eval/cka.py`` turns into CKA matrices across models, plus route codes for the
ID-semantics analysis. ``delay`` is the S1 inference-time injection-delay curve.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ffnmem.analysis_mech import code_utilisation, logitlens_kl, propagation
from ffnmem.data.loader import ValWindows
from ffnmem.hooks import Intervention, load_model
from ffnmem.model import Transformer


def _amp(device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def val_windows(data_dir, seq_len, n):
    val = ValWindows(data_dir, seq_len, 57_000_000)
    return [x for x, _ in val.batches(1, n)]


def part_logitlens(model, wins, device):
    acc = None
    for x in wins:
        with _amp(device):
            kl = logitlens_kl(model, x)
        acc = kl if acc is None else [a + b for a, b in zip(acc, kl)]
    return {"kl_per_layer": [a / len(wins) for a in acc], "windows": len(wins)}


def part_propagation(model, wins, device, mem_layers):
    sets = [{l} for l in mem_layers] + ([set(mem_layers)] if len(mem_layers) > 1 else [])
    out = {}
    for s in sets:
        runs = []
        for x in wins:
            with _amp(device):
                runs.append(propagation(model, x, s))
        agg = {k: sum(r[k] for r in runs) / len(runs) for k in ("final_hidden_R", "logit_R", "js", "top1_agreement")}
        agg["per_layer"] = [
            {"layer": i, "R": sum(r["per_layer"][i]["R"] for r in runs) / len(runs),
             "retention": sum(r["per_layer"][i]["retention"] for r in runs) / len(runs)}
            for i in range(len(runs[0]["per_layer"]))
        ]
        agg["injection_cosine"] = runs[0]["injection_cosine"]
        out["+".join(map(str, sorted(s)))] = agg
    return out


def part_codes(model, wins, device):
    import math

    with torch.no_grad(), _amp(device):
        diag = code_utilisation(model, [(x, None) for x in wins], device, max_batches=len(wins))
    for d in diag.values():
        if "eff_codes_mean" in d:
            d["norm_entropy_of_mean_eff"] = math.log(d["eff_codes_mean"]) / math.log(16)
    return diag


def part_delay(state, device, data_dir, n_windows, max_delay=6):
    cfg = copy.deepcopy(state["config"]["model"])
    mem = list(cfg["memory_layers"])
    wins = ValWindows(data_dir, cfg["seq_len"], 57_000_000)
    rows = []
    for n in range(0, max_delay + 1):
        c = copy.deepcopy(cfg)
        c["memory_injection_map"] = {p: p + n for p in mem}
        if max(c["memory_injection_map"].values()) >= len(c["layers"]):
            break
        m = Transformer(c)
        m.load_state_dict(state["model"])
        m.set_attention_backend("sdpa")
        m.to(device).eval()
        tot, cnt = 0.0, 0
        with torch.no_grad(), _amp(device):
            for x, y in wins.batches(4, n_windows // 4):
                logits = m(x.to(device))
                tot += float(torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]), y.to(device).view(-1), reduction="sum"))
                cnt += y.numel()
        rows.append({"delay": n, "val_loss": tot / cnt, "tokens": cnt})
        print(json.dumps(rows[-1]), flush=True)
        del m
        torch.cuda.empty_cache()
    return {"memory_layers": mem, "curve": rows}


def fewnerd_items(tok, n_sent: int, max_len: int = 256):
    """Sentences from Few-NERD supervised test; final sub-token index of each entity span."""
    from datasets import load_dataset

    ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="test")
    coarse_names = ds.features["ner_tags"].feature.names
    fine_names = ds.features["fine_ner_tags"].feature.names
    items = []
    for r in ds.select(range(min(n_sent, len(ds)))):
        ids, ends, word_tok = [2], [], []
        for k, w in enumerate(r["tokens"]):
            piece = tok.encode(w if k == 0 else " " + w).ids or [tok.encode(" ").ids[0]]
            if len(ids) + len(piece) > max_len:
                break
            word_tok.append((len(ids), len(ids) + len(piece) - 1))
            ids += piece
        tags = r["ner_tags"][: len(word_tok)]
        fine = r["fine_ner_tags"][: len(word_tok)]
        k = 0
        while k < len(tags):
            if tags[k] == 0:
                k += 1
                continue
            j = k
            while j + 1 < len(tags) and tags[j + 1] == tags[k] and fine[j + 1] == fine[k]:
                j += 1
            ends.append({"pos": word_tok[j][1], "coarse": coarse_names[tags[k]], "fine": fine_names[fine[k]],
                         "n_words": j - k + 1})
            k = j + 1
        # Token-level labels for the ID-semantics readout (entity tokens and non-entity tokens).
        tok_labels = []
        for (a, b), t, f in zip(word_tok, tags, fine):
            for p in range(a, b + 1):
                tok_labels.append({"pos": p, "coarse": coarse_names[t], "fine": fine_names[f], "first": p == a})
        items.append({"ids": ids, "ends": ends, "tokens": tok_labels, "id": r["id"]})
    return items


@torch.no_grad()
def part_fewnerd(model, tok, device, out_path: Path, n_sent: int):
    items = fewnerd_items(tok, n_sent)
    mem_pos = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    states, labels, codes, code_meta = [], [], {p: [] for p in mem_pos}, []
    for sent, it in enumerate(items):
        x = torch.tensor([it["ids"]], device=device)
        captured = {}
        hooks = []
        for p in mem_pos:
            mem = model.blocks[p].memory
            orig = mem.forward

            def fwd(h, h_norm, _orig=orig, _p=p, _mem=mem):
                captured[_p] = (_mem.module.quantize_route_logits(_mem.route_logits(h_norm))[0].to(torch.uint8).cpu(),
                                _mem.route_logits(h_norm)[0].half().cpu())
                return _orig(h, h_norm)

            mem.forward = fwd
            hooks.append((mem, orig))
        try:
            with _amp(device):
                _, extra = model(x, return_layers=True)
        finally:
            for mem, orig in hooks:
                mem.forward = orig
        if it["ends"]:
            pos = torch.tensor([e["pos"] for e in it["ends"]], device=device)
            states.append(torch.stack([h[0, pos] for h in extra["layers"]]).half().cpu())  # [L, n, d]
            labels += [{"sent": sent, **e} for e in it["ends"]]
        for t in it["tokens"]:
            code_meta.append({"sent": sent, "doc": it["id"], **t})
        tp = torch.tensor([t["pos"] for t in it["tokens"]], dtype=torch.long)
        for p in mem_pos:
            codes[p].append(captured[p][0][tp])
        # Final-layer hidden and router logits at every labelled token (for the ID-semantics baselines).
        it["h_final"] = extra["layers"][-1][0, tp].half().cpu()
        if mem_pos:
            it["q_route"] = {p: captured[p][1][tp] for p in mem_pos}
    blob = {
        "states": torch.cat(states, dim=1) if states else None,  # [L, N_entities, d]
        "labels": labels,
        "codes": {p: torch.cat(v) for p, v in codes.items()} if mem_pos else {},
        "code_meta": code_meta,
        "h_final": torch.cat([it["h_final"] for it in items]),
        "q_route": {p: torch.cat([it["q_route"][p] for it in items]) for p in mem_pos} if mem_pos else {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    return {"sentences": len(items), "entities": len(labels), "labelled_tokens": len(code_meta), "file": str(out_path)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data_dir", default=os.path.expanduser("~/morbius_data/climbmix"))
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--parts", default="logitlens,propagation,codes,fewnerd")
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--fewnerd_sentences", type=int, default=3000)
    p.add_argument("--ctx", type=int, default=2048, help="tokens per val window (cut for small GPUs)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="results/eval")
    args = p.parse_args(argv)
    from tokenizers import Tokenizer

    device = torch.device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    model, state = load_model(args.ckpt, device)
    seq_len = int(state["config"]["model"]["seq_len"])
    mem_layers = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    root = Path(args.out_dir) / args.name / "mech"
    wins = None
    for part in [x for x in args.parts.split(",") if x]:
        out = root / f"{part}.json"
        if out.exists():
            print(f"exists {out}", flush=True)
            continue
        if part in ("propagation", "codes", "delay") and not mem_layers:
            continue
        t0 = time.time()
        if wins is None and part in ("logitlens", "propagation", "codes"):
            wins = [x[:, : args.ctx] for x in val_windows(args.data_dir, seq_len, args.windows)]
        if part == "logitlens":
            res = part_logitlens(model, wins, device)
        elif part == "propagation":
            res = part_propagation(model, wins[:4], device, mem_layers)
        elif part == "codes":
            res = part_codes(model, wins, device)
        elif part == "fewnerd":
            res = part_fewnerd(model, tok, device, root / "fewnerd_states.pt", args.fewnerd_sentences)
        elif part == "delay":
            res = part_delay(state, device, args.data_dir, 64)
        else:
            raise SystemExit(f"unknown part {part}")
        res = {"name": args.name, "part": part, "seconds": round(time.time() - t0, 1), "result": res}
        _write(out, res)
        print(json.dumps({"name": args.name, "part": part, "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

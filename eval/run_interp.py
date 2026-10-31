"""Memory interpretability on a real checkpoint (plan section 6).

    python eval/run_interp.py --ckpt <final>.pt --name R2 --parts attribution,entries,patching,locality,idsem,viz

Parts (each writes ``<out_dir>/<name>/interp/<part>.json``):
- attribution (6b): per-token delta loss / KL / top-1 flip under mean and zero ablation,
  Sink mass and memory/residual norm, bucketed by token class, train-set bigram and
  trigram frequency, copyability (target bigram seen earlier in the window) and position.
- entries (6c): row hit statistics, Zipf concentration, n-gram purity of the top rows,
  code-token mutual information (current vs previous token, shuffle-corrected),
  lexical determinism of codes, direct logit attribution vs a norm-matched random vector.
- patching (6d): Few-NERD multi-token entities, first entity token corrupted; denoising
  patches of the residual (layer x {corrupt, last} position), attention and FFN outputs,
  and the memory output at every position; noising for the residual grid.
- locality (6d): mean-ablate memory at one position, KL at offsets 0..16.
- idsem (6f): ID-only KNN on route-code Hamming vs H_model, Q_route and nuisance;
  excess AP retained, stratified permutation p; code-concept associations with a
  nuisance-matched baseline, held-out replication and BH; top-m route curve.
- viz (6e): HTML heatmaps of (1 - Sink mass) and delta loss on val passages.
``idsem`` reads ``<name>/mech/fewnerd_states.pt`` from ``eval/run_mech.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from ffnmem.analysis_mech import benjamini_hochberg
from ffnmem.data.loader import ValWindows
from ffnmem.hooks import Intervention, load_model, memory_means
from ffnmem.memory.lngram_v2 import _aligned_ngram_address
from ffnmem.model import Transformer
from ffnmem.probe_memory import sink_per_token

V = 32768


def _amp(device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


class MemTap:
    """Captures (h, h_norm, codes, out) of every memory block for one forward."""

    def __init__(self, model):
        self.model = model
        self.data: dict[int, dict] = {}
        self._saved = []

    def __enter__(self):
        for pos, b in enumerate(self.model.blocks):
            if b.memory is None:
                continue
            mem = b.memory
            orig = mem.forward

            def fwd(h, h_norm, _orig=orig, _pos=pos, _mem=mem):
                out = _orig(h, h_norm)
                codes = _mem.module.quantize_route_logits(_mem.route_logits(h_norm))
                self.data[_pos] = {"h": h.detach(), "h_norm": h_norm.detach(), "codes": codes.detach(), "out": out.detach()}
                return out

            mem.forward = fwd
            self._saved.append((mem, orig))
        return self

    def __exit__(self, *exc):
        for mem, orig in self._saved:
            mem.forward = orig
        return False


# ------------------------------------------------------------------ n-gram counts
class NgramCounts:
    def __init__(self, shard: Path, max_tokens: int = 60_000_000):
        toks = np.memmap(shard, dtype=np.uint16, mode="r")[:max_tokens].astype(np.int64)
        self.n_tokens = len(toks)
        bi = toks[:-1] * V + toks[1:]
        self.bi_keys, self.bi_counts = np.unique(bi, return_counts=True)
        tri = bi[:-1] * V + toks[2:]
        self.tri_keys, self.tri_counts = np.unique(tri, return_counts=True)

    @staticmethod
    def _lookup(keys, counts, q):
        idx = np.searchsorted(keys, q)
        idx = np.clip(idx, 0, len(keys) - 1)
        return np.where(keys[idx] == q, counts[idx], 0)

    def bigram(self, a, b):
        return self._lookup(self.bi_keys, self.bi_counts, a.astype(np.int64) * V + b)

    def trigram(self, a, b, c):
        return self._lookup(self.tri_keys, self.tri_counts, (a.astype(np.int64) * V + b) * V + c)


def freq_bin(c: np.ndarray) -> np.ndarray:
    edges = np.array([1, 10, 100, 1000, 10000])
    return np.searchsorted(edges, c, side="right")  # 0: unseen, 1: 1-9, ..., 5: >=10k


FREQ_NAMES = ["0", "1-9", "10-99", "100-999", "1k-10k", ">=10k"]


def token_classes(tok, ids: np.ndarray) -> list[str]:
    out = []
    for i in ids.tolist():
        s = tok.id_to_token(int(i)) or ""
        if "Ċ" in s:
            out.append("newline")
        elif s.startswith("Ġ") and s[1:].isalpha():
            out.append("word_start_cap" if s[1:2].isupper() else "word_start")
        elif s.isalpha():
            out.append("word_cont")
        elif any(ch.isdigit() for ch in s):
            out.append("digit")
        else:
            out.append("punct_other")
    return out


# ------------------------------------------------------------------ 6b attribution
@torch.no_grad()
def part_attribution(model, tok, wins, means, counts, device):
    rec = defaultdict(list)
    mem_pos = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    for x, y in wins:
        x, y = x.to(device), y.to(device)
        with _amp(device), MemTap(model) as tap:
            full = _logits(model(x)).float()
        sinks = {p: sink_per_token(model.blocks[p].memory, d["h"], d["h_norm"])[0].float().cpu().numpy() for p, d in tap.data.items()}
        ratio = {p: (d["out"].float().norm(dim=-1) / d["h"].float().norm(dim=-1).clamp_min(1e-6))[0].cpu().numpy() for p, d in tap.data.items()}
        lf = torch.log_softmax(full, -1)
        loss_full = -lf[0].gather(-1, y[0, :, None])[:, 0]
        for mode in ("mean", "zero"):
            with _amp(device), Intervention(model, memory=mode, mean=means):
                abl = _logits(model(x)).float()
            la = torch.log_softmax(abl, -1)
            rec[f"dloss_{mode}"].append((-la[0].gather(-1, y[0, :, None])[:, 0] - loss_full).cpu().numpy())
            rec[f"kl_{mode}"].append((lf[0].exp() * (lf[0] - la[0])).sum(-1).cpu().numpy())
            rec[f"flip_{mode}"].append((abl[0].argmax(-1) != full[0].argmax(-1)).float().cpu().numpy())
            del abl, la
        for p in mem_pos:
            rec[f"sink_{p}"].append(sinks[p])
            rec[f"ratio_{p}"].append(ratio[p])
        xs = x[0].cpu().numpy()
        ys = y[0].cpu().numpy()
        prev = np.concatenate([[2], xs[:-1]])
        prev2 = np.concatenate([[2, 2], xs[:-2]])
        rec["bigram"].append(freq_bin(counts.bigram(prev, xs)))
        rec["trigram"].append(freq_bin(counts.trigram(prev2, prev, xs)))
        rec["target_bigram"].append(freq_bin(counts.bigram(xs, ys)))
        seen, cop = set(), np.zeros(len(xs), dtype=bool)
        for t in range(len(xs)):
            key = (int(xs[t]), int(ys[t]))
            cop[t] = key in seen
            seen.add(key)
        rec["copyable"].append(cop)
        rec["pos"].append(np.searchsorted([16, 128, 1024], np.arange(len(xs)), side="right"))
        rec["cls"].append(np.array(token_classes(tok, xs)))
    arr = {k: np.concatenate(v) for k, v in rec.items()}
    metrics = [k for k in arr if k.startswith(("dloss", "kl", "flip", "sink", "ratio"))]
    out = {"tokens": int(len(arr["cls"])), "overall": {m: float(arr[m].mean()) for m in metrics}, "buckets": {}}
    names = {
        "cls": None, "bigram": FREQ_NAMES, "trigram": FREQ_NAMES, "target_bigram": FREQ_NAMES,
        "copyable": ["novel", "copyable"], "pos": ["0-15", "16-127", "128-1023", "1024+"],
    }
    for key, labels in names.items():
        col = arr[key]
        vals = sorted(set(col.tolist()), key=str)
        res = {}
        for v in vals:
            mask = col == v
            name = labels[int(v)] if labels else str(v)
            res[name] = {"n": int(mask.sum()), **{m: float(arr[m][mask].mean()) for m in metrics}}
        out["buckets"][key] = res
    return out, arr


# ------------------------------------------------------------------ 6c entries
def _mi(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in mutual information (nats) between two discrete arrays."""
    n = len(a)
    ja = Counter(zip(a.tolist(), b.tolist()))
    ca, cb = Counter(a.tolist()), Counter(b.tolist())
    return sum(c / n * math.log(c * n / (ca[x] * cb[y])) for (x, y), c in ja.items())


def _entropy(a: np.ndarray) -> float:
    n = len(a)
    return -sum(c / n * math.log(c / n) for c in Counter(a.tolist()).values())


@torch.no_grad()
def part_entries(model, tok, wins, device, n_routes_mi: int = 16):
    caps = defaultdict(list)
    ids_all, tgt_all, final_rms = [], [], []
    dla = defaultdict(lambda: {"rank": [], "rand_rank": [], "top10": [], "rand_top10": []})
    W = model.output.weight.float()
    g = model.norm.weight.float()
    examples = defaultdict(list)
    for wi, (x, y) in enumerate(wins):
        x, y = x.to(device), y.to(device)
        with _amp(device), MemTap(model) as tap:
            _, extra = model(x, return_layers=True)
        hfin = extra["layers"][-1][0].float()
        rms = hfin.pow(2).mean(-1, keepdim=True).add(model.norm.eps if hasattr(model.norm, "eps") else 1e-5).sqrt()
        for p, d in tap.data.items():
            caps[p].append(d["codes"][0].to(torch.int16).cpu().numpy())
            m = d["out"][0].float()
            for kind, vec in (("mem", m), ("rand", torch.randn_like(m) * (m.norm(dim=-1, keepdim=True) / math.sqrt(m.shape[-1])))):
                lg = (vec / rms * g) @ W.T
                tl = lg.gather(-1, y[0, :, None])
                rank = (lg > tl).sum(-1).float()
                key = "rank" if kind == "mem" else "rand_rank"
                dla[p][key].append(rank.cpu().numpy())
                dla[p]["top10" if kind == "mem" else "rand_top10"].append((rank < 10).float().cpu().numpy())
                if kind == "mem" and wi == 0:
                    norms = m.norm(dim=-1)
                    for t in norms.topk(8).indices.tolist():
                        top = lg[t].topk(8).indices.tolist()
                        examples[p].append({
                            "context": tok.decode(x[0, max(0, t - 8): t + 1].tolist()),
                            "next": tok.decode([int(y[0, t])]),
                            "promoted": [tok.decode([i]) for i in top],
                        })
        ids_all.append(x[0].cpu().numpy())
        tgt_all.append(y[0].cpu().numpy())
    ids = np.concatenate(ids_all)
    prev = np.concatenate([np.concatenate([[2], a[:-1]]) for a in ids_all])
    out = {"tokens": int(len(ids)), "layers": {}}
    for p, chunks in caps.items():
        mem = model.blocks[p].memory.module
        codes = np.concatenate(chunks)  # [N, R]
        R = codes.shape[1]
        layer = {"routes": R}
        # Row hits per n-gram order: windows never cross the val windows we stitched together.
        for order in mem.ngrams:
            rows_all, grams_all = [], []
            for c, a in zip(chunks, ids_all):
                ct = torch.from_numpy(c.astype(np.int64))[None]
                addr = _aligned_ngram_address(ct, order, mem.bits_per_route)[0].numpy()  # [T-o+1, R]
                rows = addr + (np.arange(R) * (mem.alphabet_size ** order))[None]
                rows_all.append(rows)
                gr = np.stack([a[k: len(a) - order + 1 + k] for k in range(order)], -1)
                grams_all.append(gr)
            rows = np.concatenate(rows_all)  # [N', R]
            grams = np.concatenate(grams_all)  # [N', order]
            n_rows = R * mem.alphabet_size ** order
            hits = np.bincount(rows.reshape(-1), minlength=n_rows)
            hit_sorted = np.sort(hits[hits > 0])[::-1]
            tot = hit_sorted.sum()
            top = lambda f: float(hit_sorted[: max(1, int(len(hit_sorted) * f))].sum() / tot)
            rr = np.arange(1, min(len(hit_sorted), 2000) + 1)
            slope = float(np.polyfit(np.log(rr), np.log(hit_sorted[: len(rr)]), 1)[0]) if len(rr) > 10 else None
            # Purity: share of each row's hits taken by its most common token n-gram.
            gram_key = grams[:, 0].astype(np.int64)
            for k in range(1, order):
                gram_key = gram_key * V + grams[:, k]
            flat_rows = rows.reshape(-1)
            flat_gram = np.repeat(gram_key, R)
            busy = np.argsort(-hits)[:50]
            purity, top_rows = [], []
            busy_set = set(busy.tolist())
            sel = np.isin(flat_rows, busy)
            per_row = defaultdict(Counter)
            for r_, g_ in zip(flat_rows[sel].tolist(), flat_gram[sel].tolist()):
                per_row[r_][g_] += 1
            for r_ in busy.tolist():
                cnt = per_row[r_]
                if not cnt:
                    continue
                (g1, c1), = cnt.most_common(1)
                purity.append(c1 / sum(cnt.values()))
                if len(top_rows) < 10:
                    def dec(key):
                        parts = []
                        for _ in range(order):
                            parts.append(int(key % V))
                            key //= V
                        return tok.decode(parts[::-1])
                    top_rows.append({"route": int(r_ // mem.alphabet_size ** order), "hits": int(hits[r_]),
                                     "top_ngrams": [(dec(k), c) for k, c in cnt.most_common(3)],
                                     "distinct_ngrams": len(cnt)})
            # Weighted purity over all rows with >= 5 hits, on a sample of routes for speed.
            layer[f"order{order}"] = {
                "rows": n_rows, "rows_hit_frac": float((hits > 0).mean()),
                "top1pct_share": top(0.01), "top10pct_share": top(0.10), "zipf_slope": slope,
                "top50_purity_mean": float(np.mean(purity)) if purity else None, "top_rows": top_rows,
            }
        # Code-token mutual information, shuffle-corrected, on a subset of routes.
        rng = np.random.default_rng(0)
        mi_cur, mi_prev, determinism = [], [], []
        for r in rng.choice(R, size=min(n_routes_mi, R), replace=False).tolist():
            c = codes[:, r]
            h = _entropy(c)
            shuf = rng.permutation(ids)
            mi_cur.append((_mi(c, ids) - _mi(c, shuf)) / h)
            mi_prev.append((_mi(c, prev) - _mi(c, rng.permutation(prev))) / h)
        # Lexical determinism: share of a frequent token's occurrences using its modal code.
        freq = [t for t, n in Counter(ids.tolist()).items() if n >= 20]
        for t in freq[:300]:
            rows_t = codes[ids == t]
            determinism.append(float(np.mean([(rows_t[:, r] == np.bincount(rows_t[:, r]).argmax()).mean() for r in range(R)])))
        layer["code_token_mi_norm"] = {"current_token": float(np.mean(mi_cur)), "previous_token": float(np.mean(mi_prev))}
        layer["lexical_determinism"] = {"tokens": len(determinism), "mean_modal_share": float(np.mean(determinism)) if determinism else None}
        d = dla[p]
        rk, rrk = np.concatenate(d["rank"]), np.concatenate(d["rand_rank"])
        layer["direct_logit"] = {
            "median_rank": float(np.median(rk)), "random_median_rank": float(np.median(rrk)),
            "top10_rate": float(np.concatenate(d["top10"]).mean()), "random_top10_rate": float(np.concatenate(d["rand_top10"]).mean()),
            "examples": examples[p],
        }
        out["layers"][str(p)] = layer
    return out


# ------------------------------------------------------------------ 6d patching
def entity_pairs(tok, n_items: int, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="test")
    rng = random.Random(seed)
    starts = [i for i in range(V) if (tok.id_to_token(i) or "").startswith("Ġ") and (tok.id_to_token(i) or "")[1:].isalpha()]
    pairs = []
    for r in ds:
        if len(pairs) >= n_items:
            break
        words, tags = r["tokens"], r["ner_tags"]
        k = 0
        while k < len(tags):
            if tags[k] == 0:
                k += 1
                continue
            j = k
            while j + 1 < len(tags) and tags[j + 1] == tags[k]:
                j += 1
            if k > 0:
                ids = [2]
                for w_i, w in enumerate(words[: j + 1]):
                    piece = tok.encode(w if w_i == 0 else " " + w).ids
                    if w_i == k:
                        ent_start = len(ids)
                    ids += piece
                if len(ids) - ent_start >= 3 and len(ids) <= 128:
                    clean = ids[:-1]
                    target = ids[-1]
                    corrupt = list(clean)
                    corrupt[ent_start] = rng.choice(starts)
                    pairs.append({"clean": clean, "corrupt": corrupt, "target": target, "pos": ent_start,
                                  "text": tok.decode(ids)})
                    break
            k = j + 1
    return pairs


@torch.no_grad()
def part_patching(model, tok, device, n_items: int):
    pairs = entity_pairs(tok, n_items)
    L = len(model.blocks)
    mem_pos = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    acc = defaultdict(list)
    kept = 0

    def tlogp(x, target, **kw):
        with _amp(device), Intervention(model, **kw):
            lg = _logits(model(x)).float()
        return float(torch.log_softmax(lg[0, -1], -1)[target])

    for it in pairs:
        xc = torch.tensor([it["clean"]], device=device)
        xk = torch.tensor([it["corrupt"]], device=device)
        T = xc.shape[1]
        store_c, store_k = {}, {}
        with _amp(device), Intervention(model, memory="capture", attn="capture", ffn="capture", residual="capture", store=store_c):
            lc = float(torch.log_softmax(_logits(model(xc)).float()[0, -1], -1)[it["target"]])
        with _amp(device), Intervention(model, memory="capture", attn="capture", ffn="capture", residual="capture", store=store_k):
            lk = float(torch.log_softmax(_logits(model(xk)).float()[0, -1], -1)[it["target"]])
        gap = lc - lk
        if gap < 1.0:
            continue
        kept += 1
        positions = {"corrupt": it["pos"], "last": T - 1}
        for site in ("residual", "attn", "ffn"):
            for pname, pidx in positions.items():
                mask = torch.zeros(1, T, dtype=torch.bool, device=device)
                mask[0, pidx] = True
                row_d, row_n = [], []
                for l in range(L):
                    src = {site: {l: store_c[site][l][0]}}
                    pl = tlogp(xk, it["target"], **{site: "patch"}, layers={l}, patch=src, positions=mask)
                    row_d.append((pl - lk) / gap)
                    if site == "residual":
                        srcn = {site: {l: store_k[site][l][0]}}
                        pn = tlogp(xc, it["target"], **{site: "patch"}, layers={l}, patch=srcn, positions=mask)
                        row_n.append((lc - pn) / gap)
                acc[f"{site}_{pname}_denoise"].append(row_d)
                if row_n:
                    acc[f"{site}_{pname}_noise"].append(row_n)
        for l in mem_pos:
            rel = []
            for t in range(T):
                mask = torch.zeros(1, T, dtype=torch.bool, device=device)
                mask[0, t] = True
                pl = tlogp(xk, it["target"], memory="patch", layers={l}, patch={"memory": {l: store_c["memory"][l][0]}}, positions=mask)
                rel.append((t - it["pos"], (pl - lk) / gap))
            acc[f"memory_{l}_by_offset"].append(rel)
            # All positions at once: the memory's total causal share.
            pl = tlogp(xk, it["target"], memory="patch", layers={l}, patch={"memory": {l: store_c["memory"][l][0]}})
            acc[f"memory_{l}_all_positions"].append(pl - lk)
            acc[f"memory_{l}_all_positions_norm"].append((pl - lk) / gap)
    out = {"items": len(pairs), "kept_gap_ge_1nat": kept, "layers": L, "memory_layers": mem_pos}
    for k, v in acc.items():
        if k.endswith("_by_offset"):
            by = defaultdict(list)
            for rel in v:
                for off, val in rel:
                    by[off].append(val)
            out[k] = {str(o): float(np.mean(by[o])) for o in sorted(by) if -8 <= o <= 8}
        elif isinstance(v[0], list):
            out[k] = np.mean(np.array(v), axis=0).tolist()
        else:
            out[k] = float(np.mean(v))
    return out


@torch.no_grad()
def part_locality(model, wins, means, device, n_pos: int = 48, horizon: int = 16):
    rng = random.Random(0)
    kl = defaultdict(list)
    for x, _ in wins[:4]:
        x = x.to(device)
        with _amp(device):
            full = torch.log_softmax(_logits(model(x)).float()[0], -1)
        for t in rng.sample(range(64, x.shape[1] - horizon - 1), n_pos // 4):
            mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=device)
            mask[0, t] = True
            with _amp(device), Intervention(model, memory="mean", mean=means, positions=mask):
                abl = torch.log_softmax(_logits(model(x)).float()[0], -1)
            seg = slice(t, t + horizon + 1)
            vals = (full[seg].exp() * (full[seg] - abl[seg])).sum(-1).cpu().tolist()
            for off, v in enumerate(vals):
                kl[off].append(v)
            kl[-1].append(float((full[t - 1].exp() * (full[t - 1] - abl[t - 1])).sum()))
    return {"kl_by_offset": {str(k): float(np.mean(v)) for k, v in sorted(kl.items())}, "positions": len(kl[0])}


# ------------------------------------------------------------------ 6f ID semantics
def _macro_ap(scores: np.ndarray, labels: np.ndarray, classes: list[int]) -> float:
    from sklearn.metrics import average_precision_score

    aps = [average_precision_score(labels == c, scores[:, k]) for k, c in enumerate(classes) if (labels == c).any()]
    return float(np.mean(aps))


def _knn_votes(nbr: np.ndarray, ref_y: np.ndarray, classes: list[int]) -> np.ndarray:
    lab = ref_y[nbr]  # [Q, K]
    return np.stack([(lab == c).mean(1) for c in classes], 1)


def _knn(q: torch.Tensor, r: torch.Tensor, q_doc, r_doc, metric: str, k: int = 10, pool: int = 60, device="cpu"):
    """Indices of the K nearest references from distinct documents (never the query's)."""
    q, r = q.to(device), r.to(device)
    out = np.zeros((q.shape[0], k), dtype=np.int64)
    r_doc = np.asarray(r_doc)
    for s in range(0, q.shape[0], 1024):
        qq = q[s: s + 1024]
        if metric == "hamming":
            d = qq.sum(1, keepdim=True) + r.sum(1)[None] - 2 * qq @ r.T
        else:
            d = -F.normalize(qq, dim=-1) @ F.normalize(r, dim=-1).T
        idx = d.topk(pool, largest=False).indices.cpu().numpy()
        for i, row in enumerate(idx):
            seen, keep = set(), []
            for j in row:
                dj = r_doc[j]
                if dj == q_doc[s + i] or dj in seen:
                    continue
                seen.add(dj)
                keep.append(j)
                if len(keep) == k:
                    break
            while len(keep) < k:
                keep.append(row[len(keep)])
            out[s + i] = keep
    return out


def part_idsem(blob, device, n_perm: int = 200, max_tokens: int = 40000):
    meta = blob["code_meta"]
    rng = np.random.default_rng(0)
    n = len(meta)
    sel = np.sort(rng.choice(n, size=min(n, max_tokens), replace=False))
    meta = [meta[i] for i in sel]
    concepts = sorted({m["coarse"] for m in meta} - {"O"})
    cidx = {c: i + 1 for i, c in enumerate(concepts)}
    y = np.array([cidx.get(m["coarse"], 0) for m in meta])
    docs = np.array([m["doc"] for m in meta])
    udocs = np.unique(docs)
    ref_docs = set(rng.choice(udocs, size=int(0.7 * len(udocs)), replace=False).tolist())
    is_ref = np.array([d in ref_docs for d in docs])
    # Nuisance strata: position bucket x word-initial flag.
    posb = np.searchsorted([4, 16, 48], np.array([m["pos"] for m in meta]), side="right")
    first = np.array([int(m["first"]) for m in meta])
    stratum = posb * 2 + first
    classes = list(range(1, len(concepts) + 1))
    ry, qy = y[is_ref], y[~is_ref]
    rdoc, qdoc = docs[is_ref], docs[~is_ref]
    # Nuisance-only predictor: class frequencies of the stratum in reference data.
    p_nuis = np.zeros((stratum.max() + 1, len(classes)))
    for s in range(stratum.max() + 1):
        m = is_ref & (stratum == s)
        if m.any():
            p_nuis[s] = [(y[m] == c).mean() for c in classes]
    ap_n = _macro_ap(p_nuis[stratum[~is_ref]], qy, classes)
    h = blob["h_final"][sel].float()
    nbr_h = _knn(h[~is_ref], h[is_ref], qdoc, rdoc, "cos", device=device)
    ap_h = _macro_ap(_knn_votes(nbr_h, ry, classes), qy, classes)
    out = {"tokens": int(len(meta)), "concepts": concepts, "AP_nuisance": ap_n, "AP_H_model": ap_h, "layers": {}}
    for p, codes_all in blob["codes"].items():
        codes = codes_all[sel].long()
        R = codes.shape[1]
        bits = ((codes[..., None] >> torch.arange(4)) & 1).reshape(len(meta), R * 4).float()
        nbr = _knn(bits[~is_ref], bits[is_ref], qdoc, rdoc, "hamming", device=device)
        ap_id = _macro_ap(_knn_votes(nbr, ry, classes), qy, classes)
        q = blob["q_route"][p][sel].float()
        nbr_q = _knn(q[~is_ref], q[is_ref], qdoc, rdoc, "cos", device=device)
        ap_q = _macro_ap(_knn_votes(nbr_q, ry, classes), qy, classes)
        # Stratified permutation of reference labels (neighbours fixed).
        rstrat = stratum[is_ref]
        perm_aps = []
        for _ in range(n_perm):
            py = ry.copy()
            for s in np.unique(rstrat):
                m = np.where(rstrat == s)[0]
                py[m] = py[rng.permutation(m)]
            perm_aps.append(_macro_ap(_knn_votes(nbr, py, classes), qy, classes))
        pval = (1 + sum(a >= ap_id for a in perm_aps)) / (1 + n_perm)
        rho = (ap_id - ap_n) / (ap_h - ap_n) if ap_h != ap_n else None
        # Code-concept associations vs the nuisance-matched baseline.
        c_np = codes.numpy()
        tests = []
        for r in range(R):
            for code in range(16):
                for k, c in enumerate(classes):
                    mref = is_ref & (c_np[:, r] == code)
                    if mref.sum() < 30:
                        continue
                    P = (y[mref] == c).mean()
                    P0 = p_nuis[stratum[mref], k].mean()
                    if abs(P - P0) < 0.05:
                        continue
                    mh = (~is_ref) & (c_np[:, r] == code)
                    nh = int(mh.sum())
                    if nh < 10:
                        continue
                    Ph = (y[mh] == c).mean()
                    P0h = p_nuis[stratum[mh], k].mean()
                    se = math.sqrt(max(P0h * (1 - P0h), 1e-9) / nh)
                    z = (Ph - P0h) / se
                    from math import erfc
                    pv = erfc(abs(z) / math.sqrt(2))
                    tests.append({"route": r, "code": code, "concept": concepts[k], "P_ref": float(P), "P0_ref": float(P0),
                                  "P_held": float(Ph), "P0_held": float(P0h), "n_held": nh, "p": pv,
                                  "same_sign": bool(np.sign(Ph - P0h) == np.sign(P - P0))})
        if tests:
            adj = benjamini_hochberg([t["p"] for t in tests])
            for t, a in zip(tests, adj):
                t["q_bh"] = a
            repl = [t for t in tests if t["same_sign"] and t["q_bh"] < 0.05]
            top = sorted(repl, key=lambda t: -(t["P_held"] - t["P0_held"]))[:15]
        else:
            repl, top = [], []
        # Distributed semantics: AP_ID from the top-m routes ranked by reference effect.
        eff = np.zeros(R)
        for t in tests:
            eff[t["route"]] = max(eff[t["route"]], abs(t["P_ref"] - t["P0_ref"]))
        order = np.argsort(-eff)
        curve = {}
        m = 1
        while True:
            mm = min(m, R)
            cols = torch.tensor(np.concatenate([np.arange(r * 4, r * 4 + 4) for r in order[:mm]]))
            nb = _knn(bits[~is_ref][:, cols], bits[is_ref][:, cols], qdoc, rdoc, "hamming", device=device)
            curve[str(mm)] = _macro_ap(_knn_votes(nb, ry, classes), qy, classes)
            if mm == R:
                break
            m *= 2
        out["layers"][str(p)] = {
            "routes": R, "AP_ID": ap_id, "AP_Q_route": ap_q, "excess_retained_rho": rho, "perm_p": pval,
            "n_perm": n_perm, "assoc_tested": len(tests), "assoc_replicated": len(repl),
            "replication_rate": len(repl) / len(tests) if tests else None, "top_assoc": top,
            "top_m_curve": curve, "AP_R_over_AP_1": curve[str(R)] / curve["1"] if curve.get("1") else None,
        }
    return out


# ------------------------------------------------------------------ 6e viz
def part_viz(model, tok, arr, out_dir: Path, n_passages: int = 8, length: int = 200):
    from ffnmem.interp.viz import write_heatmap

    mem_pos = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    files = []
    ids = arr["_ids"]
    win = int(arr["_win"])
    for k in range(n_passages):
        s = k * win
        toks = [tok.decode([int(t)]) for t in ids[s: s + length]]
        for p in mem_pos:
            f = out_dir / f"gate_L{p}_{k}.html"
            write_heatmap(toks, (1 - arr[f"sink_{p}"][s: s + length]).tolist(), f, f"1 - Sink mass, memory L{p}")
            files.append(str(f))
        f = out_dir / f"dloss_{k}.html"
        write_heatmap(toks, np.clip(arr["dloss_mean"][s: s + length], 0, None).tolist(), f, "delta loss under mean ablation")
        files.append(str(f))
    return {"files": files}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data_dir", default=os.path.expanduser("~/morbius_data/climbmix"))
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--parts", default="attribution,entries,patching,locality,idsem,viz")
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--patch_items", type=int, default=150)
    p.add_argument("--ctx", type=int, default=2048, help="tokens per val window (cut for small GPUs)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="results/eval")
    args = p.parse_args(argv)
    from tokenizers import Tokenizer

    device = torch.device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    model, state = load_model(args.ckpt, device)
    del state
    seq_len = model.cfg["seq_len"]
    mem_pos = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    root = Path(args.out_dir) / args.name / "interp"
    val = ValWindows(args.data_dir, seq_len, 57_000_000)
    ctx = min(args.ctx, seq_len)
    wins = [(x[:, :ctx], y[:, :ctx]) for x, y in val.batches(1, args.windows)]
    means = None
    if mem_pos:
        mp = Path(args.out_dir) / args.name / "memory_means.pt"
        if mp.exists():
            means = torch.load(mp)
        else:
            val2 = ValWindows(args.data_dir, seq_len, 57_000_000)
            val2.ids = val2.ids[-128:]
            means = memory_means(model, (x for x, _ in val2.batches(1)), device)
            mp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(means, mp)
    arr = None
    for part in [x for x in args.parts.split(",") if x]:
        out = root / f"{part}.json"
        if out.exists() and part != "viz":
            print(f"exists {out}", flush=True)
            continue
        if part != "patching" and not mem_pos:
            continue
        t0 = time.time()
        if part in ("attribution", "viz"):
            if arr is None:
                counts = NgramCounts(Path(args.data_dir) / "shard_00000.bin")
                res, arr = part_attribution(model, tok, wins, means, counts, device)
                arr["_ids"] = np.concatenate([x[0].numpy() for x, _ in wins])
                arr["_win"] = wins[0][0].shape[1]
                if part == "attribution":
                    _write(out, {"name": args.name, "part": part, "seconds": round(time.time() - t0, 1), "result": res})
            if part == "viz":
                res = part_viz(model, tok, arr, root / "html")
            else:
                print(json.dumps({"name": args.name, "part": part, "seconds": round(time.time() - t0, 1)}), flush=True)
                continue
        elif part == "entries":
            res = part_entries(model, tok, wins, device)
        elif part == "patching":
            res = part_patching(model, tok, device, args.patch_items)
        elif part == "locality":
            res = part_locality(model, wins, means, device)
        elif part == "idsem":
            bp = Path(args.out_dir) / args.name / "mech" / "fewnerd_states.pt"
            if not bp.exists():
                print(f"skip idsem: {bp} missing (run eval/run_mech.py --parts fewnerd)", flush=True)
                continue
            res = part_idsem(torch.load(bp), device)
        else:
            raise SystemExit(f"unknown part {part}")
        _write(out, {"name": args.name, "part": part, "seconds": round(time.time() - t0, 1), "result": res})
        print(json.dumps({"name": args.name, "part": part, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()

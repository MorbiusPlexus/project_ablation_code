"""Collect ada eval-suite results into one markdown report.

    python eval/aggregate_ada.py --root ~/morbius_eval_runs/results --out RESULTS_ADA.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ["R1", "R2", "R3", "R4", "R5", "A", "S1"]
SPECS = ["none", "zero", "mean", "zero@1", "mean@1", "zero@11", "mean@11"]


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _f(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _score(entry: dict):
    """Track entry score; falls back to any ``metric,<filter>`` key in the raw lm-eval results."""
    if not isinstance(entry, dict) or "error" in entry:
        return None
    if entry.get("score") is not None:
        return entry["score"]
    metric = entry.get("metric")
    for res in (entry.get("results") or {}).values():
        for k, v in res.items():
            if k.startswith(f"{metric},") and "_stderr" not in k and isinstance(v, (int, float)):
                return v
    return None


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def val_section(root: Path) -> str:
    rows = []
    for m in MODELS:
        for s in SPECS:
            d = _load(root / m / s / "val.json")
            syn = _load(root / m / s / "synthetic.json")
            if not d and not syn:
                continue
            r = (d or {}).get("result", {})
            sr = (syn or {}).get("result", {})
            pk = sr.get("passkey") or {}
            rows.append([m, s, _f(r.get("loss")), _f(r.get("bpb")), _f(r.get("acc")),
                         _f(sr.get("induction")), _f(sr.get("mqar")),
                         " / ".join(f"{k}:{_f(v, 2)}" for k, v in pk.items()) or "–"])
    return "## ClimbMix validation and synthetic tasks\n\n" + _table(
        ["model", "memory", "loss", "bpb", "next-tok acc", "induction", "MQAR", "passkey by length"], rows)


def lm_section(root: Path, track: str, title: str) -> str:
    tasks: list[str] = []
    data = {}
    for m in MODELS:
        for s in SPECS:
            d = _load(root / m / s / f"{track}.json")
            if not d:
                continue
            data[(m, s)] = d["result"]
            for t in d["result"]:
                if t not in tasks and t != "siqa":
                    tasks.append(t)
    if not data:
        return ""
    cols = list(data)
    rows = [[t] + [_f(_score(data[c].get(t))) for c in cols] for t in tasks]
    header = ["task"] + [f"{m} {s}" for m, s in cols]
    return f"## {title}\n\n" + _table(header, rows)


def ft_seeds_section(root: Path) -> str:
    import re
    import statistics as st

    groups: dict = {}
    for f in sorted((root / "ft").glob("*/*/*.json")) if (root / "ft").is_dir() else []:
        r = _load(f) or {}
        sc = r.get("score", {})
        s = sc.get("acc", sc.get("f1"))
        if s is None:
            continue
        m = re.match(r"(.+)_lr([0-9.e-]+)_s(\d+)$", f.stem)
        regime = m.group(1) if m else {"ffn_layers_2-3-4-5-12-13-14-15": "ffn8"}.get(f.stem, f.stem)
        lr = float(m.group(2)) if m else float(r.get("lr"))
        n_tr = (r.get("trainable") or {}).get("trainable", 0)
        g = groups.setdefault((f.parts[-3], f.parts[-2], regime, lr), {"s": [], "n": n_tr})
        g["s"].append(s)
    rows = []
    for (task, model, regime, lr), g in sorted(groups.items()):
        xs = g["s"]
        sd = st.stdev(xs) if len(xs) > 1 else None
        rows.append([task, model, regime, f"{lr:g}", f"{g['n'] / 1e6:.1f}M", str(len(xs)),
                     _f(st.mean(xs)) + ("" if sd is None else f" ± {sd:.3f}")])
    if not rows:
        return ""
    return ("## Fine-tuning across seeds (mean ± sd)\n\nSeed 0 is the original run; extra seeds and the "
            "learning-rate / parameter-matched controls come from the extra-seeds sweep.\n\n"
            + _table(["task", "model", "regime", "lr", "trainable", "seeds", "score"], rows))


def ruler_section(root: Path) -> str:
    rows, header = [], None
    for m in MODELS:
        for s in ("none", "zero", "mean"):
            d = _load(root / m / s / "ruler.json")
            if not d:
                continue
            res = d["result"]
            lengths = sorted({L for t in res.values() for L in t}, key=int)
            header = header or ["model", "memory"] + [f"{t}@{L}" for t in res for L in lengths] + ["mean"]
            vals = [res[t].get(L) for t in res for L in lengths]
            ok = [v for v in vals if v is not None]
            rows.append([m, s] + [_f(v, 2) for v in vals] + [_f(sum(ok) / len(ok))])
    if not rows:
        return ""
    return ("## Mini-RULER long context (30 items per cell; recall of the expected strings)\n\n"
            + _table(header, rows))


def counterfact_section(root: Path) -> str:
    rows, loc = [], []
    for m in MODELS:
        d = _load(root / m / "counterfact.json")
        if not d:
            continue
        rc = d["recall"]
        rows.append([m] + [_f(rc[k]["acc"]) if k in rc else "–" for k in
                           ("none", "zero", "mean", "zero@1", "zero@11", "mean@1", "mean@11")]
                    + [_f(rc["none"]["margin"], 2)])
        L = d.get("localisation")
        if L:
            loc.append([m, str(L["n_known"])] + [_f(L[f"{mo}:{k}"]["margin_drop"], 3) for mo in ("zero", "mean")
                                                 for k in ("subject", "subject_last", "last", "non_subject", "prompt")])
    if not rows:
        return ""
    out = ("## CounterFact factual recall (2000 items; acc = log p(true) > log p(counterfactual))\n\n"
           + _table(["model", "intact", "zero", "mean", "zero@1", "zero@11", "mean@1", "mean@11", "margin"], rows))
    if loc:
        out += ("\n\nMargin drop (nats) when memory is ablated only at some positions, on items known with memory intact:\n\n"
                + _table(["model", "known"] + [f"{mo} {k}" for mo in ("zero", "mean") for k in
                                               ("subject", "subj-last", "last", "non-subject", "prompt")], loc))
    return out


def _mean_score(d) -> float | None:
    if not d:
        return None
    xs = [_score(v) for v in d["result"].values()]
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def cpt_section(root: Path) -> str:
    base = {}
    for m in ("R2", "R4", "S1"):
        base[m] = {
            "pub": ((_load(root / "cpt_pubmedval" / m / "none" / "val.json") or {}).get("result") or {}).get("loss"),
            "cm": ((_load(root / m / "none" / "val.json") or {}).get("result") or {}).get("loss"),
            "zs": _mean_score(_load(root / m / "none" / "zeroshot.json")),
            "dom": _load(root / "cpt_base" / m / "none" / "domain.json"),
        }
    rows = []
    runs = sorted(p.name for p in (root / "cpt").iterdir() if p.is_dir()) if (root / "cpt").is_dir() else []
    for name in runs:
        src = name.split("_")[1]
        b = base.get(src, {})
        pub = ((_load(root / "cpt_pubmedval" / name / "none" / "val.json") or {}).get("result") or {}).get("loss")
        cm = ((_load(root / "cpt" / name / "none" / "val.json") or {}).get("result") or {}).get("loss")
        zs = _mean_score(_load(root / "cpt" / name / "none" / "zeroshot.json"))
        dom = _load(root / "cpt" / name / "none" / "domain.json")
        n_tr = None
        log = root / "cpt" / name / "train.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("event") == "start":
                    n_tr = (rec.get("trainable") or {}).get("trainable")
                    break
        pq = lambda d: _score(d["result"].get("pubmedqa")) if d else None  # noqa: E731
        rows.append([name, "–" if n_tr is None else f"{n_tr / 1e6:.1f}M",
                     f"{_f(b.get('pub'))} → {_f(pub)}", f"{_f(b.get('cm'))} → {_f(cm)}",
                     f"{_f(b.get('zs'))} → {_f(zs)}", f"{_f(_mean_score(b.get('dom')))} → {_f(_mean_score(dom))}",
                     f"{_f(pq(b.get('dom')))} → {_f(pq(dom))}"])
    if not rows:
        return ""
    return ("## PubMed continued pre-training (20M tokens, one GPU, fresh cosine schedule)\n\n"
            "Each cell is base → after. PubMed and ClimbMix columns are validation loss (lower is better); "
            "zero-shot is the mean over the 15-task zero-shot track (forgetting); domain is the mean over "
            "PubMedQA and 7 MMLU medical subjects.\n\n"
            + _table(["run", "trainable", "PubMed loss", "ClimbMix loss", "zero-shot mean", "domain mean", "PubMedQA"], rows))


def ft_section(root: Path) -> str:
    rows = []
    for task in ("sst2", "boolq", "ner"):
        for m in MODELS:
            d = root / "ft" / task / m
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.json")):
                if "_lr" in f.stem:
                    continue
                r = _load(f) or {}
                sc = r.get("score", {})
                tr = r.get("trainable", {})
                rows.append([task, m, r.get("regime", f.stem), _f(sc.get("acc", sc.get("f1"))),
                             f"{tr.get('trainable', 0) / 1e6:.1f}M", _f(r.get("train_seconds"), 0),
                             _f(r.get("peak_mem_gb"), 1)])
    return "## Fine-tuning (dev set; acc for SST-2/BoolQ, entity F1 for NER)\n\n" + _table(
        ["task", "model", "regime", "score", "trainable", "train s", "peak GB"], rows)


def mech_section(root: Path) -> str:
    rows = []
    for m in MODELS:
        p = _load(root / m / "mech" / "propagation.json")
        c = _load(root / m / "mech" / "codes.json")
        if not p:
            continue
        for key, v in p["result"].items():
            code = (c or {}).get("result", {}).get(key, {}) if c else {}
            rows.append([m, key, _f(v.get("final_hidden_R")), _f(v.get("logit_R")), _f(v.get("js"), 4),
                         _f(v.get("top1_agreement")), _f(code.get("sink_mass")), _f(code.get("eff_codes_mean"), 1)])
    ll = []
    for m in MODELS:
        d = _load(root / m / "mech" / "logitlens.json")
        if d:
            k = d["result"]["kl_per_layer"]
            ll.append([m] + [_f(k[i], 2) for i in (0, 2, 5, 11, 17, len(k) - 2)])
    out = "## Memory propagation and code use\n\n" + _table(
        ["model", "memory layers", "final-hidden R", "logit R", "JS", "top-1 agree", "sink mass", "eff. codes"], rows)
    if ll:
        out += "\n\nLogit-lens KL(final ‖ layer), nats per token:\n\n" + _table(
            ["model", "L0", "L2", "L5", "L11", "L17", f"L{len(k) - 2}"], ll)
    return out


def interp_section(root: Path) -> str:
    parts = []
    for m in MODELS:
        base = root / m / "interp"
        att, ent, idsem, loc, pat = (_load(base / f"{n}.json") for n in
                                     ("attribution", "entries", "idsem", "locality", "patching"))
        if not any((att, ent, idsem, loc, pat)):
            continue
        lines = [f"### {m}"]
        if att:
            o = att["result"]["overall"]
            lines.append(f"- Attribution: mean-ablation Δloss {_f(o['dloss_mean'])}, zero-ablation Δloss "
                         f"{_f(o['dloss_zero'])}, top-1 flip rate {_f(o['flip_mean'])}.")
            tri = att["result"]["buckets"].get("trigram", {})
            if tri:
                lines.append("  - Δloss by trigram frequency: " + ", ".join(
                    f"{b} {_f(v['dloss_mean'])}" for b, v in tri.items()))
        if ent:
            for L, v in ent["result"]["layers"].items():
                mi = v.get("code_token_mi_norm", {})
                dl = v.get("direct_logit", {})
                lines.append(f"- Entries L{L}: code–current-token MI {_f(mi.get('current_token'))}, "
                             f"previous-token {_f(mi.get('previous_token'))}; lexical determinism "
                             f"{_f(v.get('lexical_determinism', {}).get('mean_modal_share'))}; direct-logit median rank "
                             f"{_f(dl.get('median_rank'), 0)} vs random {_f(dl.get('random_median_rank'), 0)}.")
        if idsem:
            r = idsem["result"]
            for L, v in r["layers"].items():
                lines.append(f"- ID semantics L{L}: AP(ID) {_f(v['AP_ID'])} vs AP(router) {_f(v['AP_Q_route'])}, "
                             f"AP(hidden) {_f(r['AP_H_model'])}, nuisance {_f(r['AP_nuisance'])}; "
                             f"ρ {_f(v['excess_retained_rho'])}, p={_f(v['perm_p'], 3)}; "
                             f"associations replicated {v['assoc_replicated']}/{v['assoc_tested']}.")
        if loc:
            k = loc["result"]["kl_by_offset"]
            lines.append("- Locality KL by offset: " + ", ".join(f"{o}:{_f(v, 4)}" for o, v in list(k.items())[:6]))
        if pat:
            r = pat["result"]
            lines.append(f"- Patching: {r.get('kept_gap_ge_1nat')}/{r.get('items')} entity pairs kept (gap ≥ 1 nat).")
        parts.append("\n".join(lines))
    cka = []
    for f in sorted((root / "cka").glob("*.json")) if (root / "cka").is_dir() else []:
        d = _load(f)
        if d:
            cka.append(f"- {d['memory']} vs {d['dense']}: diagonal CKA (first 6 layers) "
                       + ", ".join(_f(x, 2) for x in d["diag"][:6]))
    out = "## Interpretability\n\n" + "\n\n".join(parts)
    if cka:
        out += "\n\n### CKA\n\n" + "\n".join(cka)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / "morbius_eval_runs/results"))
    p.add_argument("--out", default="RESULTS_ADA.md")
    a = p.parse_args()
    root = Path(a.root).expanduser()
    secs = [
        "# Morbius eval suite: ada results\n\nAll seven wave-1/2 checkpoints. R1 and R3 were extracted from "
        "the step-3815 run checkpoints on gnode078:/ssd_scratch/aditya.shankar. "
        "One pre-training run per architecture; fine-tuning headline regimes have 3 seeds, the rest 1. "
        "lm-eval paper/few-shot tracks use 300 items per task.",
        val_section(root),
        lm_section(root, "zeroshot", "Zero-shot (full sets)"),
        lm_section(root, "siqa", "SIQA (top-up)"),
        lm_section(root, "paper", "Engram-paper few-shot protocol (300 items per task)"),
        lm_section(root, "fewshot", "5-shot"),
        ft_section(root),
        ft_seeds_section(root),
        cpt_section(root),
        ruler_section(root),
        counterfact_section(root),
        mech_section(root),
        interp_section(root),
    ]
    Path(a.out).write_text("\n\n".join(s for s in secs if s) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""Paired statistics over the ada results.

    python eval/stats_ada.py --root results --out STATS_ADA.md

A. lm-eval tasks: paired bootstrap (items aligned across runs) of the score difference
   between model pairs, and memory intact vs zeroed within a model. Holm correction per table.
B. Fine-tuning: mean +- sd over seeds per (task, model, regime, lr); Welch t-tests for the
   key comparisons (memory vs FFN subset vs full, and the learning-rate controls).
C. CounterFact: recall accuracy with McNemar's exact test (memory intact vs zeroed) and a
   paired bootstrap between models.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

MODELS = ["R1", "R2", "R3", "R4", "R5", "A", "S1"]
MEMORY = ["R1", "R2", "R3", "A", "S1"]
PAIRS = [("R1", "R4"), ("R2", "R4"), ("R3", "R4"), ("R5", "R4"), ("A", "R4"), ("S1", "R4"),
         ("R2", "R5"), ("R3", "R5"), ("R1", "R2"), ("S1", "R2"), ("A", "R2")]
TRACKS = ["zeroshot", "paper", "fewshot"]
NOT_ACCURACY = {"wikitext"}
B = 2000


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def items(samples: dict, task: str) -> np.ndarray | None:
    sub = samples.get(task)
    if not sub:
        return None
    vals = []
    for name in sorted(sub):
        vals += [float(x) for x in sub[name]]
    return np.asarray(vals, dtype=float)


def boot(a: np.ndarray, b: np.ndarray, rng) -> dict:
    d = a - b
    n = len(d)
    step = max(1, 2_000_000 // n)
    m = np.concatenate([d[rng.integers(0, n, size=(min(step, B - i), n))].mean(1) for i in range(0, B, step)])
    # The empirical tail floors p at 1/B, which Holm over ~200 rows turns into "nothing significant",
    # so p comes from the bootstrap SE (normal approximation; the mean is over hundreds of items).
    se = float(m.std(ddof=1))
    p = 1.0 if se == 0 else math.erfc(abs(float(d.mean())) / se / math.sqrt(2))
    return {"diff": float(d.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)),
            "p": p, "n": n}


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    k = len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n10": n10, "n01": n01, "p": p}


def welch(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    try:
        from scipy import stats

        return float(stats.ttest_ind(x, y, equal_var=False).pvalue)
    except Exception:
        return None


def fmt_p(p):
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def section_a(root: Path, rng, out: list[str], js: dict) -> None:
    out.append("## A. Benchmarks: paired bootstrap on per-item scores\n")
    out.append("Difference = first minus second (accuracy points x100). 95% CI from 2000 resamples; "
               "p two-sided from the bootstrap standard error, Holm-corrected within each table. "
               "`*` marks p_holm < 0.05. WikiText is a perplexity, not an accuracy, and is left out.\n")
    js["A"] = {}
    for track in TRACKS:
        rows = []
        for a, b in PAIRS:
            sa, sb = load(root / a / "none" / f"{track}_samples.json"), load(root / b / "none" / f"{track}_samples.json")
            if not sa or not sb:
                continue
            for task in sa:
                if task in NOT_ACCURACY:
                    continue
                x, y = items(sa, task), items(sb, task)
                if x is None or y is None or len(x) != len(y) or not len(x):
                    continue
                rows.append({"cmp": f"{a} vs {b}", "task": task, **boot(x, y, rng)})
        for m in MEMORY:
            sa, sz = load(root / m / "none" / f"{track}_samples.json"), load(root / m / "zero" / f"{track}_samples.json")
            if not sa or not sz:
                continue
            for task in sa:
                if task in NOT_ACCURACY:
                    continue
                x, y = items(sa, task), items(sz, task)
                if x is None or y is None or len(x) != len(y) or not len(x):
                    continue
                rows.append({"cmp": f"{m} intact vs zeroed", "task": task, **boot(x, y, rng)})
        if not rows:
            continue
        holm(rows)
        js["A"][track] = rows
        out.append(f"### {track}\n")
        out.append("| comparison | task | diff | 95% CI | p | p_holm |\n|---|---|---:|---:|---:|---:|")
        for r in rows:
            star = "*" if r["p_holm"] < 0.05 else ""
            out.append(f"| {r['cmp']} | {r['task']} | {100 * r['diff']:+.2f}{star} | [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] "
                       f"| {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} |")
        sig = {}
        for r in rows:
            s = sig.setdefault(r["cmp"], [0, 0, 0])
            s[2] += 1
            if r["p_holm"] < 0.05:
                s[0 if r["diff"] > 0 else 1] += 1
        out.append("\nSignificant tasks per comparison (better / worse / total):\n")
        out += [f"- {k}: {v[0]} / {v[1]} / {v[2]}" for k, v in sig.items()]
        out.append("")


def ft_key(path: Path, rec: dict) -> tuple:
    m = re.match(r"(.+)_lr([0-9.e-]+)_s(\d+)$", path.stem)
    if m:
        return m.group(1), float(m.group(2)), int(m.group(3))
    tag = {"ffn_layers_2-3-4-5-12-13-14-15": "ffn8"}.get(path.stem, path.stem)
    return tag, float(rec.get("lr")), int(rec.get("seed", 0))


def ft_score(rec: dict) -> float | None:
    sc = rec.get("score") or {}
    for k in ("acc", "f1", "micro_f1", "macro_f1"):
        if isinstance(sc.get(k), (int, float)):
            return float(sc[k])
    return None


def section_b(root: Path, out: list[str], js: dict) -> None:
    groups: dict = {}
    for p in sorted((root / "ft").glob("*/*/*.json")):
        rec = load(p)
        s = ft_score(rec) if rec else None
        if s is None:
            continue
        task, model = p.parts[-3], p.parts[-2]
        regime, lr, seed = ft_key(p, rec)
        groups.setdefault((task, model, regime, lr), {})[seed] = s
    js["B"] = {f"{t}|{m}|{r}|{lr:g}": v for (t, m, r, lr), v in groups.items()}
    out.append("## B. Fine-tuning over seeds\n")
    out.append("| task | model | regime | lr | n | mean | sd | seeds |\n|---|---|---|---:|---:|---:|---:|---|")
    for (t, m, r, lr), v in sorted(groups.items()):
        xs = list(v.values())
        sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
        out.append(f"| {t} | {m} | {r} | {lr:g} | {len(xs)} | {np.mean(xs):.3f} | {sd:.3f} | "
                   + ", ".join(f"{k}:{x:.3f}" for k, x in sorted(v.items())) + " |")
    cmps = [(("R2", "memory", 5e-4), ("R4", "ffn8", 5e-5), "memory vs dense FFN8, each at its default LR"),
            (("R2", "memory", 5e-4), ("R4", "ffn8", 5e-4), "memory vs dense FFN8, same LR 5e-4"),
            (("R2", "memory", 5e-5), ("R4", "ffn8", 5e-5), "memory vs dense FFN8, same LR 5e-5"),
            (("R2", "memory", 5e-4), ("R2", "memory", 5e-5), "memory: LR 5e-4 vs 5e-5"),
            (("R4", "ffn8", 5e-4), ("R4", "ffn8", 5e-5), "FFN8: LR 5e-4 vs 5e-5"),
            (("R2", "memory", 5e-4), ("R2", "ffn12", 5e-4), "R2 memory vs R2 FFN12 (similar size), LR 5e-4"),
            (("R2", "memory", 5e-5), ("R2", "ffn12", 5e-5), "R2 memory vs R2 FFN12, LR 5e-5"),
            (("R2", "memory", 5e-4), ("R2", "all", 5e-5), "R2 memory-only vs full fine-tuning"),
            (("R2", "all", 5e-5), ("R4", "all", 5e-5), "full fine-tuning R2 vs R4"),
            (("S1", "memory", 5e-4), ("R2", "memory", 5e-4), "memory-only S1 vs R2"),
            (("R1", "memory", 5e-4), ("R2", "memory", 5e-4), "memory-only R1 vs R2")]
    out.append("\n### Key comparisons (Welch t-test over seeds)\n")
    out.append("| task | comparison | A mean (n) | B mean (n) | diff | p |\n|---|---|---:|---:|---:|---:|")
    js["B_tests"] = []
    for t in sorted({k[0] for k in groups}):
        for (ma, ra, la), (mb, rb, lb), label in cmps:
            x = list(groups.get((t, ma, ra, la), {}).values())
            y = list(groups.get((t, mb, rb, lb), {}).values())
            if not x or not y:
                continue
            p = welch(x, y)
            js["B_tests"].append({"task": t, "cmp": label, "a": x, "b": y, "p": p})
            out.append(f"| {t} | {label} | {np.mean(x):.3f} ({len(x)}) | {np.mean(y):.3f} ({len(y)}) | "
                       f"{np.mean(x) - np.mean(y):+.3f} | {fmt_p(p)} |")
    out.append("")


def section_c(root: Path, rng, out: list[str], js: dict) -> None:
    cf = {m: load(root / m / "counterfact.json") for m in MODELS}
    cf = {m: v for m, v in cf.items() if v}
    if not cf:
        return
    out.append("## C. CounterFact recall\n")
    out.append("| model | acc intact | acc zeroed | McNemar p (intact vs zeroed) | items lost / gained |\n|---|---:|---:|---:|---:|")
    js["C"] = {}
    for m, v in cf.items():
        a = [x > 0 for x in v["per_item_none"]]
        row = {"acc_none": float(np.mean(a))}
        if "per_item_zero" in v:
            z = [x > 0 for x in v["per_item_zero"]]
            t = mcnemar(a, z)
            row.update(acc_zero=float(np.mean(z)), **t)
            out.append(f"| {m} | {row['acc_none']:.3f} | {row['acc_zero']:.3f} | {fmt_p(t['p'])} | {t['n10']} / {t['n01']} |")
        else:
            out.append(f"| {m} | {row['acc_none']:.3f} | — | — | — |")
        js["C"][m] = row
    out.append("\n| comparison | acc diff | 95% CI | p |\n|---|---:|---:|---:|")
    for a, b in PAIRS:
        if a in cf and b in cf and cf[a]["case_ids"] == cf[b]["case_ids"]:
            x = np.asarray([v > 0 for v in cf[a]["per_item_none"]], float)
            y = np.asarray([v > 0 for v in cf[b]["per_item_none"]], float)
            r = boot(x, y, rng)
            out.append(f"| {a} vs {b} | {100 * r['diff']:+.2f} | [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] | {fmt_p(r['p'])} |")
    out.append("")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results")
    p.add_argument("--out", default="STATS_ADA.md")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    root = Path(a.root).expanduser()
    rng = np.random.default_rng(a.seed)
    out = ["# Statistical tests (ada results)\n", __doc__.split("\n\n", 1)[1].strip() + "\n"]
    js: dict = {}
    section_a(root, rng, out, js)
    section_b(root, out, js)
    section_c(root, rng, out, js)
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

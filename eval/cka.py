"""Engram 6.1.2: layer-by-layer CKA between a memory model and a dense model.

    python eval/cka.py --a R2 --b R4 --out_dir results/eval

Reads ``<out_dir>/<name>/mech/fewnerd_states.pt`` (final token of each Few-NERD
entity, every layer) and writes ``<out_dir>/cka/<a>_vs_<b>.json``: the CKA matrix
S[dense layer, memory layer] from minibatch unbiased HSIC, and the soft alignment
index a_j (top-5 S-weighted centroid) with the shift a_j - j.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ffnmem.analysis_mech import soft_alignment, unbiased_hsic


def minibatch_cka(xs: list[torch.Tensor], ys: list[torch.Tensor], batch: int = 256, seed: int = 0) -> torch.Tensor:
    """S[i, j] = sum_b HSIC(X_i, Y_j) / sqrt(sum_b HSIC(X_i, X_i) * sum_b HSIC(Y_j, Y_j))."""
    n = xs[0].shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    la, lb = len(xs), len(ys)
    kl = torch.zeros(la, lb, dtype=torch.float64)
    kk = torch.zeros(la, dtype=torch.float64)
    ll = torch.zeros(lb, dtype=torch.float64)
    for s in range(0, n - batch + 1, batch):
        idx = perm[s: s + batch]
        gx = [(x[idx].double() @ x[idx].double().T) for x in xs]
        gy = [(y[idx].double() @ y[idx].double().T) for y in ys]
        for i, k in enumerate(gx):
            kk[i] += unbiased_hsic(k, k)
        for j, l in enumerate(gy):
            ll[j] += unbiased_hsic(l, l)
        for i, k in enumerate(gx):
            for j, l in enumerate(gy):
                kl[i, j] += unbiased_hsic(k, l)
    return (kl / (kk.clamp_min(1e-12).sqrt()[:, None] * ll.clamp_min(1e-12).sqrt()[None, :])).float()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="memory model name")
    p.add_argument("--b", required=True, help="dense reference name")
    p.add_argument("--out_dir", default="results/eval")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args(argv)
    root = Path(args.out_dir)
    A = torch.load(root / args.a / "mech" / "fewnerd_states.pt")
    B = torch.load(root / args.b / "mech" / "fewnerd_states.pt")
    key = lambda lab: (lab["sent"], lab["pos"])
    ia = {key(l): i for i, l in enumerate(A["labels"])}
    common = [(ia[key(l)], j) for j, l in enumerate(B["labels"]) if key(l) in ia]
    sa = A["states"][:, [c[0] for c in common]]
    sb = B["states"][:, [c[1] for c in common]]
    # rows = dense layers, columns = memory layers, as in Engram Fig. 5
    S = minibatch_cka(list(sb.float()), list(sa.float()), args.batch)
    a = soft_alignment(S, k=5)
    out = {"memory": args.a, "dense": args.b, "n": len(common), "S": S.tolist(), "a_j": a,
           "shift": [aj - j for j, aj in enumerate(a)], "diag": [float(S[i, i]) for i in range(min(S.shape))]}
    path = root / "cka" / f"{args.a}_vs_{args.b}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"n": len(common), "mean_shift": sum(out["shift"]) / len(out["shift"])}))


if __name__ == "__main__":
    main()

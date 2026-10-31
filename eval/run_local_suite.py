"""Phase-A end-to-end run on a tiny random model.

Writes the JSON the aggregator reads. Numbers are infrastructure checks, not
the 400M results: those need the ada checkpoints listed in ``ckpt_inventory.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ffnmem.analysis_mech import (
    bootstrap_ci, cka_matrix, code_utilisation, holm, linear_cka, logitlens_kl,
    mcnemar_p, propagation, retention, soft_alignment,
)
from ffnmem.config import load_config, resolve_model
from ffnmem.finetune import main as finetune_main
from ffnmem.hooks import Intervention
from ffnmem.inference.bench import run as bench_run
from ffnmem.inference.decode import decode_all
from ffnmem.inference.offload import offload_tables
from ffnmem.interp.entries import code_token_mi, direct_logit_rank, ngram_purity, route_codes
from ffnmem.interp.id_semantics import excess_retained, knn_ap, permutation_pvalue, signatures
from ffnmem.interp.patching import locality, patch_grid
from ffnmem.interp.probesets import write_builtin
from ffnmem.interp.token_attribution import bucket_mean, token_deltas
from ffnmem.interp.viz import write_heatmap
from ffnmem.model import Transformer
from ffnmem.synthetic import induction_accuracy, mqar_accuracy, next_token_accuracy, passkey_accuracy
from ffnmem.trainable import apply_trainable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "local_suite"


def _model(delay: int = 0) -> Transformer:
    cfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    if delay:
        cfg["model"]["memory_delay"] = delay
        cfg["model"] = resolve_model(cfg["model"])
    model = Transformer(cfg["model"])
    model.set_attention_backend("naive")
    model.eval()
    return model


def _ids(model, n=2, t=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, model.cfg["vocab"], (n, t), generator=g)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = _model()
    ids = _ids(model)
    # --- general ---
    general = {
        "induction": induction_accuracy(model, n=4),
        "passkey": passkey_accuracy(model, n=4),
        "mqar": mqar_accuracy(model, n=4),
        "next": next_token_accuracy(model, [(ids[:, :-1], ids[:, 1:])]),
        "note": "tiny random init; not a trained checkpoint",
    }
    (OUT / "general.json").write_text(json.dumps(general, indent=2), encoding="utf-8")

    # --- ablation ---
    full = general["next"]["loss"]
    with Intervention(model, memory="zero"):
        zero = next_token_accuracy(model, [(ids[:, :-1], ids[:, 1:])])
    captured = {}
    with Intervention(model, memory="capture", store=captured):
        model(ids[:, :-1])
    mean = {pos: t[0].mean(dim=(0, 1)).cpu() for pos, t in captured["memory"].items()}
    with Intervention(model, memory="mean", mean=mean):
        mean_score = next_token_accuracy(model, [(ids[:, :-1], ids[:, 1:])])
    ablation = {
        "zero": retention(full, zero["loss"]),
        "mean": retention(full, mean_score["loss"]),
        "logitlens": logitlens_kl(model, ids),
        "cka_self_diag": float(torch.diagonal(cka_matrix(model, model, ids)).mean()),
        "soft_align": soft_alignment(cka_matrix(model, model, ids)),
        "propagation": propagation(model, ids, {1}),
        "codes": code_utilisation(model, [(ids, ids[:, 1:])], torch.device("cpu")),
    }
    # JSON-friendly
    ablation["propagation"]["per_layer"] = ablation["propagation"]["per_layer"]
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")

    # --- interp ---
    codes = route_codes(model, ids)
    layer = next(iter(codes))
    deltas = token_deltas(model, ids, mean)
    labels = ["entity" if int(t) % 2 == 0 else "other" for t in ids[:, 1:].reshape(-1).tolist()]
    buckets = bucket_mean(deltas["delta_loss"], labels)
    mi = code_token_mi(codes[layer], ids, route=0)
    purity = ngram_purity(codes[layer], ids, route=0, order=2)
    direct = direct_logit_rank(model, ids, layer)
    clean = ids.clone()
    corrupt = ids.clone()
    corrupt[:, 3] = (corrupt[:, 3] + 7) % model.cfg["vocab"]
    grid = patch_grid(model, clean[:1], corrupt[:1], target=int(clean[0, -1]), site="memory")
    loc = locality(model, ids[:1], layer, token=3, horizon=4)
    sig_c, sig_h, sig_q = signatures(model, ids, layer)
    # One label per token, flattened.
    y = [int(t) % 3 for t in ids.reshape(-1).tolist()]
    flat_c = [row for row in sig_c.reshape(-1, sig_c.shape[-1])]
    flat_h = [row for row in sig_h.reshape(-1, sig_h.shape[-1])]
    ref, query = flat_c[:8], flat_c[8:]
    y_ref, y_q = y[:8], y[8:]
    if query:
        ap_id = knn_ap(query, ref, y_q, y_ref, lambda a, b: int((a != b).sum()), k=3)
        ap_h = knn_ap(
            [row for row in sig_h.reshape(-1, sig_h.shape[-1])][8:],
            [row for row in sig_h.reshape(-1, sig_h.shape[-1])][:8],
            y_q, y_ref, lambda a, b: float((a - b).pow(2).sum()), k=3,
        )
        ap_n = 0.0
        rho = excess_retained(ap_id, ap_h, ap_n)
    else:
        ap_id = ap_h = rho = None
    write_builtin(OUT / "probes")
    write_heatmap(["t" + str(i) for i in range(deltas["delta_loss"].shape[-1])],
                  deltas["delta_loss"][0].tolist(), OUT / "gating.html", "delta loss")
    interp = {
        "buckets": buckets, "mi": mi, "purity": purity, "direct": direct,
        "patch_denoising": {str(k): v for k, v in grid["denoising"].items()},
        "locality": loc, "id_ap": ap_id, "hidden_ap": ap_h, "rho": rho,
        "perm_p": permutation_pvalue(0.5, [0.1, 0.2, 0.4]),
    }
    (OUT / "interp.json").write_text(json.dumps(interp, indent=2), encoding="utf-8")

    # --- finetune regimes ---
    ft = {}
    for regime in ("all", "memory", "backbone", "tables"):
        r = finetune_main([
            "--task", "sst2", "--synthetic", "--trainable", regime,
            "--max_steps", "2", "--epochs", "1", "--batch", "4", "--device", "cpu",
            "--limit", "8",
        ])
        ft[regime] = {"score": r["score"], "trainable": r["trainable"]["trainable"], "steps": r["steps"]}
    (OUT / "finetune.json").write_text(json.dumps(ft, indent=2), encoding="utf-8")

    # --- offload + decode ---
    dec = decode_all(model, ids[:1, :6])
    full_logits = model(ids[:1, :6])
    if isinstance(full_logits, tuple):
        full_logits = full_logits[0]
    decode_err = float((dec - full_logits).abs().max().detach())
    with offload_tables(model, "sync"):
        off_logits = model(ids[:1, :6])
        if isinstance(off_logits, tuple):
            off_logits = off_logits[0]
    off_err = float((off_logits - full_logits).abs().max())
    bench = bench_run(model, batch=1, seq=4, iters=2)
    delayed = _model(delay=1)
    delayed_ok = delayed.memory_injection_map.get(1) == 2
    off = {"decode_max_abs": decode_err, "offload_max_abs": off_err, "bench": bench, "delay_injects_at": delayed.memory_injection_map, "delay_ok": delayed_ok}
    (OUT / "offload.json").write_text(json.dumps(off, indent=2), encoding="utf-8")

    # stats smoke
    stats = {
        "ci": bootstrap_ci([0.1, 0.2, 0.3, 0.4]),
        "mcnemar": mcnemar_p([1, 1, 0, 0], [1, 0, 1, 0]),
        "holm": holm([0.01, 0.04, 0.2]),
        "cka_identical": linear_cka(torch.randn(8, 4), torch.randn(8, 4)),
    }
    x = torch.randn(8, 4)
    stats["cka_self"] = linear_cka(x, x)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # trainable counts
    counts = {spec: apply_trainable(_model(), spec) for spec in ("all", "backbone", "memory", "tables", "ffn_layers:0,1")}
    (OUT / "trainable.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "decode_max_abs": decode_err, "offload_max_abs": off_err, "cka_self": stats["cka_self"]}))


if __name__ == "__main__":
    main()

"""General-capability and memory-ablation evaluation of one checkpoint.

    python -m ffnmem.run_lm_eval --ckpt final_ckpts/w1_mem_cut_near_step3815.pt --name R2 \
        --data_dir ~/morbius_data/climbmix --tracks val,synthetic,zeroshot,paper \
        --interventions none,zero,mean,zero@1,zero@11 --out_dir results/eval

Writes ``<out_dir>/<name>/<intervention>/<track>.json`` and skips files that
already exist, so a preempted job is simply resubmitted. lm-eval tracks also
write ``<track>_samples.json`` (per-item primary metric) for paired tests.
Interventions: ``none``, ``zero``, ``mean`` (per-layer mean memory output on
val), optionally restricted to blocks with ``@i,j``. ``mean`` vectors are
cached in ``<out_dir>/<name>/memory_means.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from eval.tasks import TRACKS
from ffnmem.hooks import Intervention, load_model, memory_means, parse_intervention


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def _ctx(model, spec, means):
    mode, layers = parse_intervention(spec)
    if mode is None:
        return nullcontext()
    return Intervention(model, memory=mode, layers=layers, mean=means)


def val_track(model, data_dir, seq_len, batches, batch, device, tok, spec, means) -> dict:
    from ffnmem.data.loader import ValWindows
    from ffnmem.synthetic import next_token_accuracy

    val = ValWindows(data_dir, seq_len, 57_000_000)
    used = list(val.batches(batch, batches))
    n_bytes = sum(len(tok.decode(y.reshape(-1).tolist()).encode("utf-8")) for _, y in used)
    n_tok = sum(y.numel() for _, y in used)
    with _ctx(model, spec, means):
        res = next_token_accuracy(model, used, bytes_per_token=n_bytes / n_tok)
    res.update(tokens=n_tok, bytes_per_token=n_bytes / n_tok, ppl=math.exp(min(res["loss"], 50)))
    return res


def synthetic_track(model, tok, spec, means, passkey_lengths=(256, 1024, 2000)) -> dict:
    from ffnmem.synthetic import induction_accuracy, mqar_accuracy, passkey_accuracy

    with _ctx(model, spec, means):
        return {
            "induction": induction_accuracy(model),
            "mqar": mqar_accuracy(model),
            "passkey": passkey_accuracy(model, tok, lengths=tuple(passkey_lengths)),
        }


def _primary(sample: dict, metric: str):
    for key in (metric, metric.split(",")[0], "acc", "exact_match", "f1", "em"):
        v = sample.get(key)
        if isinstance(v, (int, float, bool)):
            return float(v)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float, bool)):
            return float(v[0])
    return None


def lm_eval_track(model, seq_len, device, tokenizer, track, spec, means, limit, token_budget,
                  only=None) -> tuple[dict, dict]:
    import lm_eval
    from lm_eval.tasks import TaskManager

    from ffnmem.lm_eval_adapter import MorbiusLM

    lm = MorbiusLM(model=model, seq_len=seq_len, tokenizer=tokenizer, device=str(device),
                   intervention=spec, means=means, token_budget=token_budget)
    tm = TaskManager()
    results, samples = {}, {}
    for task, shots, metric, group in TRACKS[track]:
        if only is not None and task not in only:
            continue
        t0 = time.time()
        try:
            res = lm_eval.simple_evaluate(
                model=lm, tasks=[task], num_fewshot=shots, limit=limit, task_manager=tm,
                log_samples=True, bootstrap_iters=1000, random_seed=1234, numpy_random_seed=1234,
                torch_random_seed=1234, fewshot_random_seed=1234,
            )
        except Exception as e:  # a missing dataset or task must not sink the rest of the track
            results[task] = {"error": f"{type(e).__name__}: {e}"[:500], "shots": shots}
            print(f"[{track}] {task} failed: {e}", flush=True)
            continue
        entry = {"shots": shots, "metric": metric, "group": group, "seconds": round(time.time() - t0, 1),
                 "results": res["results"], "n_samples": {k: len(v) for k, v in res.get("samples", {}).items()}}
        top = res["results"].get(task) or {}
        src = top if top.get(f"{metric},none") is not None or not res.get("groups", {}).get(task) else res["groups"][task]
        key = f"{metric},none" if f"{metric},none" in src else next(
            (k for k in src if k.startswith(f"{metric},") and "_stderr" not in k), f"{metric},none")
        entry["score"] = src.get(key)
        entry["stderr"] = src.get(key.replace(",", "_stderr,", 1))
        results[task] = entry
        samples[task] = {
            sub: [_primary(s, metric) for s in sorted(v, key=lambda s: s["doc_id"])]
            for sub, v in res.get("samples", {}).items()
        }
        print(f"[{track}] {task} shots={shots} {metric}={entry['score']} ({entry['seconds']}s)", flush=True)
    return results, samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data_dir", default=os.path.expanduser("~/morbius_data/climbmix"))
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--tracks", default="val,synthetic,zeroshot")
    p.add_argument("--interventions", default="none")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="items per lm-eval task")
    p.add_argument("--val_batches", type=int, default=64)
    p.add_argument("--val_batch", type=int, default=8)
    p.add_argument("--mean_batches", type=int, default=16)
    p.add_argument("--token_budget", type=int, default=16384)
    p.add_argument("--out_dir", default="results/eval")
    p.add_argument("--redo", type=lambda s: set(s.split(",")), default=set(),
                   help="lm-eval tasks to rerun even if a result exists")
    p.add_argument("--passkey_lengths", type=lambda s: [int(x) for x in s.split(",")], default=[256, 1024, 2000])
    args = p.parse_args(argv)

    from tokenizers import Tokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = Tokenizer.from_file(args.tokenizer)
    model, state = load_model(args.ckpt, device)
    cfg = state["config"]
    seq_len = int(cfg["model"]["seq_len"])
    del state
    has_memory = any(b.memory is not None for b in model.blocks)
    root = Path(args.out_dir) / args.name
    meta = {"ckpt": args.ckpt, "config": cfg.get("name"), "memory_layers": [i for i, b in enumerate(model.blocks) if b.memory is not None],
            "git_rev": Path("GIT_REV").read_text().strip() if Path("GIT_REV").exists() else None,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    _write(root / "meta.json", meta)

    means = None
    specs = [s for s in args.interventions.split(",") if s]
    if any(parse_intervention(s)[0] == "mean" for s in specs) and has_memory:
        mpath = root / "memory_means.pt"
        if mpath.exists():
            means = torch.load(mpath)
        else:
            from ffnmem.data.loader import ValWindows

            # Means come from the far end of val so they never overlap the scored windows.
            val = ValWindows(args.data_dir, seq_len, 57_000_000)
            val.ids = val.ids[-args.mean_batches * args.val_batch:]
            means = memory_means(model, (x for x, _ in val.batches(args.val_batch)), device)
            mpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(means, mpath)

    for spec in specs:
        if spec != "none" and not has_memory:
            continue
        mode, layers = parse_intervention(spec)
        if layers and not layers <= set(meta["memory_layers"]):
            print(f"skip {spec}: not a memory layer of {args.name}", flush=True)
            continue
        for track in [t for t in args.tracks.split(",") if t]:
            out = root / spec / f"{track}.json"
            prev = None
            if out.exists():
                prev = json.loads(out.read_text())
                failed = []
                if track in TRACKS:
                    names = [t for t, *_ in TRACKS[track]]
                    prev["result"] = {t: v for t, v in prev["result"].items() if t in names}
                    failed = [t for t in names if "error" in prev["result"].get(t, {"error": "missing"})
                              or t in args.redo]
                if not failed:
                    print(f"exists {out}", flush=True)
                    continue
                print(f"retrying failed tasks of {out}: {failed}", flush=True)
            t0 = time.time()
            if track == "val":
                res = val_track(model, args.data_dir, seq_len, args.val_batches, args.val_batch, device, tok, spec, means)
            elif track == "synthetic":
                res = synthetic_track(model, tok, spec, means, args.passkey_lengths)
            elif track in TRACKS:
                only = set(failed) if prev else None
                res, samples = lm_eval_track(model, seq_len, device, args.tokenizer, track, spec, means,
                                             args.limit, args.token_budget, only=only)
                spath = root / spec / f"{track}_samples.json"
                if prev:
                    res = {**prev["result"], **res}
                    if spath.exists():
                        samples = {**json.loads(spath.read_text()), **samples}
                _write(spath, samples)
            else:
                raise SystemExit(f"unknown track {track}")
            res = {"name": args.name, "intervention": spec, "track": track, "limit": args.limit,
                   "seconds": round(time.time() - t0, 1), "result": res}
            _write(out, res)
            print(json.dumps({"name": args.name, "spec": spec, "track": track, "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

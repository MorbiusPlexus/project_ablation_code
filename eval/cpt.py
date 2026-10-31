"""Domain continued pre-training of one checkpoint with a trainable subset, then evaluation.

    python eval/cpt.py --src <ckpt> --name CPT_R2_memory --trainable memory \
        --data_dir ~/morbius_data/pubmed --run_root /ssd_scratch/$USER/cpt --results results

The model config is taken from the source checkpoint so the architecture always matches.
After training the optimizer state is dropped (``final.pt`` holds model + config only) and
the model is scored on ClimbMix val + zero-shot (forgetting), the biomedical ``domain``
track, and PubMed val loss (``<results>/cpt_pubmedval``). Every stage is skipped when its
output exists, so the job can be resubmitted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--trainable", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--run_root", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--global_batch", type=int, default=32)
    p.add_argument("--micro_batch", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--general_data", default=os.path.expanduser("~anangappara/morbius_data/climbmix"),
                   help="ClimbMix shards for the forgetting check")
    a = p.parse_args()

    run_dir = Path(a.run_root) / a.name
    final = run_dir / "final.pt"
    res_dir = Path(a.results) / "cpt" / a.name
    res_dir.mkdir(parents=True, exist_ok=True)
    if not final.exists():
        cfg = torch.load(a.src, map_location="cpu", weights_only=False, mmap=True)["config"]
        cfg.pop("_source", None)
        cfg["name"] = a.name
        cfg["train"].update({
            "data_dir": a.data_dir, "val_tokens": 2_000_000, "global_batch": a.global_batch,
            "micro_batch": a.micro_batch, "tokens_total": a.tokens, "lr": a.lr, "warmup_frac": 0.05,
            "warmup_min_steps": 10, "ckpt_every": 100000, "keep_last": 1, "keep_every": 100000,
            "eval_every": 50, "eval_max_batches": 64, "trainable": a.trainable, "zero1": False,
        })
        cfg["train"].pop("seq_len", None)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "cpt_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "-m", "ffnmem.train", "--config", str(cfg_path), "--init_from", a.src,
               "--new_schedule", "--data_restart", "--out_dir", str(run_dir), "--device", a.device]
        if a.max_steps:
            cmd += ["--max_steps", str(a.max_steps)]
        run(cmd)
        ckpts = sorted(run_dir.rglob("step_*.pt"))
        if not ckpts:
            raise SystemExit(f"no checkpoint written under {run_dir}")
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        torch.save({"model": state["model"], "config": state["config"], "step": state["step"],
                    "params": state.get("params")}, final)
        del state
        for c in ckpts:
            c.unlink()
    for log in run_dir.rglob("*.jsonl"):
        shutil.copy(log, res_dir / log.name)
    (res_dir / "meta.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    if a.skip_eval:
        return
    py = [sys.executable, "-m", "ffnmem.run_lm_eval", "--ckpt", str(final), "--name", a.name]
    run(py + ["--out_dir", str(Path(a.results) / "cpt"), "--data_dir", a.general_data, "--tracks", "val,zeroshot,domain",
              "--interventions", "none"])
    run(py + ["--out_dir", str(Path(a.results) / "cpt_pubmedval"), "--data_dir", a.data_dir, "--tracks", "val",
              "--interventions", "none", "--val_batches", "100"])


if __name__ == "__main__":
    main()

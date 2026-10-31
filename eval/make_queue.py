"""Write the GPU work queue for ``slurm/queue_worker.sh`` in priority order.

    python eval/make_queue.py --phase main,ideas,ft > ~/morbius_eval_runs/queue.txt

Phases: ``main`` (non-finetuned main-branch checkpoints), ``ideas`` (S1 from the
ideas branch), ``ft`` (task fine-tuning matrix). Every command is idempotent: it
skips outputs that already exist, so the queue can be regenerated and rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.tasks import FT_TASKS, MODELS, SHORT

PRE = ". ~/morbius_eval_runs/env.sh && "
INTERP_MODELS = {"w1_mem_cut_near", "w2_mem_super1", "w1_dense_full"}
OUT = "~/morbius_eval_runs/results"


def ckpt(cfg: str, root: str) -> str:
    return f"{root}/{cfg}_step3815.pt"


def eval_lines(cfg: str, root: str, paper_limit: int) -> list[str]:
    name = SHORT[cfg]
    kind = MODELS[cfg]["kind"]
    layers = MODELS[cfg]["layers"]
    base = f"{PRE}python -m ffnmem.run_lm_eval --ckpt {ckpt(cfg, root)} --name {name} --out_dir {OUT}"
    mech = f"{PRE}python eval/run_mech.py --ckpt {ckpt(cfg, root)} --name {name} --out_dir {OUT}"
    interp = (f"{PRE}while [ ! -f {OUT}/{name}/mech/fewnerd.json ]; do sleep 60; done && "
              f"python eval/run_interp.py")
    interp += f" --ckpt {ckpt(cfg, root)} --name {name} --out_dir {OUT}"
    lines = [f"{base} --tracks val,synthetic,zeroshot --interventions none"]
    if kind == "memory":
        lines.append(f"{base} --tracks val,synthetic,zeroshot --interventions zero,mean")
        if len(layers) > 1:
            per = ",".join(f"{m}@{l}" for m in ("zero", "mean") for l in layers)
            lines.append(f"{base} --tracks val,synthetic --interventions {per}")
        lines.append(f"{mech} --parts logitlens,propagation,codes,fewnerd")
        if cfg in INTERP_MODELS:
            lines.append(f"{interp} --parts attribution,viz,entries,locality,idsem,patching")
    else:
        lines.append(f"{mech} --parts logitlens,fewnerd")
        if cfg in INTERP_MODELS:
            lines.append(f"{interp} --parts patching")
    lines.append(f"{base} --tracks paper,fewshot --interventions none --limit {paper_limit}")
    if kind == "memory":
        lines.append(f"{base} --tracks paper --interventions zero --limit {paper_limit}")
    return lines


def ft_lines(cfg: str, root: str, regimes: list[str]) -> list[str]:
    name = SHORT[cfg]
    out = []
    for regime in regimes:
        tag = regime.replace(":", "_").replace(",", "-")
        for task in FT_TASKS:
            extra = {"sst2": "--batch 32 --max_len 128", "boolq": "--batch 16 --max_len 384",
                     "ner": "--batch 32 --max_len 160"}[task]
            res = f"{OUT}/ft/{task}/{name}/{tag}.json"
            out.append(f"{PRE}test -f {res} || python -m ffnmem.finetune --task {task} --ckpt {ckpt(cfg, root)} "
                       f"--trainable {regime} --device cuda {extra} --out {res}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", default="main,ideas,ft")
    p.add_argument("--root", default="/home2/sparsh.bhartia/morbius/final_ckpts")
    p.add_argument("--paper_limit", type=int, default=300)
    p.add_argument("--models", default="w1_mem_cut_near,w1_dense_full,w1_dense_cut_near,w2_mem_rank1",
                   help="main-branch checkpoints readable from this account")
    args = p.parse_args(argv)
    phases = args.phase.split(",")
    main_models = [m for m in args.models.split(",") if m]
    lines: list[str] = []
    if "main" in phases:
        # Breadth first: the cheap core of every model before any model's paper track.
        per = {m: eval_lines(m, args.root, args.paper_limit) for m in main_models}
        for k in range(max(len(v) for v in per.values())):
            for m in main_models:
                if k < len(per[m]):
                    lines.append(per[m][k])
    if "ideas" in phases:
        lines += eval_lines("w2_mem_super1", args.root, args.paper_limit)
        wait = f"{PRE}for f in R2 R4 S1; do while [ ! -f {OUT}/$f/mech/fewnerd.json ]; do sleep 60; done; done && "
        lines.append(wait + f"python eval/cka.py --a R2 --b R4 --out_dir {OUT} && python eval/cka.py --a S1 --b R4 --out_dir {OUT}")
    if "ft" in phases:
        mem_regimes = ["none", "memory", "backbone", "all"]
        lines += ft_lines("w1_mem_cut_near", args.root, mem_regimes)
        lines += ft_lines("w2_mem_super1", args.root, mem_regimes)
        lines += ft_lines("w1_dense_full", args.root, ["none", "all", "ffn_layers:2,3,4,5,12,13,14,15"])
        lines += ft_lines("w1_dense_cut_near", args.root, ["none", "all"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Queue lines for the extra fine-tuning seeds and the learning-rate / parameter-count controls.

    python eval/make_ft_extra.py --env <env.sh> --results <results root> >> queue.txt

Main regimes get seeds 1 and 2 at the default LR (seed 0 is the original ``<regime>.json``).
Controls (seeds 0-2) separate the LR from the parameter subset:
  R2 memory @5e-5        memory at the backbone LR
  R2 FFN12 @5e-5 / 5e-4  12 non-cut FFN layers of R2 (about the memory's parameter count) at both LRs
  R4 FFN8 @5e-4          the dense model's 8 cut-position FFNs at the memory LR
Outputs: <results>/ft/<task>/<model>/<tag>_lr<lr>_s<seed>.json, skipped when present.
"""
from __future__ import annotations

import argparse

CKPT = {
    "R1": "/home2/anangappara/morbius_ckpts_aditya/w1_mem_full_step3815.pt",
    "R2": "/home2/sparsh.bhartia/morbius/final_ckpts/w1_mem_cut_near_step3815.pt",
    "R4": "/home2/sparsh.bhartia/morbius/final_ckpts/w1_dense_full_step3815.pt",
    "S1": "/home2/sparsh.bhartia/morbius/final_ckpts/w2_mem_super1_step3815.pt",
}
TASKS = {"sst2": (32, 128, 1), "boolq": (16, 384, 2), "ner": (32, 160, 1)}
FFN8 = "ffn_layers:2,3,4,5,12,13,14,15"
FFN12 = "ffn_layers:0,1,6,7,8,9,10,11,16,17,18,19"
TAG = {FFN8: "ffn8", FFN12: "ffn12"}


def default_lr(regime: str) -> float:
    return 5e-4 if regime in ("memory", "tables") else 5e-5


def jobs():
    for model, regimes in [("R1", ["memory", "all"]), ("R2", ["memory", "all"]), ("S1", ["memory", "all"]),
                           ("R4", [FFN8, "all"])]:
        for regime in regimes:
            for seed in (1, 2):
                yield model, regime, default_lr(regime), seed
    for model, regime, lr in [("R2", "memory", 5e-5), ("R2", FFN12, 5e-5), ("R2", FFN12, 5e-4), ("R4", FFN8, 5e-4)]:
        for seed in (0, 1, 2):
            yield model, regime, lr, seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--results", required=True)
    a = p.parse_args()
    for task, (batch, max_len, accum) in TASKS.items():
        for model, regime, lr, seed in jobs():
            tag = TAG.get(regime, regime)
            out = f"{a.results}/ft/{task}/{model}/{tag}_lr{lr:g}_s{seed}.json"
            print(f". {a.env} && test -f {out} || python -m ffnmem.finetune --task {task} --ckpt {CKPT[model]} "
                  f"--trainable {regime} --lr {lr:g} --seed {seed} --device cuda --batch {batch} --accum {accum} "
                  f"--max_len {max_len} --out {out}")


if __name__ == "__main__":
    main()

"""Queue lines for the PubMed continued-pretraining study and its base-model reference evals.

    python eval/make_cpt_jobs.py --env <env.sh> >> queue.txt

Lines expand $CK, $PUB, $R, $RR from the sourced env file.
"""
from __future__ import annotations

import argparse

FFN8 = "ffn_layers:2,3,4,5,12,13,14,15"
FFN12 = "ffn_layers:0,1,6,7,8,9,10,11,16,17,18,19"
SRC = {"R2": "$CK/w1_mem_cut_near_step3815.pt", "R4": "$CK/w1_dense_full_step3815.pt",
       "S1": "$CK/w2_mem_super1_step3815.pt"}
RUNS = [  # name, source, trainable, lr
    ("CPT_R2_memory", "R2", "memory", 1e-4),
    ("CPT_R2_memory_lr5e-4", "R2", "memory", 5e-4),
    ("CPT_R2_all", "R2", "all", 1e-4),
    ("CPT_R2_ffn12", "R2", FFN12, 1e-4),
    ("CPT_R4_all", "R4", "all", 1e-4),
    ("CPT_R4_ffn8", "R4", FFN8, 1e-4),
    ("CPT_S1_memory", "S1", "memory", 1e-4),
    ("CPT_S1_all", "S1", "all", 1e-4),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--part", choices=["cpt", "base"], required=True)
    a = p.parse_args()
    if a.part == "cpt":
        for name, src, trainable, lr in RUNS:
            print(f". {a.env} && python eval/cpt.py --src {SRC[src]} --name {name} --trainable {trainable} "
                  f"--lr {lr:g} --data_dir $PUB --run_root $R/cpt --results $RR/results")
    else:
        for name, src in SRC.items():
            ivs = "none" if name == "R4" else "none,zero"
            print(f". {a.env} && python -m ffnmem.run_lm_eval --ckpt {src} --name {name} --out_dir $RR/results/cpt_base "
                  f"--tracks domain --interventions {ivs} && python -m ffnmem.run_lm_eval --ckpt {src} --name {name} "
                  f"--out_dir $RR/results/cpt_pubmedval --data_dir $PUB --tracks val --interventions {ivs} --val_batches 100")


if __name__ == "__main__":
    main()

#!/bin/bash
# Extract R1/R3 final weights (model + config, no optimizer) from Aditya's gnode078 scratch to home.
#SBATCH -A research -p u22 -w gnode078 -c 2 --mem=16G -t 01:00:00 -J save_r1r3
#SBATCH -o /home2/anangappara/morbius_eval_runs/save_r1r3_%j.out
set -euo pipefail
. ~/morbius_evalenv/bin/activate
DEST=$HOME/morbius_ckpts_aditya
mkdir -p "$DEST"
for CFG in w1_mem_full w1_mem_cut_far; do
  SRC=/ssd_scratch/aditya.shankar/morbius/runs/$CFG/checkpoints/step_0003815.pt
  python - "$SRC" "$DEST" "$CFG" <<'PYEOF'
import os, sys, torch
src, dest, cfg = sys.argv[1:4]
s = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
out = os.path.join(dest, f"{cfg}_step{s['step']:04d}.pt")
tmp = out + ".tmp"
torch.save({"model": s["model"], "config": s["config"], "step": s["step"], "params": s.get("params"),
            "git": s.get("git"), "source": src}, tmp)
os.replace(tmp, out)
print(f"saved {out} ({os.path.getsize(out) / 2**30:.2f} GiB) config={s['config'].get('name')} step={s['step']}")
PYEOF
done

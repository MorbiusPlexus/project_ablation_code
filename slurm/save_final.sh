#!/bin/bash
# Save a finished run's final weights (fp32 model + config + step, no optimizer state)
# from this node's ssd to home NFS:  ~/morbius/final_ckpts/<cfg>_step<N>.pt  (~1.6 GB)
#   bash slurm/save_final.sh <config>         (on the run's node, inside any job of $USER there)
# Submitted as a small CPU job pinned to the run's node, e.g.
#   sbatch -w <node> --dependency=afterok:<train job> -c 2 --mem=8G -t 02:00:00 -J save_<cfg> \
#          -o ~/morbius/joblogs/%x-%j.out ~/morbius/slurm/save_final.sh <cfg>
set -euo pipefail
CFG=$1
B=/ssd_scratch/$USER/morbius
PY=$(ls -d $B/python/cpython-3.12*/bin/python3.12 2>/dev/null | head -1 || true)
[ -n "$PY" ] || PY=$(ls -d /ssd_scratch/aditya.shankar/morbius/python/cpython-3.12*/bin/python3.12 | head -1)
SRC=$(ls $B/runs/$CFG/checkpoints/step_*.pt | sort | tail -1)
DEST=$HOME/morbius/final_ckpts
mkdir -p "$DEST"
nice -n 10 "$PY" - "$SRC" "$DEST" "$CFG" <<'PYEOF'
import os, sys, torch
src, dest, cfg = sys.argv[1:4]
s = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
out = os.path.join(dest, f"{cfg}_step{s['step']:04d}.pt")
tmp = out + ".tmp"
torch.save({"model": s["model"], "config": s["config"], "step": s["step"], "params": s.get("params"),
            "git": s.get("git"), "source": src}, tmp)
os.replace(tmp, out)
print(f"saved {out} ({os.path.getsize(out) / 2**30:.2f} GiB) from {src}")
PYEOF

"""Move a borrowed account's queue from /home2/<user>/mrun to /ssd_scratch/<user>/mrun.

Run on the compute node, under no lock (it takes the queue lock itself):
    python3 slurm/migrate_queue.py --user aditya.shankar --workers 4 [--take 6]

Pending lines are rewritten to write under scratch and moved to the scratch queue. The home
queue is left with one hand-off line per worker: when a worker finishes its current job it
picks one up and becomes a worker on the scratch queue for the same GPU. ``--take T`` prints
the tier-T lines instead of queueing them (to move them to another node).
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from reorder_queue import tier  # noqa: E402

WORKER = "/home2/anangappara/morbius_eval/slurm/queue_worker_wait.sh"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--take", type=int, default=None)
    a = p.parse_args()
    home, scratch = Path(f"/home2/{a.user}/mrun"), Path(f"/ssd_scratch/{a.user}/mrun")
    (scratch / "results").mkdir(parents=True, exist_ok=True)
    (scratch / "logs").mkdir(parents=True, exist_ok=True)
    env = scratch / "env.sh"
    if env.exists():
        env.write_text(re.sub(r"RR=\S+", f"RR={scratch}", env.read_text()))
    hq, sq = home / "queue.txt", scratch / "queue.txt"
    with open(str(hq) + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = [l for l in hq.read_text().splitlines() if l.strip()] if hq.exists() else []
        moved = [l.replace(f"{home}/results", f"{scratch}/results") for l in lines if not l.startswith("bash " + WORKER)]
        taken = [l for l in moved if a.take is not None and tier(l) == a.take]
        keep = [l for l in moved if l not in taken]
        with open(str(sq) + ".lock", "a") as lk2:
            fcntl.flock(lk2, fcntl.LOCK_EX)
            with open(sq, "a") as f:
                f.writelines(l + "\n" for l in keep)
        # Distinct lines: a worker removes the line it takes with an exact-match grep.
        handoff = [f"bash {WORKER} {sq} $CUDA_VISIBLE_DEVICES > {scratch}/worker_g$CUDA_VISIBLE_DEVICES.out 2>&1 # handoff {i}"
                   for i in range(a.workers)]
        tmp = Path(str(hq) + ".tmp")
        tmp.write_text("\n".join(handoff) + "\n")
        os.replace(tmp, hq)
    print("\n".join(taken))
    print(f"moved {len(keep)} lines to {sq}; took {len(taken)}", file=sys.stderr)


if __name__ == "__main__":
    main()

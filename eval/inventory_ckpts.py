"""Record which final checkpoints are reachable from this machine."""

from __future__ import annotations

import json
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    "w1_mem_full_step3815.pt",
    "w1_mem_cut_near_step3815.pt",
    "w1_mem_cut_far_step3815.pt",
    "w1_dense_full_step3815.pt",
    "w1_dense_cut_near_step3815.pt",
    "w2_mem_rank1_step3815.pt",
    "w2_mem_super1_step3815.pt",
]
REMOTE = {
    "aditya": "/home/aditya.shankar/morbius/final_ckpts",
    "sparsh": "/home2/sparsh.bhartia/morbius/final_ckpts",
}


def main():
    local = [str(p) for p in ROOT.rglob("*step3815*.pt")]
    ada = False
    try:
        socket.getaddrinfo("ada.iiit.ac.in", 22)
        ada = True
    except socket.gaierror:
        ada = False
    out = {
        "host": socket.gethostname(),
        "ada_resolves": ada,
        "local_final_ckpts": local,
        "remote_expected": REMOTE,
        "files": CANDIDATES,
        "note": "Phase B reads these from ada. export_local.sh makes a 4050-sized fp16 copy.",
    }
    path = ROOT / "results" / "ckpt_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

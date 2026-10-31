"""Stable-sort a queue file by priority, in place. Run under the queue's flock.

    flock queue.txt.lock python3 eval/reorder_queue.py queue.txt [--dry]

Lower tier runs first; no line is added or removed.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

CPT_MAIN = r"--name CPT_(R2_memory|R2_all|R2_ffn12|R4_all) "
FT_MAIN_SEEDS = r"/ft/\w+/(R2/(memory|all)|R4/(ffn8|all))_lr[0-9.e-]+_s[12]\.json"
FT_LR_CTRL = r"/ft/\w+/(R4/ffn8_lr0\.0005|R2/memory_lr5e-05)_s[012]\.json"
TIERS = [
    (0, r"eval/counterfact\.py"),
    (1, r"--tracks val,synthetic --interventions zero@|--tracks paper,fewshot "),
    (2, CPT_MAIN),
    (3, r"mini_ruler\.py .*--name (R2|R4|A|S1) "),
    (4, r"cpt_base --tracks domain.*|--name (R2|R4) --out_dir \S*cpt_base"),
    (5, r"/ft/\w+/(R1|R3)/(none|memory|backbone|all)\.json"),
    (6, FT_MAIN_SEEDS + "|" + FT_LR_CTRL),
    (7, r"eval/cka\.py"),
]


def tier(line: str) -> int:
    for t, pat in TIERS:
        if re.search(pat, line):
            if t == 4 and "--name S1" in line:
                return 9
            return t
    return 9


def main():
    path = sys.argv[1]
    lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    if "--take" in sys.argv:
        # Remove the lines of one tier from the queue and print them (for moving to another node).
        t = int(sys.argv[sys.argv.index("--take") + 1])
        keep = [l for l in lines if tier(l) != t]
        print("\n".join(l for l in lines if tier(l) == t))
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
        import os

        os.replace(path + ".tmp", path)
        return
    ordered = sorted(lines, key=tier)
    print(dict(sorted(Counter(tier(l) for l in lines).items())))
    if "--dry" not in sys.argv:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write("\n".join(ordered) + ("\n" if ordered else ""))
        import os

        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()

"""Probe sets T1-T9. Each row is context, a target token span, and a category.

Built-in rows are small and have no download. ``--hf`` adds Few-NERD / CounterFact
when those datasets are already cached.
"""

from __future__ import annotations

import json
from pathlib import Path

BUILTIN = {
    "T1": [
        {"context": "the capital of france is", "target": " paris", "category": "entity"},
        {"context": "alexander the great ruled", "target": " macedon", "category": "entity"},
        {"context": "the milky way galaxy contains", "target": " stars", "category": "entity"},
    ],
    "T2": [
        {"context": "by the way i", "target": " meant", "category": "idiom"},
        {"context": "it was a piece of cake to", "target": " finish", "category": "idiom"},
        {"context": "once in a blue moon we", "target": " see", "category": "idiom"},
    ],
    "T3": [
        {"context": "the word antidisestablishment", "target": "arianism", "category": "subword"},
        {"context": "hyperparameter", "target": "ization", "category": "subword"},
    ],
    "T4": [
        {"context": "The capital of France is", "target": " Paris", "category": "fact"},
        {"context": "The Eiffel Tower is located in", "target": " Paris", "category": "fact"},
    ],
    "T5": [
        {"context": "the keys to the cabinet", "target": " are", "category": "agreement"},
        {"context": "the key to the cabinets", "target": " is", "category": "agreement"},
    ],
    "T6": [
        {"context": "7 4 9 2 7 4 9", "target": " 2", "category": "induction"},
    ],
    "T7": [
        {"context": "2 + 2 =", "target": " 4", "category": "number"},
        {"context": "10 - 3 =", "target": " 7", "category": "number"},
    ],
    "T8": [
        {"context": "import numpy as", "target": " np", "category": "code"},
        {"context": "for i in range(", "target": "10", "category": "code"},
    ],
    "T9": [
        {"context": "messenger rna is translated into", "target": " protein", "category": "biomed"},
        {"context": "the mitral valve sits between the left", "target": " atrium", "category": "biomed"},
    ],
}


def write_builtin(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in BUILTIN.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        paths.append(path)
    return paths


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/interp/probes")
    args = p.parse_args()
    paths = write_builtin(args.out)
    print(json.dumps({"wrote": [str(x) for x in paths]}))


if __name__ == "__main__":
    main()

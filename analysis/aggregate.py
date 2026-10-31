"""Collect suite JSON into EVAL_REPORT.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    suite = ROOT / "results" / "local_suite"
    inv = _load(ROOT / "results" / "ckpt_inventory.json")
    general = _load(suite / "general.json")
    ablation = _load(suite / "ablation.json")
    interp = _load(suite / "interp.json")
    ft = _load(suite / "finetune.json")
    off = _load(suite / "offload.json")
    gpu = _load(suite / "gpu4050.json")
    perf = _load(suite / "perf_model.json")
    stats = _load(suite / "stats.json")
    lines = [
        "# Evaluation report",
        "",
        "This file is produced by `analysis/aggregate.py`.",
        "",
        "## What was run here",
        "",
        "The local RTX 4050 path and the CPU suite exercise every evaluation module on a tiny random model.",
        "The seven trained checkpoints live on ada (`~/morbius/final_ckpts`). This machine cannot reach ada",
        "(see `results/ckpt_inventory.json`), so the tables below are pipeline checks, not paper numbers.",
        "Phase B is `eval/submit.py` against `eval/manifest.yaml` once a GPU allocation exists.",
        "",
        "## Protocol",
        "",
        "Engram Table 1 shots, the factual-vs-reading suppression split, LogitLens, CKA with the soft",
        "alignment index, Lngram code utilisation, address-stability metrics, ID-only KNN and the",
        "host-RAM offload model are implemented. Tasks a 400M model scores at chance on are marked",
        "`floor` in `eval/tasks.py` and are not dropped silently.",
        "",
        "## Local suite",
        "",
    ]
    for name, obj in (("general", general), ("ablation", ablation), ("interp", interp),
                      ("finetune", ft), ("offload", off), ("gpu4050", gpu), ("perf_model", perf),
                      ("stats", stats), ("inventory", inv)):
        lines.append(f"### {name}")
        lines.append("")
        if obj is None:
            lines.append("not run")
        else:
            text = json.dumps(obj, indent=2)
            if len(text) > 4000:
                text = text[:4000] + "\n..."
            lines.append("```json")
            lines.append(text)
            lines.append("```")
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Single seed. Differences inside a paired bootstrap interval are ties.",
        "- Zero ablation is off-distribution. Mean ablation is reported beside it.",
        "- Wave-1 run 1 (2080 Ti) and run 4 (3090) still carry the GPU-type confound.",
        "- Fine-tuning hyperparameters are fixed. No learning-rate sweep.",
        "- `w2_mem_delayed4` stopped at step 90 and is not a result row.",
        "- Local numbers use a 4-layer random model. Do not compare them to the val losses in `state_and_results.md`.",
        "",
    ]
    path = ROOT / "EVAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

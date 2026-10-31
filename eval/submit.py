"""Print (or submit) every Phase B sbatch line.

    python eval/submit.py --dry-run
    python eval/submit.py --dry-run --only 'w1_mem_full*' --account aditya
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eval.tasks import FT_REGIMES_DENSE, FT_REGIMES_MEMORY, FT_TASKS, MODELS, PAPER_TASKS

ROOT = Path(__file__).resolve().parents[1]


def _jobs() -> list[dict]:
    jobs = []
    for account in ("aditya", "sparsh"):
        jobs.append({
            "id": f"preflight-{account}", "account": account, "gpus": 1,
            "minutes": 20, "dep": None,
            "cmd": "bash slurm/preflight.sh",
        })
    runnable = [n for n, s, m, g, st in PAPER_TASKS if st == "run"]
    task_arg = ",".join(runnable)
    for name, meta in MODELS.items():
        ckpt = f"$CKPT/{name}_step3815.pt"
        jobs.append({
            "id": f"lmeval-{name}", "account": meta["account"], "gpus": 1,
            "minutes": 180, "dep": f"preflight-{meta['account']}",
            "cmd": f"python -m ffnmem.run_lm_eval --ckpt {ckpt} --tasks {task_arg} --device cuda --out results/eval/{name}/lm_eval.json",
        })
        if meta["kind"] == "memory":
            for mode, which in (("zero", "all"), ("mean", "all"), ("zero", "first"), ("zero", "rest")):
                chosen = meta["layers"] if which == "all" else meta["layers"][:1] if which == "first" else meta["layers"][1:] or meta["layers"][:1]
                layer_arg = "--layers " + ",".join(str(x) for x in chosen)
                jobs.append({
                    "id": f"ablate-{name}-{mode}-{which}", "account": meta["account"], "gpus": 1,
                    "minutes": 120, "dep": f"preflight-{meta['account']}",
                    "cmd": (
                        f"python -m ffnmem.run_lm_eval --ckpt {ckpt} --tasks {task_arg} "
                        f"--intervention {mode} {layer_arg} --device cuda "
                        f"--out results/eval/{name}/ablate_{mode}_{which}.json"
                    ),
                })
    mem_models = [n for n, m in MODELS.items() if m["kind"] == "memory" and n in ("w1_mem_full", "w1_mem_cut_near", "w2_mem_super1")]
    dense_models = ["w1_dense_full", "w1_dense_cut_near"]
    for task in FT_TASKS:
        for name in mem_models:
            for regime in FT_REGIMES_MEMORY:
                jobs.append(_ft(task, name, regime))
        for name in dense_models:
            jobs.append(_ft(task, name, "all"))
        jobs.append(_ft(task, "w1_dense_full", FT_REGIMES_DENSE[1]))
    for name in mem_models:
        for regime in ("backbone", "memory", "all", "tables"):
            jobs.append(_cpt(name, regime))
    for name in dense_models:
        jobs.append(_cpt(name, "all"))
    jobs.append(_cpt("w1_dense_full", "ffn_layers:2,3,4,5,12,13,14,15"))
    for name, meta in MODELS.items():
        if name in ("w1_mem_full", "w2_mem_super1", "w1_dense_full"):
            jobs.append({
                "id": f"interp-{name}", "account": meta["account"], "gpus": 1,
                "minutes": 90, "dep": f"preflight-{meta['account']}",
                "cmd": f"python -m ffnmem.interp.probesets --out results/interp/probes && python eval/run_local_suite.py",
            })
    for account in ("aditya", "sparsh"):
        jobs.append({
            "id": f"offload-bench-{account}", "account": account, "gpus": 1,
            "minutes": 40, "dep": f"preflight-{account}",
            "cmd": "python -m ffnmem.inference.bench --device cuda --out results/eval/bench_${SLURM_JOB_ID}.json",
        })
    return jobs


def _ft(task, name, regime):
    meta = MODELS[name]
    safe = regime.replace(":", "-").replace(",", "_")
    return {
        "id": f"ft-{task}-{name}-{safe}", "account": meta["account"], "gpus": 1,
        "minutes": 40, "dep": f"preflight-{meta['account']}",
        "cmd": (
            f"python -m ffnmem.finetune --task {task} --ckpt $CKPT/{name}_step3815.pt "
            f"--trainable {regime} --device cuda --out results/ft/{task}/{name}/{safe}.json"
        ),
    }


def _cpt(name, regime):
    meta = MODELS[name]
    safe = regime.replace(":", "-").replace(",", "_")
    return {
        "id": f"cpt-{name}-{safe}", "account": meta["account"], "gpus": 1,
        "minutes": 50, "dep": f"preflight-{meta['account']}",
        "cmd": (
            f"python -m ffnmem.train --config configs/{name}.yaml --init_from $CKPT/{name}_step3815.pt "
            f"--new_schedule --data_restart --data_dir $DOMAIN --set train.trainable={regime} "
            f"--set train.tokens_total=20000000 --set train.lr=0.0001 --max_steps 500"
        ),
    }


def _sbatch(job: dict, accounts: dict, dry: bool) -> str:
    acc = accounts[job["account"]]
    excl = f"-x {acc['exclude']}" if acc.get("exclude") else ""
    cons = f"--constraint={acc['constraint']}" if acc.get("constraint") else ""
    note = f"  # after {job['dep']}" if job.get("dep") else ""
    line = (
        f"sbatch -A {acc['account']} -p {acc['partition']} {cons} {excl} --gres=gpu:{job['gpus']} "
        f"-c {acc['cpus']} --mem={acc['mem']} -t 0:{job['minutes']}:00 -J {job['id']} "
        f"-o ~/morbius_eval/joblogs/%x-%j.out "
        f"--wrap 'cd ~/morbius_eval && export CKPT={acc.get('ckpt', '$CKPT')} && {job['cmd']}'"
    )
    return " ".join(line.split()) + note


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default="*")
    p.add_argument("--account", default=None)
    p.add_argument("--manifest", default=str(ROOT / "eval" / "manifest.yaml"))
    args = p.parse_args(argv)
    spec = yaml.safe_load(Path(args.manifest).read_text())
    accounts = spec["accounts"]
    for name, meta in accounts.items():
        meta["ckpt"] = spec["ckpt_root"][name]
    jobs = [j for j in _jobs() if fnmatch.fnmatch(j["id"], args.only)]
    if args.account:
        jobs = [j for j in jobs if j["account"] == args.account]
    print(f"# {len(jobs)} jobs")
    for job in jobs:
        print(_sbatch(job, accounts, args.dry_run))
    if not args.dry_run:
        print("# not submitted: pass this script's stdout to a shell on ada. Refusing to sbatch from here.")
    return jobs


if __name__ == "__main__":
    main()

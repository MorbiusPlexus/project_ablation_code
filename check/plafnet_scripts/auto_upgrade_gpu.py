#!/usr/bin/env python3
"""Project Morbius | Autonomous GPU Upgrade & Auto-Scaler Monitor.

Monitors Ada cluster (partition plafnet2) every N seconds.
If current training job is on 1 GPU (or 2 GPUs):
1. Detects when 4 GPUs become free on gnode118 (or 2 GPUs on any plafnet node).
2. Sends SIGUSR1 to the active job so it saves an exact atomic checkpoint at the
   current step boundary and cleanly exits.
3. Automatically launches the upgraded job (4 GPUs or 2 GPUs) with --resume auto
   to double or quadruple throughput seamlessly.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout.strip()


def get_current_job(user: str, config: str) -> dict | None:
    """Finds active or pending training job for the specified config."""
    out = run_cmd(["squeue", "-u", user, "-o", "%i|%j|%T|%b|%N|%D|%r", "-h"])
    if not out:
        return None

    # Matches config prefix e.g. w2_mem_delayed4 or w2_mem_d
    short_cfg = config[:10]
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 7:
            jid, jname, state, gres, nodelist, nodes, reason = parts[:7]
            if short_cfg in jname or config in jname:
                # Determine GPU count
                gpu_count = 1
                gpu_match = re.search(r"gpu:(\d+)", gres)
                if gpu_match:
                    gpu_count = int(gpu_match.group(1))
                elif nodes:
                    try:
                        n = int(nodes)
                        if n > 1:
                            gpu_count = n
                    except ValueError:
                        pass

                return {
                    "job_id": jid,
                    "name": jname,
                    "state": state,
                    "gres": gres,
                    "gpu_count": gpu_count,
                    "node": nodelist,
                    "reason": reason,
                }
    return None


def get_node_status(node: str) -> dict:
    """Parses total and allocated GPUs for a given node via scontrol."""
    out = run_cmd(["scontrol", "show", "node", node])
    if not out or f"NodeName={node}" not in out:
        return {"total_gpus": 0, "alloc_gpus": 0, "free_gpus": 0, "state": "UNKNOWN"}

    # Total GPUs
    total_gpus = 0
    gres_match = re.search(r"Gres=gpu:(\d+)", out)
    if gres_match:
        total_gpus = int(gres_match.group(1))

    # Allocated GPUs
    alloc_gpus = 0
    alloc_match = re.search(r"AllocTRES=.*?gres/gpu=(\d+)", out)
    if alloc_match:
        alloc_gpus = int(alloc_match.group(1))

    # State
    state = "UNKNOWN"
    state_match = re.search(r"State=(\w+)", out)
    if state_match:
        state = state_match.group(1)

    free_gpus = max(0, total_gpus - alloc_gpus)
    return {
        "node": node,
        "total_gpus": total_gpus,
        "alloc_gpus": alloc_gpus,
        "free_gpus": free_gpus,
        "state": state,
    }


def find_upgrade_target(current_gpus: int, current_node: str, current_state: str) -> int | None:
    """Evaluates whether an upgrade to 4 GPUs or 2 GPUs is possible right now."""
    # 1. Check gnode118 for 4-GPU upgrade
    status_118 = get_node_status("gnode118")
    free_on_118 = status_118["free_gpus"]

    # If current job is already occupying some GPUs on gnode118, treat them as reclaimable
    if current_node == "gnode118" and current_state == "RUNNING":
        free_on_118 += current_gpus

    if free_on_118 >= 4 and current_gpus < 4:
        return 4

    # 2. If currently on 1 GPU, check if 2 GPUs are available anywhere
    if current_gpus < 2:
        if free_on_118 >= 2:
            return 2
        for n in ("gnode114", "gnode115", "gnode116"):
            st = get_node_status(n)
            free = st["free_gpus"]
            if current_node == n and current_state == "RUNNING":
                free += current_gpus
            if free >= 2:
                return 2

    return None


def execute_upgrade(current_job: dict, target_gpus: int, config: str, dry_run: bool = False) -> bool:
    """Gracefully checkpoints current job and submits the upgraded job."""
    jid = current_job["job_id"]
    log(f"*** UPGRADE DETECTED: Upgrading from {current_job['gpu_count']} GPU(s) -> {target_gpus} GPU(s) ***")

    if dry_run:
        log(f"[DRY-RUN] Would send USR1 to job {jid} and submit {target_gpus}-GPU job.")
        return True

    # 1. Send USR1 for graceful checkpointing at next step boundary
    if current_job["state"] == "RUNNING":
        log(f"Sending USR1 to Job {jid} for step-boundary checkpointing...")
        run_cmd(["scancel", "--signal=B:USR1@0", jid])
        
        # Wait up to 90 seconds for job to exit cleanly
        log("Waiting for current job to save checkpoint and exit...")
        for _ in range(30):
            time.sleep(3)
            out = run_cmd(["squeue", "-j", jid, "-h"])
            if not out:
                log(f"Job {jid} exited cleanly with saved checkpoint!")
                break
        else:
            log(f"Job {jid} did not exit in 90s, forcing cancel...")
            run_cmd(["scancel", jid])
            time.sleep(5)
    else:
        log(f"Job {jid} was in state {current_job['state']}, cancelling directly...")
        run_cmd(["scancel", jid])
        time.sleep(3)

    # 2. Launch upgraded run
    log(f"Launching upgraded job ({target_gpus} GPUs)...")
    launcher = ROOT / "check" / "plafnet_scripts" / f"run_train_{config.split('_')[-1]}_plafnet.sh"
    if not launcher.exists():
        launcher = ROOT / "check" / "plafnet_scripts" / "run_train_delayed4_plafnet.sh"

    cmd = ["bash", str(launcher)]
    if target_gpus == 2:
        cmd.append("--2gpu")
    elif target_gpus == 4:
        cmd.extend(["--node", "gnode118"])

    log(f"Executing: {' '.join(cmd)}")
    launch_out = run_cmd(cmd)
    log(f"Launcher output:\n{launch_out}")

    # Verify new job
    time.sleep(3)
    user = os.environ.get("USER", "sparsh.bhartia")
    new_job = get_current_job(user, config)
    if new_job:
        log(f"Successfully launched upgraded Job {new_job['job_id']} ({target_gpus} GPUs, State: {new_job['state']})!")
        return True
    else:
        log("WARNING: Could not immediately detect new job in squeue. Check joblogs!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Autonomous GPU Upgrade Monitor for Ada Plafnet")
    parser.add_argument("--config", default="w2_mem_delayed4", help="Config name to monitor/upgrade")
    parser.add_argument("--user", default=os.environ.get("USER", "sparsh.bhartia"), help="Slurm username")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Simulate upgrade decisions without sending signals")
    args = parser.parse_args()

    log("=" * 70)
    log("  Project Morbius | Autonomous GPU Auto-Upgrade Monitor")
    log(f"  Target Config: {args.config} | User: {args.user}")
    log(f"  Poll Interval: {args.interval}s | Dry Run: {args.dry-run if hasattr(args, 'dry-run') else args.dry_run}")
    log("=" * 70)

    while True:
        try:
            job = get_current_job(args.user, args.config)
            if not job:
                log(f"No active job found for {args.config}. Will retry in {args.interval}s...")
                time.sleep(args.interval)
                continue

            jid = job["job_id"]
            state = job["state"]
            gpus = job["gpu_count"]
            node = job["node"]

            # If already running on 4 GPUs on gnode118, we are at max capacity!
            if gpus >= 4 and state == "RUNNING":
                log(f"Job {jid} is running at MAXIMUM capacity (4x RTX 3090 on {node}). No upgrade needed. Sleeping {args.interval*2}s...")
                time.sleep(args.interval * 2)
                continue

            log(f"Current Job: {jid} | State: {state} | GPUs: {gpus} | Node: {node}")

            target_upgrade = find_upgrade_target(gpus, node, state)
            if target_upgrade:
                success = execute_upgrade(job, target_upgrade, args.config, dry_run=args.dry_run)
                if success:
                    log(f"Grace period after upgrade. Sleeping 60s...")
                    time.sleep(60)
                    continue

            # Node status snapshot for logs
            st118 = get_node_status("gnode118")
            log(f"Cluster Status: gnode118 free GPUs = {st118['free_gpus']}/4 (Alloc: {st118['alloc_gpus']}/4)")

        except KeyboardInterrupt:
            log("Auto-upgrade monitor stopped by user.")
            break
        except Exception as e:
            log(f"Error during poll cycle: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()

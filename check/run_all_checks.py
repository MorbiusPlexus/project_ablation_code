#!/usr/bin/env python3
"""Master test runner for Project Morbius Pivot Ideas 1 & 2.

Executes all verification suites:
1. Idea 1 Parameter Matching (369.77M params, 0.0027% residual)
2. Idea 1 Model Forward/Backward & Zero-out probe
3. Idea 2 Delayed Injection (Delta=4) Model & Gradient Backprop
4. Idea 2 Host-RAM Latency Harness (Decode loop & Offload timing)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CHECK_DIR = Path(__file__).resolve().parent

TESTS = [
    ("Idea 1: Parameter Accounting (R=184 at Layer 1)", [sys.executable, str(CHECK_DIR / "idea1" / "verify_super1_params.py")]),
    ("Idea 1: Super-Memory Forward/Backward & Probe", [sys.executable, str(CHECK_DIR / "idea1" / "test_super1_model.py")]),
    ("Idea 2: Delayed Injection (Delta=4) Verification", [sys.executable, str(CHECK_DIR / "idea2" / "test_delayed_injection.py")]),
    ("Idea 2: Host-RAM Latency Harness Dry-Run", [sys.executable, str(CHECK_DIR / "idea2" / "bench_offload_latency.py"), "--tokens", "20", "--warmup", "5"]),
]


def run_all():
    print("=" * 75)
    print("      PROJECT MORBIUS: COMPREHENSIVE VERIFICATION SUITE")
    print("                   (PIVOT IDEAS 1 & 2)")
    print("=" * 75)
    print()

    results = []
    overall_start = time.time()

    for name, cmd in TESTS:
        print(f">>> Running: {name} ...")
        t0 = time.time()
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"    [PASSED] ({elapsed:.2f}s)\n")
            results.append((name, "PASSED", elapsed, res.stdout))
        else:
            print(f"    [FAILED] ({elapsed:.2f}s)\n")
            print("--- Output ---")
            print(res.stdout)
            print("--------------")
            results.append((name, "FAILED", elapsed, res.stdout))

    overall_time = time.time() - overall_start

    print("=" * 75)
    print("                      TEST SUMMARY REPORT")
    print("=" * 75)
    print(f"{'Test Suite':<50} {'Status':<12} {'Duration':<10}")
    print("-" * 75)
    all_passed = True
    for name, status, duration, _ in results:
        status_str = f"\033[92m{status}\033[0m" if status == "PASSED" else f"\033[91m{status}\033[0m"
        print(f"{name:<50} {status_str:<21} {duration:>6.2f}s")
        if status != "PASSED":
            all_passed = False
    print("-" * 75)
    print(f"Total Time: {overall_time:.2f}s")
    print("=" * 75)

    if all_passed:
        print("\nALL CHECKS PASSED SUCCESSFULLY! Ready for deployment on Ada Plafnet nodes.\n")
        sys.exit(0)
    else:
        print("\nSOME CHECKS FAILED. Please review the output logs above.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_all()

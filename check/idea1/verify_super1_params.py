#!/usr/bin/env python3
"""Verification script for Idea 1: Single-Layer Super-Memory (w2_mem_super1).

Checks:
1. Exact non-embedding parameter accounting and matching against reference run (w1_mem_cut_near).
2. Elimination of Layer 11 memory module and consolidation of budget to Layer 1 (R = 184).
3. Parameter difference is within the strict 0.5% tolerance (target < 0.01%).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ffnmem.config import load_config
from ffnmem.params import check_match, model_counts


def verify_super1_parameters():
    print("=" * 70)
    print("IDEA 1: Single-Layer Super-Memory Parameter Accounting Check")
    print("=" * 70)

    cfg_super1_path = ROOT / "configs" / "w2_mem_super1.yaml"
    cfg_ref_path = ROOT / "configs" / "w1_mem_cut_near.yaml"

    if not cfg_super1_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_super1_path}")
    if not cfg_ref_path.exists():
        raise FileNotFoundError(f"Missing reference config: {cfg_ref_path}")

    cfg_super1 = load_config(cfg_super1_path)
    cfg_ref = load_config(cfg_ref_path)

    ref_counts = model_counts(cfg_ref["model"])
    super1_counts = model_counts(cfg_super1["model"])

    ref_non_emb = ref_counts["non_embedding"]
    super1_non_emb = super1_counts["non_embedding"]
    diff = super1_non_emb - ref_non_emb
    rel_error = abs(diff) / ref_non_emb

    print(f"Reference Run (w1_mem_cut_near):")
    print(f"  Memory Layers: {cfg_ref['model']['memory_layers']}")
    print(f"  Memory Routes: {cfg_ref['model']['memory']['routes']}")
    print(f"  Non-Embedding Parameters: {ref_non_emb:,}")
    print()
    print(f"Proposed Super-Memory (w2_mem_super1):")
    print(f"  Memory Layers: {cfg_super1['model']['memory_layers']}")
    print(f"  Memory Routes: {cfg_super1['model']['memory']['routes']}")
    print(f"  Non-Embedding Parameters: {super1_non_emb:,}")
    print(f"  Difference: {diff:+d} parameters ({rel_error:.4%})")
    print()

    print(f"{'Component':<25} {'Reference (w1)':<20} {'Super-Memory (w2)':<20}")
    print("-" * 65)
    for comp in ("embedding", "attention", "ffn", "norms", "memory_tables", "memory_readout", "total", "non_embedding"):
        ref_val = ref_counts.get(comp, 0)
        s1_val = super1_counts.get(comp, 0)
        print(f"{comp:<25} {ref_val:>18,} {s1_val:>18,}")
    print("-" * 65)

    # Assertions
    assert cfg_super1["model"]["memory_layers"] == [1], f"Expected memory_layers=[1], got {cfg_super1['model']['memory_layers']}"
    assert cfg_super1["model"]["memory"]["routes"] == 184, f"Expected routes=184, got {cfg_super1['model']['memory']['routes']}"
    assert rel_error < 0.005, f"Parameter residual {rel_error:.4%} exceeds 0.5% tolerance!"

    print()
    print(f"SUCCESS: w2_mem_super1 perfectly matches reference within {rel_error:.4%} (target was <0.5%)!")
    print("=" * 70)


if __name__ == "__main__":
    verify_super1_parameters()

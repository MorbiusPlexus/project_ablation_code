#!/usr/bin/env python3
"""Functional & gradient verification for Idea 1: Single-Layer Super-Memory.

Tests:
1. Architecture verification: Memory is present only at Layer 1 with R=184 routes,
   and absent at Layer 11 and all other layers.
2. Forward pass with synthetic inputs (verifying shape & logits).
3. Backward pass (verifying gradient computation on R=184 router & tables).
4. Zero-out probe sensitivity check on Layer 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ffnmem.config import load_config
from ffnmem.model import Transformer


def test_super1_model():
    print("=" * 70)
    print("IDEA 1: Single-Layer Super-Memory Forward/Backward & Probe Test")
    print("=" * 70)

    cfg = load_config(ROOT / "configs" / "w2_mem_super1.yaml")
    # Use compact dimensions for fast execution
    cfg["model"]["seq_len"] = 32
    cfg["model"]["vocab"] = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = Transformer(cfg["model"]).to(device)
    model.train()

    # 1. Structure Verification
    mem_layers = [i for i, b in enumerate(model.blocks) if b.memory is not None]
    print(f"Memory-bearing layers: {mem_layers}")
    assert mem_layers == [1], f"Expected memory only at layer 1, found {mem_layers}"
    assert model.blocks[1].memory.routes == 184, f"Expected 184 routes, found {model.blocks[1].memory.routes}"
    assert model.blocks[11].memory is None, "Layer 11 should have NO memory module!"
    print("Architecture verification passed: Layer 1 holds R=184 Super-Memory; Layer 11 has none.")

    # 2. Forward Pass
    B, T = 2, 32
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)

    # Initialize readout_o_proj with small weights so gradients flow at init
    torch.nn.init.normal_(model.blocks[1].memory.module.readout_o_proj.weight, std=0.02)

    logits, loss = model(x, y)
    print(f"Forward pass completed. Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    assert logits.shape == (B, T, 256), f"Unexpected logits shape: {logits.shape}"

    # 3. Backward Pass
    loss.backward()
    router_grad = model.blocks[1].memory.router.weight.grad
    assert router_grad is not None, "Router gradients should not be None!"
    print(f"Router gradient norm: {router_grad.norm().item():.4e}")
    table2_grad = model.blocks[1].memory.module.tables.ngram_2.weight.grad
    assert table2_grad is not None, "Table ngram_2 gradients should not be None!"
    print(f"Table (ngram=2) gradient norm: {table2_grad.norm().item():.4e}")

    # 4. Zero-Out Probe Verification on Layer 1
    model.eval()
    with torch.no_grad():
        logits_normal = model(x)

        # Intercept memory output and zero it out
        mem = model.blocks[1].memory
        orig_forward = mem.forward
        ratios = []

        def zero_forward(h, h_norm):
            out = orig_forward(h, h_norm)
            ratio = (out.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)).mean().item()
            ratios.append(ratio)
            return torch.zeros_like(out)

        mem.forward = zero_forward
        logits_zeroed = model(x)
        mem.forward = orig_forward

        diff = (logits_normal - logits_zeroed).abs().mean().item()
        print(f"Zero-out probe difference on Layer 1: {diff:.6f}")
        print(f"Memory output-to-residual norm ratio: {ratios[0]:.4f}")
        assert diff > 0.0, "Zeroing memory output should change network predictions!"

    print()
    print("SUCCESS: Single-Layer Super-Memory model passed all checks!")
    print("=" * 70)


if __name__ == "__main__":
    test_super1_model()

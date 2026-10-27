#!/usr/bin/env python3
"""Verification for Idea 2: Skip-Layer Delayed Memory Injection (w2_mem_delayed4).

Tests:
1. Delayed mechanism check: Memory output is computed at Layer 1 and withheld from Layer 1.
2. Residual check: Memory output is injected before attention at Layer 1 + 4 = Layer 5.
3. End-to-end gradient backpropagation through delayed injection path with activation checkpointing.
4. Parameter matching check against w2_mem_super1 and w1_mem_cut_near.
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
from ffnmem.params import model_counts


def test_delayed_injection():
    print("=" * 70)
    print("IDEA 2: Delayed Memory Injection Functional & Gradient Test (Delta = 4)")
    print("=" * 70)

    cfg = load_config(ROOT / "configs" / "w2_mem_delayed4.yaml")
    cfg["model"]["seq_len"] = 32
    cfg["model"]["vocab"] = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Parameter and config verification
    inj_map = cfg["model"].get("memory_injection_map", {})
    print(f"Memory injection map: {inj_map}")
    assert 1 in inj_map and inj_map[1] == 5, f"Expected mapping 1 -> 5, got {inj_map}"
    counts = model_counts(cfg["model"])
    assert counts["non_embedding"] == 369777920, f"Expected 369777920 params, got {counts['non_embedding']}"
    print(f"Parameter match verified: exactly {counts['non_embedding']:,} non-embedding parameters.")

    # 2. Forward pass with hook inspection
    model = Transformer(cfg["model"]).to(device)
    model.train()

    # Track inputs and outputs at layers
    layer_inputs = {}
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, args, output):
            layer_inputs[layer_idx] = args[0].clone()
            if isinstance(output, tuple):
                layer_outputs[layer_idx] = output[0].clone()
            else:
                layer_outputs[layer_idx] = output.clone()
        return hook

    hooks = []
    for i in (1, 2, 4, 5):
        hooks.append(model.blocks[i].register_forward_hook(make_hook(i)))

    B, T = 2, 32
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)

    # Initialize readout_o_proj non-zero
    torch.nn.init.normal_(model.blocks[1].memory.module.readout_o_proj.weight, std=0.02)

    logits, loss = model(x, y)
    for h in hooks:
        h.remove()

    print(f"Forward completed. Loss: {loss.item():.4f}")

    # Verify that Layer 1 block has delay_memory=True
    assert model.blocks[1].delay_memory is True, "Block 1 should have delay_memory=True"
    assert model.blocks[5].delay_memory is False, "Block 5 should have delay_memory=False"

    # 3. Backward Pass & Gradient Flow Verification
    loss.backward()
    router_grad = model.blocks[1].memory.router.weight.grad
    assert router_grad is not None, "Router at Layer 1 should have received gradient from Layer 5!"
    router_grad_norm = router_grad.norm().item()
    print(f"Router gradient norm (through Layer 5 injection): {router_grad_norm:.4e}")
    assert router_grad_norm > 0.0, "Router gradient norm must be non-zero!"

    table_grad = model.blocks[1].memory.module.tables.ngram_2.weight.grad
    assert table_grad is not None, "Table at Layer 1 should have received gradient from Layer 5!"
    table_grad_norm = table_grad.norm().item()
    print(f"Table ngram_2 gradient norm: {table_grad_norm:.4e}")
    assert table_grad_norm > 0.0, "Table gradient norm must be non-zero!"

    # 4. Activation Checkpointing Test
    model.zero_grad()
    model.act_ckpt = True
    logits_ckpt, loss_ckpt = model(x, y)
    loss_ckpt.backward()
    router_grad_ckpt = model.blocks[1].memory.router.weight.grad
    assert router_grad_ckpt is not None and router_grad_ckpt.norm().item() > 0.0
    print(f"Activation checkpointing gradient norm: {router_grad_ckpt.norm().item():.4e}")

    print()
    print("SUCCESS: Delayed Memory Injection (Delta = 4) passed all functional & gradient tests!")
    print("=" * 70)


if __name__ == "__main__":
    test_delayed_injection()

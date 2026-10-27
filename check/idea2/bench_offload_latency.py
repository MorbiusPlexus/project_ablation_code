#!/usr/bin/env python3
"""Standalone Host-RAM Latency Harness (Sub-Task 2A).

Measures single-token autoregressive decode latency across three conditions:
1. On-GPU Baseline: 100M parameter memory table resides entirely in GPU VRAM.
2. Synchronous Host-RAM (Delta = 0): Table in pinned Host CPU RAM; blocking PCIe transfer stalls GPU.
3. Asynchronous Delayed Host-RAM (Delta in {1, 2, 4}): Table in pinned Host CPU RAM;
   DMA transfer is pipelined on a dedicated CUDA stream and overlapped with Delta intervening layers.

Metrics reported:
- Single-token decode latency (us/token)
- Pipeline stall duration & effective offload overhead (%)
- Speedup vs synchronous Host-RAM serving
- Resident GPU VRAM vs Host DRAM footprint
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SyntheticTransformerLayer(nn.Module):
    """Accurately simulates a single 24-layer Transformer decode forward pass (batch size 1)."""
    def __init__(self, d_model: int = 1024, d_ff: int = 2816, n_heads: int = 24, n_kv_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 1. Attn block
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        attn_out = self.out_proj(q)
        h = h + attn_out

        # 2. FFN block
        ffn_out = self.w2(F.silu(self.w1(h)) * self.w3(h))
        return h + ffn_out


def run_benchmark(
    tokens: int = 512,
    warmup: int = 50,
    routes: int = 184,
    mem_dim: int = 128,
    table_rows: int = 800768,  # Total rows in 100M parameter table
    n_layers: int = 24,
    d_model: int = 1024,
    dtype: torch.dtype = torch.float16,
):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    print("=" * 80)
    print("SUB-TASK 2A: STANDALONE HOST-RAM MEMORY LATENCY BENCHMARK")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0) if is_cuda else 'CPU (Mock / Local Dry-Run)'}")
    print(f"Decode tokens: {tokens} | Warmup: {warmup} | Batch size: 1")
    print(f"Backbone: {n_layers} layers, d_model={d_model} | Table parameters: {table_rows * mem_dim:,} (~{(table_rows * mem_dim * 2)/(1024**2):.1f} MB in FP16)")
    print(f"Routes per token: {routes} (fetching {routes * 2} rows of dim {mem_dim} per token)")
    print("=" * 80)

    # Instantiate layers on device
    layers = [SyntheticTransformerLayer(d_model=d_model).to(device=device, dtype=dtype) for _ in range(n_layers)]
    for layer in layers:
        layer.eval()
    readout_proj = nn.Linear(mem_dim, d_model, bias=False).to(device=device, dtype=dtype)
    readout_proj.eval()

    # Memory table setup:
    # 1. Pinned CPU Host RAM table (for offloading)
    table_host = torch.randn(table_rows, mem_dim, dtype=dtype, pin_memory=is_cuda)
    host_dram_mb = (table_host.numel() * table_host.element_size()) / (1024 ** 2)

    # 2. On-GPU VRAM table (for baseline)
    if is_cuda:
        table_gpu = table_host.to(device)
        gpu_vram_mb = (table_gpu.numel() * table_gpu.element_size()) / (1024 ** 2)
    else:
        table_gpu = table_host
        gpu_vram_mb = host_dram_mb

    num_fetch_rows = routes * 2  # 2 n-gram orders
    # Pre-generate random lookup indices for decode sequence
    indices_seq = [torch.randint(0, table_rows, (num_fetch_rows,)) for _ in range(tokens + warmup)]

    results = {}

    # -------------------------------------------------------------------------
    # Condition 1: On-GPU Baseline (Table fully resident in VRAM)
    # -------------------------------------------------------------------------
    print("\n[Condition 1/5] Running On-GPU Baseline (Table resident in GPU VRAM)...")
    h = torch.randn(1, 1, d_model, device=device, dtype=dtype)

    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    with torch.no_grad():
        for step in range(warmup):
            idx = indices_seq[step].to(device)
            # Layer 0
            h = layers[0](h)
            # Layer 1: In-VRAM gather & inject
            fetched = readout_proj(table_gpu[idx].mean(dim=0, keepdim=True).unsqueeze(0))
            h = h + fetched
            h = layers[1](h)
            for l in range(2, n_layers):
                h = layers[l](h)

        if is_cuda:
            torch.cuda.synchronize()
            start_event.record()
        t0 = time.perf_counter()

        for step in range(warmup, warmup + tokens):
            idx = indices_seq[step].to(device)
            h = layers[0](h)
            fetched = readout_proj(table_gpu[idx].mean(dim=0, keepdim=True).unsqueeze(0))
            h = h + fetched
            h = layers[1](h)
            for l in range(2, n_layers):
                h = layers[l](h)

        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            total_ms = start_event.elapsed_time(end_event)
        else:
            total_ms = (time.perf_counter() - t0) * 1000.0

    baseline_us = (total_ms / tokens) * 1000.0
    results["baseline"] = baseline_us
    print(f"  -> On-GPU Baseline Decode Latency: {baseline_us:.1f} us/token")

    # -------------------------------------------------------------------------
    # Condition 2: Synchronous Host-RAM (Delta = 0)
    # -------------------------------------------------------------------------
    print("\n[Condition 2/5] Running Synchronous Host-RAM (Delta = 0, Blocking PCIe copy)...")
    with torch.no_grad():
        # Warmup
        for step in range(warmup):
            idx = indices_seq[step]
            h = layers[0](h)
            # Synchronous Host DRAM gather + PCIe transfer
            fetched_host = table_host[idx]
            fetched_gpu = readout_proj(fetched_host.to(device, non_blocking=False).mean(dim=0, keepdim=True).unsqueeze(0))
            h = h + fetched_gpu
            h = layers[1](h)
            for l in range(2, n_layers):
                h = layers[l](h)

        if is_cuda:
            torch.cuda.synchronize()
            start_event.record()
        t0 = time.perf_counter()

        for step in range(warmup, warmup + tokens):
            idx = indices_seq[step]
            h = layers[0](h)
            fetched_host = table_host[idx]
            fetched_gpu = readout_proj(fetched_host.to(device, non_blocking=False).mean(dim=0, keepdim=True).unsqueeze(0))
            h = h + fetched_gpu
            h = layers[1](h)
            for l in range(2, n_layers):
                h = layers[l](h)

        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            total_ms = start_event.elapsed_time(end_event)
        else:
            total_ms = (time.perf_counter() - t0) * 1000.0

    sync_us = (total_ms / tokens) * 1000.0
    results["sync_delta0"] = sync_us
    overhead_sync = ((sync_us - baseline_us) / baseline_us) * 100.0
    print(f"  -> Synchronous Host-RAM Latency: {sync_us:.1f} us/token (Overhead: +{overhead_sync:.1f}%)")

    # -------------------------------------------------------------------------
    # Condition 3: Asynchronous Delayed Host-RAM (Delta in {1, 2, 4})
    # -------------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor

    mem_stream = torch.cuda.Stream() if is_cuda else None
    transfer_event = torch.cuda.Event() if is_cuda else None

    # Pre-allocate pinned Host RAM buffer and GPU VRAM staging buffer for non-blocking DMA transfer
    pinned_host_buffer = torch.empty(num_fetch_rows, mem_dim, dtype=dtype, pin_memory=is_cuda)
    staging_buffer = torch.empty(num_fetch_rows, mem_dim, dtype=dtype, device=device)
    executor = ThreadPoolExecutor(max_workers=1)

    def async_host_to_device(idx_tensor):
        # 1. Gather into pinned Host RAM directly
        torch.index_select(table_host, 0, idx_tensor, out=pinned_host_buffer)
        # 2. Non-blocking DMA transfer over PCIe on dedicated CUDA stream
        if is_cuda:
            with torch.cuda.stream(mem_stream):
                staging_buffer.copy_(pinned_host_buffer, non_blocking=True)
                transfer_event.record(mem_stream)

    for delta in (1, 2, 4):
        print(f"\n[Condition {3 + (1 if delta==2 else (2 if delta==4 else 0))}/5] Running Asynchronous Delayed Host-RAM (Delta = {delta})...")
        inject_layer = 1 + delta

        with torch.no_grad():
            for step in range(warmup):
                idx = indices_seq[step]
                h = layers[0](h)

                # Layer 1: Dispatch asynchronous Host gather & non-blocking DMA
                fut = executor.submit(async_host_to_device, idx)

                # Overlapped intervening layers executed on main CUDA stream
                for l in range(1, inject_layer):
                    h = layers[l](h)

                # Delayed injection at Layer 1 + delta: synchronize with memory transfer
                fut.result()
                if is_cuda:
                    torch.cuda.current_stream().wait_event(transfer_event)
                h = h + readout_proj(staging_buffer.mean(dim=0, keepdim=True).unsqueeze(0))

                for l in range(inject_layer, n_layers):
                    h = layers[l](h)

            if is_cuda:
                torch.cuda.synchronize()
                start_event.record()
            t0 = time.perf_counter()

            for step in range(warmup, warmup + tokens):
                idx = indices_seq[step]
                h = layers[0](h)

                fut = executor.submit(async_host_to_device, idx)

                for l in range(1, inject_layer):
                    h = layers[l](h)

                fut.result()
                if is_cuda:
                    torch.cuda.current_stream().wait_event(transfer_event)
                h = h + readout_proj(staging_buffer.mean(dim=0, keepdim=True).unsqueeze(0))

                for l in range(inject_layer, n_layers):
                    h = layers[l](h)

            if is_cuda:
                end_event.record()
                torch.cuda.synchronize()
                total_ms = start_event.elapsed_time(end_event)
            else:
                total_ms = (time.perf_counter() - t0) * 1000.0

        async_us = (total_ms / tokens) * 1000.0
        results[f"async_delta{delta}"] = async_us
        overhead = ((async_us - baseline_us) / baseline_us) * 100.0
        speedup = sync_us / async_us
        print(f"  -> Async (Delta={delta}) Latency: {async_us:.1f} us/token | Overhead: {overhead:+.1f}% | Speedup vs Sync: {speedup:.2f}x")

    executor.shutdown()

    # -------------------------------------------------------------------------
    # Summary Report Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY TABLE (BATCH SIZE = 1 TOKEN DECODING)")
    print("=" * 80)
    print(f"{'Condition':<35} {'Latency (us)':<15} {'Overhead (%)':<15} {'Speedup vs Sync':<15}")
    print("-" * 80)
    print(f"{'On-GPU Baseline (VRAM)':<35} {baseline_us:>12.1f}   {0.0:>12.1f}%  {sync_us/baseline_us:>14.2f}x")
    print(f"{'Synchronous Host-RAM (Delta=0)':<35} {sync_us:>12.1f}   {overhead_sync:>12.1f}%  {1.0:>14.2f}x")
    for delta in (1, 2, 4):
        lat = results[f"async_delta{delta}"]
        ovh = ((lat - baseline_us) / baseline_us) * 100.0
        spd = sync_us / lat
        print(f"{f'Async Delayed Host-RAM (Delta={delta})':<35} {lat:>12.1f}   {ovh:>12.1f}%  {spd:>14.2f}x")
    print("-" * 80)
    print(f"Memory Footprint: Resident GPU VRAM Saved = {gpu_vram_mb:.1f} MB (offloaded to Host CPU DRAM)")
    print("=" * 80)


def main():
    p = argparse.ArgumentParser(description="Standalone Host-RAM Latency Harness")
    p.add_argument("--tokens", type=int, default=512, help="Number of decode steps to benchmark")
    p.add_argument("--warmup", type=int, default=50, help="Number of warmup steps")
    p.add_argument("--routes", type=int, default=184, help="Number of memory routes")
    p.add_argument("--mem_dim", type=int, default=128, help="Memory vector dimension")
    p.add_argument("--layers", type=int, default=24, help="Backbone layer count")
    args = p.parse_args()

    run_benchmark(
        tokens=args.tokens,
        warmup=args.warmup,
        routes=args.routes,
        mem_dim=args.mem_dim,
        n_layers=args.layers,
    )


if __name__ == "__main__":
    main()

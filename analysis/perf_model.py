"""Latency model for host-RAM tables with delayed injection.

Decode step at batch B with injection delay n:

    T(n, B) = sum_l t_l(B) + t_readout(B) + max(0, T_fetch(B) - S_n(B)) + c_sync

    T_fetch = t_D2H + a_gather + bytes/BW_gather + t_pcie + bytes/BW_pcie
    S_n     = time from the end of the router to the injection point

n* is the smallest n whose overlap window covers the fetch. The quality
constraint picks the smallest n whose measured loss increase stays within
``loss_tol`` of the n=0 model.
"""

from __future__ import annotations

import json
from pathlib import Path


def fetch_us(batch: int, routes: int, orders: int, dim: int, elem_bytes: int,
             bw_gather_Bps: float, bw_pcie_Bps: float, t_d2h_us: float, t_pcie_us: float, a_gather_us: float) -> float:
    payload = batch * routes * orders * dim * elem_bytes
    return t_d2h_us + a_gather_us + payload / bw_gather_Bps * 1e6 + t_pcie_us + payload / bw_pcie_Bps * 1e6


def overlap_us(n: int, layer_us: list[float]) -> float:
    """Layers 1..n after the router, which sits at the start of layer 1."""
    if n <= 0:
        return 0.0
    return float(sum(layer_us[1 : n + 1]))


def step_us(n: int, layer_us: list[float], readout_us: float, fetch: float, c_sync_us: float) -> float:
    stall = max(0.0, fetch - overlap_us(n, layer_us))
    return float(sum(layer_us)) + readout_us + stall + c_sync_us


def n_star(layer_us: list[float], fetch: float) -> int:
    for n in range(0, len(layer_us)):
        if overlap_us(n, layer_us) >= fetch:
            return n
    return len(layer_us) - 1


def n_star_quality(loss_by_n: dict[int, float], latency_by_n: dict[int, float], tol: float = 0.005) -> int | None:
    if 0 not in loss_by_n:
        return None
    base = loss_by_n[0]
    ok = [n for n, loss in loss_by_n.items() if loss - base <= tol]
    if not ok:
        return 0
    return min(ok, key=lambda n: latency_by_n.get(n, 0.0))


def calibrate_from_bench(bench: dict, routes: int = 4, orders: int = 2, dim: int = 16) -> dict:
    """Turn one bench JSON into model constants. Bandwidths are lower bounds from the measured times."""
    rows = int(bench.get("table_rows") or 1)
    width = int(bench.get("table_dim") or dim)
    gather_s = float(bench.get("gather_ms") or 1.0) / 1000.0
    h2d_s = float(bench.get("h2d_ms") or 1.0) / 1000.0
    bytes_gather = 64 * width * 2
    bytes_h2d = min(64, rows) * width * 2
    bw_g = bytes_gather / max(gather_s, 1e-9)
    bw_p = bytes_h2d / max(h2d_s, 1e-9)
    n_layers = 4
    layer = float(bench.get("layer_ms") or 1.0) * 1000.0 / n_layers  # us, uniform split of one forward
    return {
        "bw_gather_Bps": bw_g,
        "bw_pcie_Bps": bw_p,
        "layer_us": [layer] * n_layers,
        "readout_us": layer * 0.25,
        "t_d2h_us": 10.0,
        "t_pcie_us": 5.0,
        "a_gather_us": 5.0,
        "routes": routes,
        "orders": orders,
        "dim": width,
    }


def predict(constants: dict, batch: int = 1) -> dict:
    fetch = fetch_us(
        batch, constants["routes"], constants["orders"], constants["dim"], 2,
        constants["bw_gather_Bps"], constants["bw_pcie_Bps"],
        constants["t_d2h_us"], constants["t_pcie_us"], constants["a_gather_us"],
    )
    layers = constants["layer_us"]
    n = n_star(layers, fetch)
    curve = {i: step_us(i, layers, constants["readout_us"], fetch, 0.0) for i in range(len(layers))}
    return {"fetch_us": fetch, "n_star": n, "step_us": curve}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default=None)
    p.add_argument("--out", default="results/eval/perf_model.json")
    args = p.parse_args()
    if args.bench and Path(args.bench).exists():
        constants = calibrate_from_bench(json.loads(Path(args.bench).read_text()))
    else:
        constants = {
            "bw_gather_Bps": 2e9, "bw_pcie_Bps": 12e9, "layer_us": [80.0] * 24,
            "readout_us": 20.0, "t_d2h_us": 10.0, "t_pcie_us": 5.0, "a_gather_us": 8.0,
            "routes": 184, "orders": 1, "dim": 128,
        }
    pred = {b: predict(constants, b) for b in (1, 4, 16)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"constants": constants, "predict": pred}, indent=2), encoding="utf-8")
    print(json.dumps(pred))


if __name__ == "__main__":
    main()

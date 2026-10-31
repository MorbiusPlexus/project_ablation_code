"""HTML heatmaps of per-token memory reliance. The colour is the per-token loss gap."""

from __future__ import annotations

import html
from pathlib import Path


def _color(value: float, hi: float) -> str:
    t = 0.0 if hi <= 0 else max(0.0, min(1.0, value / hi))
    red = int(255 * t)
    return f"rgb({red},{int(40 + 80 * (1 - t))},{int(40 + 40 * (1 - t))})"


def write_heatmap(tokens: list[str], values: list[float], path: str | Path, title: str) -> None:
    hi = max(values) if values else 1.0
    cells = []
    for tok, val in zip(tokens, values):
        cells.append(
            f'<span title="{val:.4f}" style="background:{_color(val, hi)};padding:2px 4px;margin:1px;">{html.escape(tok)}</span>'
        )
    body = " ".join(cells)
    page = f"<!DOCTYPE html><meta charset='utf-8'><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><p>{body}</p>\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page, encoding="utf-8")

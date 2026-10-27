#!/usr/bin/env python3
"""Generates publication-quality figures for Idea 1 (Single-Layer Super-Memory w2_mem_super1)
and saves them both to results/figures/ and the Antigravity artifact directory.
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
ARTIFACT_DIR = Path("/home/sparsh-bhartia/.gemini/antigravity-cli/brain/83b67581-66fb-4613-a308-0b9b83dc987d/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Data Points
steps = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3815])
tokens_b = steps * 256 * 2048 / 1e9  # Tokens in Billions

# Validation losses
loss_run1_mem_full = np.array([4.888, 3.931, 3.517, 3.341, 3.234, 3.156, 3.097, 3.047, 3.005, 2.970, 2.938, 2.911, 2.886, 2.865, 2.847, 2.833, 2.821, 2.812, 2.805])
loss_run2_mem_cut_near = np.array([4.984, 4.011, 3.538, 3.341, 3.222, 3.143, 3.082, 3.030, 2.990, 2.954, 2.923, 2.895, 2.871, 2.850, 2.833, 2.818, 2.807, 2.798, 2.791])
loss_run4_dense_full = np.array([4.906, 3.932, 3.534, 3.362, 3.244, 3.163, 3.101, 3.050, 3.008, 2.970, 2.938, 2.910, 2.885, 2.864, 2.845, 2.830, 2.818, 2.809, 2.802])
loss_run5_dense_cut = np.array([5.268, 4.085, 3.556, 3.347, 3.223, 3.142, 3.077, 3.024, 2.981, 2.944, 2.911, 2.882, 2.857, 2.836, 2.818, 2.802, 2.791, 2.781, 2.774])
loss_super1 = np.array([4.971, 3.977, 3.509, 3.330, 3.214, 3.135, 3.075, 3.025, 2.985, 2.949, 2.918, 2.891, 2.867, 2.846, 2.829, 2.815, 2.803, 2.794, 2.7875])

# Diagnostics for Super1
super1_sink = np.array([0.4143, 0.3915, 0.4031, 0.4025, 0.4072, 0.4107, 0.4110, 0.4134, 0.4137, 0.4142, 0.4139, 0.4147, 0.4141, 0.4155, 0.4151, 0.4151, 0.4153, 0.4154, 0.4150])
super1_codes = np.array([13.31, 14.40, 14.67, 14.78, 14.79, 14.85, 14.91, 14.90, 14.95, 14.93, 14.97, 14.96, 14.95, 14.96, 14.94, 14.97, 14.97, 14.99, 14.96])

def save_fig(fig, name):
    p1 = FIG_DIR / name
    p2 = ARTIFACT_DIR / name
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    shutil.copy(p1, p2)
    plt.close(fig)
    print(f"Saved: {p1} and {p2}")

# ==============================================================================
# Plot 1: Validation Loss Convergence Curves (Full & Zoomed)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Full training trajectory
ax1.plot(tokens_b, loss_run1_mem_full, "o-", color="#8c564b", alpha=0.7, label="Run 1: Mem Full (L1+L11, 2.805)")
ax1.plot(tokens_b, loss_run4_dense_full, "s--", color="#7f7f7f", alpha=0.7, label="Run 4: Dense Full (2.802)")
ax1.plot(tokens_b, loss_run2_mem_cut_near, "^-", color="#1f77b4", label="Run 2: Mem Cut Near (L1+L11, 2.791)")
ax1.plot(tokens_b, loss_run5_dense_cut, "d-.", color="#2ca02c", alpha=0.8, label="Run 5: Dense Cut Near (2.774)")
ax1.plot(tokens_b, loss_super1, "*-", color="#d62728", linewidth=2.5, markersize=8, label="Idea 1: Super-Memory (L1 only, 2.788)")

ax1.set_title("(A) Full Validation Loss Trajectory (2.0B Tokens)", pad=10)
ax1.set_xlabel("Processed Tokens (Billions)")
ax1.set_ylabel("Validation Loss (Nats)")
ax1.grid(True)
ax1.legend(loc="upper right", framealpha=0.9)

# Zoomed trajectory (1.0B to 2.0B tokens)
mask = tokens_b >= 1.0
ax2.plot(tokens_b[mask], loss_run1_mem_full[mask], "o-", color="#8c564b", alpha=0.7, label="Run 1 (Mem Full)")
ax2.plot(tokens_b[mask], loss_run4_dense_full[mask], "s--", color="#7f7f7f", alpha=0.7, label="Run 4 (Dense Full)")
ax2.plot(tokens_b[mask], loss_run2_mem_cut_near[mask], "^-", color="#1f77b4", linewidth=2.2, label="Run 2 (Mem Cut Near, 2.791)")
ax2.plot(tokens_b[mask], loss_run5_dense_cut[mask], "d-.", color="#2ca02c", alpha=0.8, label="Run 5 (Dense Cut Near, 2.774)")
ax2.plot(tokens_b[mask], loss_super1[mask], "*-", color="#d62728", linewidth=2.8, markersize=9, label="Idea 1: Super-Memory (2.788)")

ax2.set_title("(B) Late Convergence Zoom (1.0B - 2.0B Tokens)", pad=10)
ax2.set_xlabel("Processed Tokens (Billions)")
ax2.set_ylabel("Validation Loss (Nats)")
ax2.grid(True)
ax2.legend(loc="upper right", framealpha=0.9)

# Annotation for Super1 advantage
ax2.annotate(
    "Super1 Outperforms Run 2\n(-0.0035 nats consistently)",
    xy=(tokens_b[-1], loss_super1[-1]),
    xytext=(1.55, 2.825),
    arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d62728", lw=1.5),
    fontsize=10,
    fontweight="bold"
)

plt.tight_layout()
save_fig(fig, "val_loss_convergence.png")

# ==============================================================================
# Plot 2: Relative Advantage Delta (Super1 vs Run 2 Reference)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5))
delta_loss = (loss_super1 - loss_run2_mem_cut_near) * 1000  # In millinats (10^-3 nats)

ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="Run 2 Baseline (0 mnat)")
ax.plot(tokens_b, delta_loss, "*-", color="#d62728", linewidth=2.5, markersize=8, label="Δ(Super1 − Run 2)")
ax.fill_between(tokens_b, delta_loss, 0, where=(delta_loss <= 0), color="#d62728", alpha=0.15, label="Super1 Outperformance Zone")

ax.set_title("Per-Step Advantage of Single-Layer Super-Memory over Run 2", pad=12)
ax.set_xlabel("Processed Tokens (Billions)")
ax.set_ylabel("Validation Loss Gap (millinats, mnat)")
ax.grid(True)
ax.legend(loc="lower right", framealpha=0.9)

# Annotate early stability vs late convergence
ax.annotate(
    "Early Stabilization:\nSuper1 beats Run 2 by -34 mnat\n(eliminates deep address drift)",
    xy=(tokens_b[1], delta_loss[1]),
    xytext=(0.4, -28),
    arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1f77b4", lw=1.2),
    fontsize=9.5
)

ax.annotate(
    "Final Advantage:\n-3.5 mnat sustained",
    xy=(tokens_b[-1], delta_loss[-1]),
    xytext=(1.5, -12),
    arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d62728", lw=1.2),
    fontsize=9.5
)

plt.tight_layout()
save_fig(fig, "super1_advantage_delta.png")

# ==============================================================================
# Plot 3: All-Model Performance & Parameter Efficiency Bar Chart
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

configs = [
    "Run 1\n(mem_full)",
    "Idea A\n(w2_rank1)",
    "Run 4\n(dense_full)",
    "Run 2\n(mem_cut_near)",
    "Run 3\n(mem_cut_far)",
    "Idea 1\n(w2_super1)",
    "Run 5\n(dense_cut)"
]
val_losses = [2.805, 2.807, 2.802, 2.791, 2.790, 2.7875, 2.774]
active_params = [269.5, 269.5, 370.5, 269.5, 269.5, 267.3, 370.5]  # In Millions

colors = ["#8c564b", "#9467bd", "#7f7f7f", "#1f77b4", "#17becf", "#d62728", "#2ca02c"]

# Left bar plot: Final Validation Loss
bars1 = ax1.bar(configs, val_losses, color=colors, width=0.6, edgecolor="black", linewidth=1.1)
ax1.set_ylim(2.75, 2.82)
ax1.set_title("(A) Final Validation Loss at 2.0B Tokens", pad=12)
ax1.set_ylabel("Validation Loss (Nats)")
ax1.grid(axis="y")

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.0008, f"{yval:.4f}" if yval < 2.79 else f"{yval:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Right bar plot: Active Parameters vs Memory Efficiency
bars2 = ax2.bar(configs, active_params, color=colors, width=0.6, edgecolor="black", linewidth=1.1)
ax2.set_ylim(200, 410)
ax2.set_title("(B) Active Non-Embedding Parameters / Token", pad=12)
ax2.set_ylabel("Active Parameters (Millions)")
ax2.grid(axis="y")

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 3.0, f"{yval:.1f}M", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax2.axhline(267.3, color="#d62728", linestyle=":", linewidth=1.5, label="Super1: 27.8% Fewer Active Params vs Dense")
ax2.legend(loc="upper left")

plt.tight_layout()
save_fig(fig, "final_performance_comparison.png")

# ==============================================================================
# Plot 4: Super-Memory Diagnostic Evolution (Sink Mass & Effective Codes)
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))

color_sink = "#d62728"
color_codes = "#1f77b4"

ax1.set_xlabel("Processed Tokens (Billions)")
ax1.set_ylabel("Sink Mass (Inactive Probability)", color=color_sink)
line1 = ax1.plot(tokens_b, super1_sink, "o-", color=color_sink, label="Sink Mass μ (starts 0.50, stabilizes ~0.415)")
ax1.tick_params(axis="y", labelcolor=color_sink)
ax1.set_ylim(0.35, 0.52)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel("Effective Routing Codes (out of 16)", color=color_codes)
line2 = ax2.plot(tokens_b, super1_codes, "s--", color=color_codes, label="Mean Effective Codes / Route (~14.96 / 16)")
ax2.tick_params(axis="y", labelcolor=color_codes)
ax2.set_ylim(12.0, 16.0)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", framealpha=0.9)

plt.title("Idea 1 (w2_mem_super1): Layer 1 Super-Memory Diagnostics Across Training", pad=12)
plt.tight_layout()
save_fig(fig, "super1_diagnostics.png")

print("\nAll 4 publication-quality figures successfully generated!")

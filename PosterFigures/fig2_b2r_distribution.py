"""
Figure 2: Distribution of BFP→mCherry delay (blue-to-red) for productive cells.
Fast (early) population shown in orange, Slow (med+late) in purple.
Cutoff from GMM classification (~1087 min).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

# ── data ──────────────────────────────────────────────────────────────────────
pred = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/results/cls_en_predictions.csv"
)
# y_true: 1 = early/fast, 0 = med+late/slow
CUTOFF_MIN  = (pred.loc[pred["y_true"] == 1, "delay_b2r"].max() +
               pred.loc[pred["y_true"] == 0, "delay_b2r"].min()) / 2   # ~1087 min
delays_min  = pred["delay_b2r"].values
fast_min    = pred.loc[pred["y_true"] == 1, "delay_b2r"].values
slow_min    = pred.loc[pred["y_true"] == 0, "delay_b2r"].values

n_fast = len(fast_min)
n_slow = len(slow_min)
n_total = len(delays_min)

# ── KDE ───────────────────────────────────────────────────────────────────────
x = np.linspace(delays_min.min() - 60, delays_min.max() + 60, 2000)
kde = gaussian_kde(delays_min, bw_method=0.18)
y = kde(x)

# Split fill at cutoff
x_fast = x[x <= CUTOFF_MIN]
x_slow = x[x >= CUTOFF_MIN]
y_fast = y[x <= CUTOFF_MIN]
y_slow = y[x >= CUTOFF_MIN]

# ── colors ────────────────────────────────────────────────────────────────────
C_FAST = "#FF8C00"   # vivid orange
C_SLOW = "#6A0DAD"   # deep purple

# ── figure ────────────────────────────────────────────────────────────────────
FONTSIZE = 32

fig, ax = plt.subplots(figsize=(14, 8))

# Filled KDE regions
ax.fill_between(x_fast, y_fast, alpha=0.7, color=C_FAST, label=f"Fast  (n={n_fast})")
ax.fill_between(x_slow, y_slow, alpha=0.7, color=C_SLOW, label=f"Slow  (n={n_slow})")

# KDE outline
ax.plot(x, y, color="black", linewidth=1.5, zorder=3)

# Cutoff line
ax.axvline(CUTOFF_MIN, color="black", linewidth=1.5, linestyle="--", zorder=4)

# ── labels ────────────────────────────────────────────────────────────────────
ax.set_xlabel("Blue → Red delay (minutes)", fontsize=FONTSIZE)
ax.set_ylabel("Density", fontsize=FONTSIZE, labelpad=20)
ax.tick_params(axis="both", labelsize=FONTSIZE - 4)

ax.legend(fontsize=FONTSIZE - 2, frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(script_dir, "fig2_b2r_distribution.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(script_dir, "fig2_b2r_distribution.png"), dpi=150, bbox_inches="tight")
print(f"Saved. Cutoff = {CUTOFF_MIN:.1f} min | n_fast={n_fast} n_slow={n_slow}")

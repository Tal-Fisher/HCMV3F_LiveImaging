"""
Figure 3: Bar plot of Spearman r for regression (Blue→Red delay prediction).
Three models: GLM, XGBoost, TabICL.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── data ──────────────────────────────────────────────────────────────────────
MODELS   = ["GLM",    "XGBoost", "TabICL"]
RHO      = [0.2798,    0.3465,    0.3199]

# Consistent model colors (used across all poster figures)
COLORS = {
    "GLM":     "#1f77b4",   # blue
    "XGBoost": "#ff7f0e",   # orange
    "TabICL":  "#2ca02c",   # green
}

# ── figure ────────────────────────────────────────────────────────────────────
FONTSIZE = 34

fig, ax = plt.subplots(figsize=(10, 8))

bars = ax.bar(
    MODELS,
    RHO,
    color=[COLORS[m] for m in MODELS],
    width=0.55,
    edgecolor="white",
    linewidth=0,
)

# Value labels on top of each bar
for bar, val in zip(bars, RHO):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.006,
        f"{val:.2f}",
        ha="center", va="bottom",
        fontsize=FONTSIZE - 2,
        fontweight="bold",
    )

ax.set_ylabel("Spearman  r", fontsize=FONTSIZE, labelpad=16)
ax.set_ylim(0, max(RHO) * 1.25)
ax.tick_params(axis="x", labelsize=FONTSIZE)
ax.tick_params(axis="y", labelsize=FONTSIZE - 4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="0.88", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(script_dir, "fig3_regression_spearman.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(script_dir, "fig3_regression_spearman.png"), dpi=150, bbox_inches="tight")
print("Saved fig3.")

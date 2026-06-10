"""
Figure 5 HMF: Classification AUC comparison — half-movie filter applied (n=497).
Group 1 — 16-frame tabular features: Baseline previous best (TabICL, HMF-filtered OOF).
Group 2 — Onset-frame embeddings (GLM): GFP top-20, BF top-20, GFP+BF top-40 (HMF run).
A2+A3 dataset.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── data ──────────────────────────────────────────────────────────────────────
# Group 1: 16-frame tabular — only best model retained
TAB_LABELS = ["Baseline\nprevious best"]
TAB_AUC    = [0.6685]
TAB_COLORS = ["#2ca02c"]

# Group 2: onset-frame embeddings, GLM — HMF run (a2a3_combined_hmf_metrics.csv, n=497)
EMB_LABELS = ["GFP", "BF", "GFP+BF"]
EMB_AUC    = [0.635,  0.675, 0.710]
EMB_COLORS = ["#d62728", "#7f7f7f", "#9467bd"]

# ── layout ────────────────────────────────────────────────────────────────────
FONTSIZE = 46
GAP      = 0.4    # small gap between groups so bars sit next to each other
BAR_W    = 0.6

n1, n2 = len(TAB_LABELS), len(EMB_LABELS)
x1 = np.arange(n1, dtype=float)
x2 = np.arange(n2, dtype=float) + n1 + GAP   # shift right by group size + gap

fig, ax = plt.subplots(figsize=(22, 10))

# Group 1 bars
for x, val, color in zip(x1, TAB_AUC, TAB_COLORS):
    bar = ax.bar(x, val, width=BAR_W, color=color, linewidth=0)
    ax.text(x, val + 0.005, f"{val:.3f}", ha="center", va="bottom",
            fontsize=FONTSIZE - 4, fontweight="bold")

# Group 2 bars
for x, val, color in zip(x2, EMB_AUC, EMB_COLORS):
    bar = ax.bar(x, val, width=BAR_W, color=color, linewidth=0)
    ax.text(x, val + 0.005, f"{val:.3f}", ha="center", va="bottom",
            fontsize=FONTSIZE - 4, fontweight="bold")

# ── x-axis labels ─────────────────────────────────────────────────────────────
all_x      = np.concatenate([x1, x2])
all_labels = TAB_LABELS + EMB_LABELS
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=FONTSIZE)

# ── group bracket annotations ─────────────────────────────────────────────────
y_bracket = max(TAB_AUC + EMB_AUC) * 1.18
y_text    = y_bracket + 0.012

def draw_bracket(ax, x_left, x_right, y, color="0.3"):
    ax.annotate("", xy=(x_right + BAR_W/2, y), xytext=(x_left - BAR_W/2, y),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.5))
    ax.plot([x_left - BAR_W/2, x_left - BAR_W/2], [y - 0.008, y], color=color, lw=1.5)
    ax.plot([x_right + BAR_W/2, x_right + BAR_W/2], [y - 0.008, y], color=color, lw=1.5)

draw_bracket(ax, x1[0], x1[-1], y_bracket)
draw_bracket(ax, x2[0], x2[-1], y_bracket)

ax.text((x1[0] + x1[-1]) / 2, y_text, "16-frame\nhandcrafted features",
        ha="center", va="bottom", fontsize=FONTSIZE - 6, color="0.2")
ax.text((x2[0] + x2[-1]) / 2, y_text, "Onset-frame\nembeddings (GLM)",
        ha="center", va="bottom", fontsize=FONTSIZE - 6, color="0.2")

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_ylabel("AUC", fontsize=FONTSIZE, labelpad=16)
ax.set_ylim(0, y_text + 0.06)
ax.tick_params(axis="y", labelsize=FONTSIZE - 4)
ax.axhline(0.5, color="0.6", linewidth=1.2, linestyle="--", zorder=0)
ax.yaxis.grid(True, color="0.88", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(script_dir, "fig5_auc_comparison_hmf.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(script_dir, "fig5_auc_comparison_hmf.png"), dpi=150, bbox_inches="tight")
print("Saved fig5 HMF.")

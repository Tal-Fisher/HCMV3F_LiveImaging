"""
onset_time_by_group.py

Shows the distribution of GFP onset time (= when in the movie each cell was
infected) for early, medium, and late GMM groups.  Tests whether early-
responding cells are infected later in the movie.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from pathlib import Path

BASE = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT  = Path("/home/labs/ginossar/talfis/LiveImaging/proximity_analysis")

# ── load ──────────────────────────────────────────────────────────────────────
model = pd.read_csv(BASE / "cache" / "python_export" / "model_df.csv")
model = model.rename(columns={"Track.ID": "Track_ID"})
meta  = pd.read_csv(BASE / "Forecast" / "cell_metadata.csv",
                    usecols=["Track.ID", "group"]).rename(columns={"Track.ID": "Track_ID"})

delay = model["delay_green_to_red"].values.astype(float)
prod  = model[np.isfinite(delay)].copy()
df    = prod.merge(meta, on="Track_ID", how="left")
df    = df[df["group"].notna()].copy()
df["onset_h"] = df["abs_gfp_onset_min"] / 60.0

GROUPS     = ["early", "medium", "late"]
GROUP_COLS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

# ── summary stats ─────────────────────────────────────────────────────────────
print("GFP onset time (h) per group:")
for g in GROUPS:
    v = df[df["group"] == g]["onset_h"]
    print(f"  {g:8s}  n={len(v):4d}  median={v.median():.1f} h  "
          f"mean={v.mean():.1f} h  IQR={v.quantile(0.25):.1f}–{v.quantile(0.75):.1f} h")

# Kruskal-Wallis across all three groups
vals = [df[df["group"] == g]["onset_h"].values for g in GROUPS]
H, p_kw = kruskal(*vals)
print(f"\nKruskal-Wallis H={H:.2f}  p={p_kw:.4f}")

# pairwise Mann-Whitney
pairs = [("early","medium"), ("early","late"), ("medium","late")]
print("\nMann-Whitney pairwise:")
mw_results = {}
for g1, g2 in pairs:
    u, p = mannwhitneyu(df[df["group"]==g1]["onset_h"],
                        df[df["group"]==g2]["onset_h"],
                        alternative="two-sided")
    print(f"  {g1} vs {g2}:  p={p:.4f}")
    mw_results[(g1, g2)] = p

# ── figure: 3 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("GFP onset time (= time of infection in movie) by GMM group\n"
             "Tests whether early-responding cells are infected later in the movie",
             fontsize=11, fontweight="bold")

all_onset = df["onset_h"].values
x_min, x_max = all_onset.min(), all_onset.max()

# Panel 1: overlaid KDE
ax = axes[0]
for g in GROUPS:
    v = df[df["group"] == g]["onset_h"].values
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(v, bw_method=0.3)
    xs  = np.linspace(x_min, x_max, 400)
    ax.plot(xs, kde(xs), color=GROUP_COLS[g], lw=2.2, label=f"{g} (n={len(v)})")
    ax.axvline(np.median(v), color=GROUP_COLS[g], lw=1.2, linestyle="--", alpha=0.7)
ax.set_xlabel("GFP onset time (h into movie)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Onset time distribution\n(dashed = median)", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Panel 2: boxplot
ax = axes[1]
bp = ax.boxplot([df[df["group"] == g]["onset_h"].values for g in GROUPS],
                patch_artist=True, notch=True,
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker="o", markersize=2, alpha=0.4))
for patch, g in zip(bp["boxes"], GROUPS):
    patch.set_facecolor(GROUP_COLS[g])
    patch.set_alpha(0.8)
for flier, g in zip(bp["fliers"], GROUPS):
    flier.set_markerfacecolor(GROUP_COLS[g])

# significance brackets
y_max = df["onset_h"].max() * 1.05
bracket_h = y_max * 0.04
for xi, (g1, g2) in enumerate(pairs):
    p = mw_results[(g1, g2)]
    if p >= 0.05:
        continue
    star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
    x1 = GROUPS.index(g1) + 1
    x2 = GROUPS.index(g2) + 1
    y  = y_max + xi * bracket_h * 1.8
    ax.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y],
            lw=1, color="black")
    ax.text((x1 + x2) / 2, y + bracket_h, star, ha="center", fontsize=9)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(GROUPS, fontsize=9)
ax.set_ylabel("GFP onset time (h into movie)", fontsize=9)
ax.set_title(f"Boxplot\nKruskal-Wallis p={p_kw:.4f}", fontsize=9)
ax.tick_params(labelsize=8)

# Panel 3: cumulative distribution
ax = axes[2]
for g in GROUPS:
    v = np.sort(df[df["group"] == g]["onset_h"].values)
    ax.plot(v, np.linspace(0, 1, len(v)),
            color=GROUP_COLS[g], lw=2.2, label=g)
ax.set_xlabel("GFP onset time (h into movie)", fontsize=9)
ax.set_ylabel("Cumulative fraction", fontsize=9)
ax.set_title("Cumulative distribution", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.axhline(0.5, color="grey", lw=0.8, linestyle=":")

plt.tight_layout()
out = OUT / "figures" / "onset_time_by_group.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

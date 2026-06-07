"""
proximity_values_by_bin_group.py

Shows actual proximity values (mean ± SEM) per onset-time bin × GMM group,
rather than correlations. Two panels:
  - Distance to nearest infected neighbour (px)
  - Number of infected cells within 100 px
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem

BASE = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT  = Path("/home/labs/ginossar/talfis/LiveImaging/proximity_analysis")

# ── load data ─────────────────────────────────────────────────────────────────
model = pd.read_csv(BASE / "cache" / "python_export" / "model_df.csv")
model = model.rename(columns={"Track.ID": "Track_ID"})
prox  = pd.read_csv(OUT / "results" / "proximity_features.csv")
meta  = pd.read_csv(BASE / "Forecast" / "cell_metadata.csv",
                    usecols=["Track.ID", "group"]).rename(columns={"Track.ID": "Track_ID"})

delay = model["delay_green_to_red"].values.astype(float)
prod  = model[np.isfinite(delay)].copy()
prod["delay_min"] = prod["delay_green_to_red"].astype(float)

df = prod.merge(
    prox[["Track_ID", "dist_nearest", "n_within_100"]],
    on="Track_ID", how="left")
df = df[df["dist_nearest"].notna()].copy()
df = df.merge(meta, on="Track_ID", how="left")
df["onset_h"] = df["abs_gfp_onset_min"] / 60.0

# ── onset-time quartile bins ──────────────────────────────────────────────────
df["bin"] = pd.qcut(df["onset_h"], q=4, labels=False)
bin_edges  = pd.qcut(df["onset_h"], q=4).cat.categories
bin_labels = [f"Q{i+1}  {iv.left:.0f}–{iv.right:.0f} h"
              for i, iv in enumerate(bin_edges)]

GROUPS     = ["early", "medium", "late"]
GROUP_COLS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

FEATS = {
    "dist_nearest": "Distance to nearest\ninfected neighbour (px)",
    "n_within_100": "# infected cells\nwithin 100 px",
}

# ── compute mean ± SEM per bin × group ────────────────────────────────────────
def mean_sem(vals):
    v = vals[np.isfinite(vals)]
    if len(v) < 2:
        return np.nan, np.nan, 0
    return float(np.mean(v)), float(sem(v)), len(v)

stats = {}
for feat in FEATS:
    rows = []
    for b in range(4):
        for g in GROUPS:
            sub = df[(df["bin"] == b) & (df["group"] == g)][feat].values.astype(float)
            m, s, n = mean_sem(sub)
            rows.append(dict(bin=b, group=g, mean=m, sem=s, n=n))
    stats[feat] = pd.DataFrame(rows)

# ── plot ───────────────────────────────────────────────────────────────────────
bar_w   = 0.22
offsets = np.array([-1, 0, 1]) * bar_w

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Proximity values by onset-time bin and timing group\n"
    "mean ± SEM;  each bin = cells infected at roughly the same point in the movie",
    fontsize=11, fontweight="bold"
)

for ax, (feat, ylabel) in zip(axes, FEATS.items()):
    st = stats[feat]
    for gi, g in enumerate(GROUPS):
        sub   = st[st["group"] == g]
        means = sub["mean"].values
        sems  = sub["sem"].values
        ns    = sub["n"].values
        x_pos = np.arange(4) + offsets[gi]

        ax.bar(x_pos, means, width=bar_w * 0.9,
               color=GROUP_COLS[g], edgecolor="white", linewidth=0.6,
               label=g, alpha=0.85, yerr=sems, capsize=3,
               error_kw=dict(elinewidth=1, ecolor="grey"))

        for i, (m, n) in enumerate(zip(means, ns)):
            if not np.isfinite(m) or n < 2:
                continue
            ax.text(x_pos[i], -ax.get_ylim()[1] * 0.06 if ax.get_ylim()[1] > 0 else 0,
                    str(n), ha="center", fontsize=6.5, color=GROUP_COLS[g])

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(bin_labels, fontsize=8.5)
    ax.set_xlabel("Onset-time bin (quartile of abs_gfp_onset_min)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8.5)
    ax.tick_params(labelsize=8)

# fix n-label y position after ylims are set
for ax, (feat, _) in zip(axes, FEATS.items()):
    st = stats[feat]
    ymin = ax.get_ylim()[0]
    yoff = ymin + (ax.get_ylim()[1] - ymin) * 0.02
    for gi, g in enumerate(GROUPS):
        sub   = st[st["group"] == g]
        ns    = sub["n"].values
        x_pos = np.arange(4) + offsets[gi]
        for i, n in enumerate(ns):
            if n >= 2:
                ax.text(x_pos[i], yoff, str(n), ha="center",
                        fontsize=6, color=GROUP_COLS[g], va="bottom")

plt.tight_layout()
out = OUT / "figures" / "proximity_values_by_bin_group.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

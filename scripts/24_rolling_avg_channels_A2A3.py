"""
24_rolling_avg_channels_A2A3.py

Rolling mean ± SEM of GFP (ch2_corrected) and BFP (Mean.ch1) signals
vs. frames since first detection, split by early vs medium+late (A2+A3).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 10
WIN       = 5

# ── load data ──────────────────────────────────────────────────────────────────
ts  = pd.read_csv(EXPORT_DIR / "timeseries_data.csv", low_memory=False)
cat = pd.read_csv(EXPORT_DIR / "category_df.csv")
cat = cat[cat["productive"] == True].dropna(subset=["category"])
cat["group"] = cat["category"].apply(
    lambda c: "early" if c == "early" else ("med+late" if c in ["medium", "late"] else None)
)
cat = cat[cat["group"].notna()]

ts = ts.sort_values(["Track.ID", "T_min"]).copy()
ts["norm_frame"] = ts.groupby("Track.ID")["Frame"].transform(lambda f: f - f.min())
ts = ts[ts["Track.ID"].isin(cat["Track.ID"])]
ts = ts.merge(cat[["Track.ID", "group"]], on="Track.ID", how="left")

n_early = ts[ts["group"] == "early"]["Track.ID"].nunique()
n_rest  = ts[ts["group"] == "med+late"]["Track.ID"].nunique()
print(f"Early: {n_early}  |  Med+late: {n_rest}")

# ── rolling stats helper ───────────────────────────────────────────────────────
def pop_stats(df, feat):
    g = df.groupby("norm_frame")[feat].agg(
        mean=lambda x: x.mean(skipna=True),
        sem=lambda x: (x.std(skipna=True) / np.sqrt(x.notna().sum()))
                      if x.notna().sum() > 1 else np.nan,
        n=lambda x: x.notna().sum()
    ).reset_index()
    g = g[g["n"] >= MIN_CELLS]
    g["mean_sm"] = g["mean"].rolling(WIN, center=True, min_periods=1).mean()
    g["sem_sm"]  = g["sem"].rolling(WIN,  center=True, min_periods=1).mean()
    return g

# ── build stats ────────────────────────────────────────────────────────────────
features = [
    ("ch2_corrected", "GFP corrected (ch2)"),
    ("Mean.ch1",      "BFP nuclear (ch1)"),
]

colors = {"early": "#E8491D", "med+late": "#3B82C4"}
labels = {"early": f"Fast  (n={n_early})",
          "med+late": f"Slow  (n={n_rest})"}

stats = {}
for feat, _ in features:
    stats[feat] = {
        grp: pop_stats(ts[ts["group"] == grp], feat)
        for grp in ["early", "med+late"]
    }

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "A2+A3: GFP and BFP signals — rolling mean ± SEM by frames since first detection\n"
    "Early vs Medium+Late  (min 10 cells/bin, 5-frame smoothing)",
    fontsize=13, fontweight="bold"
)

for ax, (feat, ylabel) in zip(axes, features):
    for grp in ["early", "med+late"]:
        d   = stats[feat][grp]
        col = colors[grp]
        ax.fill_between(d["norm_frame"],
                        d["mean_sm"] - d["sem_sm"],
                        d["mean_sm"] + d["sem_sm"],
                        color=col, alpha=0.20)
        ax.plot(d["norm_frame"], d["mean_sm"],
                color=col, lw=2, label=labels[grp])

    ax.set_xlabel("Frames from first detection", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(left=0)

plt.tight_layout()
out = FIG_DIR / "rolling_avg_channels_A2A3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

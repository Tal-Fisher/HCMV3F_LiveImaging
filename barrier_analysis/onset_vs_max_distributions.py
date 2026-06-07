"""
onset_vs_max_distributions.py

Productive cells:     GFP and nucleus BFP at the frame of mCherry onset
Non-productive cells: peak (max) GFP and nucleus BFP over the full track from GFP onset

Overlaid KDE distributions reveal whether productive cells reach similar
feature levels at the moment of commitment to those non-productive cells
ever achieve over their entire lifetime.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu
from pathlib import Path

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
TS_CSV  = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"
OUT_DIR = BASE / "barrier_analysis"

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
ts = pd.read_csv(TS_CSV, low_memory=False)
md = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red"]]
ts = ts.merge(md, on="Track.ID", how="left")

ts["t_rel_min"]  = ts["T_min"] - ts["abs_gfp_onset_min"]
ts["t_to_red_min"] = ts["red_onset_min"] - ts["t_rel_min"]
ts["productive"] = ts["delay_green_to_red"].notna() & np.isfinite(ts["delay_green_to_red"])

# ── productive: value at mCherry onset ────────────────────────────────────────
# For each productive cell take the frame where t_to_red_min is smallest
# and non-negative (= last clean frame before / at red onset).
prod_ts = ts[ts["productive"] & (ts["t_rel_min"] >= 0)].copy()

onset_rows = (
    prod_ts[prod_ts["t_to_red_min"] >= 0]
    .sort_values("t_to_red_min")
    .groupby("Track.ID")
    .first()          # minimum t_to_red_min per cell = onset frame
    .reset_index()
)

prod_gfp = onset_rows["ch2_corrected"].dropna()
prod_nuc = onset_rows["Mean.ch1_nuc"].dropna()

# ── non-productive: peak over full track from GFP onset ───────────────────────
np_ts = ts[~ts["productive"] & (ts["t_rel_min"] >= 0)].copy()
np_peaks = np_ts.groupby("Track.ID").agg(
    gfp_max=("ch2_corrected", "max"),
    nuc_max=("Mean.ch1_nuc",  "max"),
).reset_index()

np_gfp = np_peaks["gfp_max"].dropna()
np_nuc = np_peaks["nuc_max"].dropna()

print(f"  Productive cells (at onset): {len(prod_gfp)} GFP, {len(prod_nuc)} nucleus", flush=True)
print(f"  Non-productive cells (peak): {len(np_gfp)} GFP, {len(np_nuc)} nucleus", flush=True)

# ── Wilcoxon one-sided p-values for annotation ────────────────────────────────
stat_gfp, p_gfp = mannwhitneyu(prod_gfp.values, np_gfp.values, alternative="greater")
stat_nuc, p_nuc = mannwhitneyu(prod_nuc.values, np_nuc.values, alternative="greater")
auc_gfp = stat_gfp / (len(prod_gfp) * len(np_gfp))
auc_nuc = stat_nuc / (len(prod_nuc) * len(np_nuc))


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── plot ──────────────────────────────────────────────────────────────────────
PROD_COLOR = "#e67e22"   # orange  – productive
NP_COLOR   = "#2980b9"   # blue    – non-productive

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def plot_panel(ax, prod_vals, np_vals, xlabel, title, p_val, auc_val):
    for vals, label, color in [
        (prod_vals, f"Productive – at mCherry onset  (n={len(prod_vals)})", PROD_COLOR),
        (np_vals,   f"Non-productive – peak over track  (n={len(np_vals)})", NP_COLOR),
    ]:
        v = vals.values
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        x = np.linspace(lo, hi, 400)
        kde = gaussian_kde(v, bw_method=0.25)
        ax.fill_between(x, kde(x), alpha=0.30, color=color)
        ax.plot(x, kde(x), color=color, lw=2.0, label=label)
        ax.axvline(np.median(v), color=color, lw=1.2, linestyle="--", alpha=0.8)

    # annotation
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sig = stars(p_val)
    annot = f"Wilcoxon (prod > non-prod): {sig}\np = {p_val:.3e}   AUC = {auc_val:.3f}"
    ax.text(0.03, 0.97, annot, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))


plot_panel(axes[0], prod_gfp, np_gfp,
           "GFP (corrected, ch2)",
           "GFP at mCherry onset vs peak GFP",
           p_gfp, auc_gfp)

plot_panel(axes[1], prod_nuc, np_nuc,
           "BFP nucleus (Mean.ch1_nuc)",
           "Nucleus BFP at mCherry onset vs peak nucleus BFP",
           p_nuc, auc_nuc)

fig.suptitle(
    "Productive cells at mCherry onset  vs  Non-productive cells (lifetime peak)\n"
    "Dashed lines = medians   |   One-sided Wilcoxon rank-sum test",
    fontsize=11, fontweight="bold", y=1.02,
)

plt.tight_layout()
out_path = OUT_DIR / "onset_vs_max_distributions.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved  →  {out_path}", flush=True)
print("Done.", flush=True)

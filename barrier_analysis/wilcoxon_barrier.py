"""
wilcoxon_barrier.py — Wilcoxon rank-sum test on feature extreme values

For each feature + direction (max/min), compare the extreme value over the
observation window between productive and non-productive cells:
  productive:     peak or trough over GFP onset → red onset  (pre-red window)
  non-productive: peak or trough over full track from GFP onset

AUC = P(productive cell is more extreme in the expected direction than
        a randomly chosen non-productive cell)
    → AUC > 0.5 means productive cells consistently reach higher peaks
      (max features) or lower troughs (min features).

The test is one-sided in the biologically expected direction.
Multiple-testing correction: Benjamini-Hochberg FDR across all features.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
TS_CSV  = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"
OUT_DIR = BASE / "barrier_analysis"

# same feature specs as barrier_analysis.py
FEAT_SPECS = [
    ("ch2_corrected", "GFP (corrected) ↑",       "max"),
    ("Mean.ch1",      "BFP cytoplasm ↑",          "max"),
    ("Mean.ch1_nuc",  "BFP nucleus ↑",            "max"),
    ("Area_cell",     "Cell area ↑",              "max"),
    ("P.stuck",       "P(stuck) ↑",               "max"),
    ("Area_nuc",      "Nucleus area ↓",           "min"),
    ("Area_nuc",      "Nucleus area ↑",           "max"),
    ("Circ_nuc",      "Nucleus circularity ↓",    "min"),
    ("Circ_nuc",      "Nucleus circularity ↑",    "max"),
    ("nuc_ratio",     "Nucleus/cell ratio ↓",     "min"),
    ("nuc_ratio",     "Nucleus/cell ratio ↑",     "max"),
    ("Solidity",      "Cell solidity ↓",          "min"),
    ("Solidity",      "Cell solidity ↑",          "max"),
    ("Ctrst.ch4",     "BF contrast ↓",            "min"),
    ("Ctrst.ch4",     "BF contrast ↑",            "max"),
    ("Shape_index",   "Shape index ↓",            "min"),
    ("gfp_bfp_ratio", "GFP/BFP ratio ↑",          "max"),
]


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = np.minimum(1.0, pvals * n / ranks)
    # enforce monotonicity right-to-left in sorted order
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_out = np.empty(n)
    q_out[order] = q_sorted
    return q_out


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
ts = pd.read_csv(TS_CSV, low_memory=False)
md = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red"]]
ts = ts.merge(md, on="Track.ID", how="left")

ts["t_rel_min"]    = ts["T_min"] - ts["abs_gfp_onset_min"]
ts["t_to_red_min"] = ts["red_onset_min"] - ts["t_rel_min"]
ts["productive"]   = ts["delay_green_to_red"].notna() & np.isfinite(ts["delay_green_to_red"])
ts["gfp_bfp_ratio"] = ts["ch2_corrected"] / ts["Mean.ch1_nuc"].replace(0, np.nan)

# productive: pre-red frames only (same window as barrier analysis)
# non-productive: all frames from GFP onset onward
prod_window = ts[ts["productive"] & (ts["t_to_red_min"] >= 0)]
np_window   = ts[~ts["productive"] & (ts["t_rel_min"] >= 0)]

n_prod = ts[ts["productive"]]["Track.ID"].nunique()
n_np   = ts[~ts["productive"]]["Track.ID"].nunique()
print(f"  {n_prod} productive  |  {n_np} non-productive", flush=True)


# ── Wilcoxon rank-sum for each spec ───────────────────────────────────────────
print("Running Wilcoxon tests ...", flush=True)

rows = []
for col, label, direction in FEAT_SPECS:
    agg_fn = "max" if direction == "max" else "min"

    prod_ext = prod_window.groupby("Track.ID")[col].agg(agg_fn).dropna()
    np_ext   = np_window.groupby("Track.ID")[col].agg(agg_fn).dropna()

    if len(prod_ext) < 5 or len(np_ext) < 5:
        rows.append({"label": label, "direction": direction, "col": col,
                     "n_prod": len(prod_ext), "n_nonprod": len(np_ext),
                     "AUC": np.nan, "p_value": np.nan})
        continue

    # flip sign for min features so "greater" always tests the expected direction
    x = prod_ext.values if direction == "max" else -prod_ext.values
    y = np_ext.values   if direction == "max" else -np_ext.values

    stat, pval = mannwhitneyu(x, y, alternative="greater")
    auc        = stat / (len(prod_ext) * len(np_ext))

    rows.append({"label": label, "direction": direction, "col": col,
                 "n_prod": len(prod_ext), "n_nonprod": len(np_ext),
                 "AUC": auc, "p_value": pval})

res = pd.DataFrame(rows)

# FDR correction on the non-NaN p-values
valid_mask = res["p_value"].notna()
q_vals     = np.full(len(res), np.nan)
q_vals[valid_mask.values] = bh_correct(res.loc[valid_mask, "p_value"].values)
res["q_value"] = q_vals

res_sorted = res.sort_values("AUC", ascending=False)
print(res_sorted[["label", "AUC", "p_value", "q_value"]].to_string(index=False), flush=True)


# ── figure ────────────────────────────────────────────────────────────────────
print("\nCreating figure ...", flush=True)

plot_df = res.dropna(subset=["AUC"]).sort_values("AUC", ascending=True).reset_index(drop=True)

dir_colors = {"max": "#e67e22", "min": "#2980b9"}
bar_colors = [dir_colors[d] for d in plot_df["direction"]]

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(range(len(plot_df)), plot_df["AUC"] - 0.5,
        left=0.5, color=bar_colors, alpha=0.82, height=0.68, edgecolor="white", lw=0.4)

# significance labels
for i, row in plot_df.iterrows():
    q = row["q_value"]
    if np.isnan(q):
        stars, star_col = "n/a", "#aaa"
    elif q < 0.001:
        stars, star_col = "***", "#c0392b"
    elif q < 0.01:
        stars, star_col = "** ", "#e74c3c"
    elif q < 0.05:
        stars, star_col = "*  ", "#e67e22"
    else:
        stars, star_col = "ns ", "#999999"

    auc_val = row["AUC"]
    offset  = 0.010
    ha      = "left" if auc_val >= 0.5 else "right"
    x_text  = auc_val + offset if auc_val >= 0.5 else auc_val - offset
    ax.text(x_text, i, f"{stars}  {auc_val:.3f}",
            va="center", ha=ha, fontsize=8, color=star_col)

ax.axvline(0.5, color="#333", lw=1.2, linestyle="--", zorder=3)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df["label"], fontsize=9)
ax.set_xlabel("AUC  [P(productive cell more extreme than non-productive cell)]", fontsize=9)

# dynamic x limits
all_auc = plot_df["AUC"].values
ax.set_xlim(min(0.35, all_auc.min() - 0.08), max(0.80, all_auc.max() + 0.14))

ax.set_title(
    "Wilcoxon rank-sum: productive vs non-productive — feature extreme values\n"
    "Productive: peak/trough over pre-red window  |  Non-productive: full track from GFP onset\n"
    "Significance (BH-FDR):  *** q<0.001   ** q<0.01   * q<0.05   ns not significant",
    fontsize=9.5, fontweight="bold", pad=10
)

legend_patches = [
    mpatches.Patch(facecolor="#e67e22", alpha=0.82, label="↑ max feature (peak compared)"),
    mpatches.Patch(facecolor="#2980b9", alpha=0.82, label="↓ min feature (trough compared)"),
]
ax.legend(handles=legend_patches, fontsize=8.5, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = OUT_DIR / "wilcoxon_barrier.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}", flush=True)
print("Done.", flush=True)

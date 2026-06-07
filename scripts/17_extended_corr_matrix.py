"""
17_extended_corr_matrix.py

Spearman correlation matrix for the extended feature set (45 features).
Matches the format of 12_feature_analysis.py but uses model_df_extended.csv
which includes the 16 new features added in 16_elasticnet_extended_features.R.

Features are median-imputed then z-normalised before plotting.
(Spearman correlation is rank-based so z-normalisation doesn't change values,
but keeps the feature space consistent with the elastic net model.)

New features highlighted with a border in the correlation matrix.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr, rankdata

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
DATA_DIR   = BASE / "results" / "elasticnet_extended"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CUT_EARLY_MED = 911
CUT_MED_LATE  = 2163
CAT_NAMES     = ["early", "medium", "late"]
CAT_COLORS    = ["#2ecc71", "#e67e22", "#e74c3c"]

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "model_df_extended.csv")
print(f"Loaded: {len(df)} cells")

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "gfp_snr_mean", "bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

# which features are new (not in original 29)
ORIG_FEATS = {
    "gfp_corr_start","gfp_corr_mean","gfp_corr_sd","gfp_corr_slope",
    "nuc_bfp_start","nuc_bfp_mean","nuc_bfp_sd","nuc_bfp_slope",
    "nuc_area_mean","nuc_area_slope","nuc_circ_mean","nuc_circ_sd",
    "nuc_ratio_mean","nuc_ratio_slope",
    "area_start","area_mean","area_sd","area_slope",
    "solidity_mean","solidity_sd","shape_idx_mean",
    "gfp_snr_sd","bf_ctrst_mean","bf_ctrst_sd",
    "gfp_ratio_start","gfp_ratio_mean","gfp_ratio_sd","gfp_ratio_slope","gfp_ratio_max",
}
is_new = [c not in ORIG_FEATS for c in feat_cols]
print(f"Total features: {len(feat_cols)}  (original: {sum(not n for n in is_new)}, "
      f"new: {sum(is_new)})")

# ── impute + z-normalise ───────────────────────────────────────────────────────
X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
X = StandardScaler().fit_transform(X_raw)
df_feat = pd.DataFrame(X, columns=feat_cols)
print(f"Imputed and z-normalised.")

# ── productive cells + group labels ───────────────────────────────────────────
delay      = df["delay_green_to_red"].values.astype(float)
productive = np.isfinite(delay)
df_prod    = df_feat[productive].copy()
delay_prod = delay[productive]

def to_cat(d):
    if d <= CUT_EARLY_MED: return "early"
    if d <= CUT_MED_LATE:  return "medium"
    return "late"

groups   = np.array([to_cat(d) for d in delay_prod])
n_groups = {n: (groups == n).sum() for n in CAT_NAMES}
print(f"Productive: {productive.sum()}  "
      + "  ".join(f"{n}={n_groups[n]}" for n in CAT_NAMES))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Spearman correlation matrix
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing correlation matrix...")
ranks    = np.apply_along_axis(rankdata, 0, df_feat[feat_cols].values.astype(float))
corr_mat = np.corrcoef(ranks.T)

# hierarchical clustering on dissimilarity = 1 - |rho|
dist_mat  = 1 - np.abs(corr_mat)
np.fill_diagonal(dist_mat, 0)
linkage_mat  = linkage(dist_mat[np.triu_indices(len(feat_cols), k=1)], method="average")
order        = leaves_list(linkage_mat)
corr_ordered = corr_mat[np.ix_(order, order)]
feats_ordered = [feat_cols[i] for i in order]

n = len(feats_ordered)
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_ordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(n))
ax.set_xticklabels(feats_ordered, rotation=90, fontsize=6.5)
ax.set_yticks(range(n))
ax.set_yticklabels(feats_ordered, fontsize=6.5)

plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Spearman ρ")
ax.set_title(
    f"Spearman correlation matrix — extended feature set ({n} features)\n"
    f"All cells, no late bloomers  (n={len(df_feat)})   "
    f"Hierarchical clustering (average linkage, 1−|ρ|)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out1 = FIG_DIR / "feature_correlation_matrix_extended.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")

# ── save correlation matrix as CSV ────────────────────────────────────────────
corr_df = pd.DataFrame(corr_mat.round(3), index=feat_cols, columns=feat_cols)
corr_df.to_csv(DATA_DIR / "feature_corr_matrix_extended.csv")
print(f"Saved feature_corr_matrix_extended.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Spearman(feature, delay) per group, extended feature set
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing feature–delay correlations...")

def feat_delay_corrs(feat_df, delay_vals, cols):
    rhos, pvals = [], []
    for col in cols:
        r, p = spearmanr(feat_df[col].values, delay_vals)
        rhos.append(r); pvals.append(p)
    return np.array(rhos), np.array(pvals)

rho_all, pval_all = feat_delay_corrs(df_prod, delay_prod, feat_cols)
panel_data = [("All productive", rho_all, pval_all, "steelblue", productive.sum())]
for name in CAT_NAMES:
    mask = groups == name
    r, p = feat_delay_corrs(df_prod[mask], delay_prod[mask], feat_cols)
    panel_data.append((f"{name.capitalize()}", r, p,
                       CAT_COLORS[CAT_NAMES.index(name)], n_groups[name]))

sort_idx     = np.argsort(np.abs(rho_all))[::-1]
sorted_feats = [feat_cols[i] for i in sort_idx]
sorted_new   = [is_new[i]    for i in sort_idx]

fig, axes = plt.subplots(1, 4, figsize=(24, 11), sharey=True)
fig.suptitle(
    "Spearman ρ(feature, GFP→mCherry delay) — extended feature set\n"
    f"GMM cutoffs: {CUT_EARLY_MED} / {CUT_MED_LATE} min   "
    "Red feature labels = new features   * = p < 0.05",
    fontsize=12, fontweight="bold"
)

for ax, (title, rhos, pvals, color, n) in zip(axes, panel_data):
    vals   = [rhos[i]  for i in sort_idx]
    pv     = [pvals[i] for i in sort_idx]
    colors = [color if p < 0.05 else "#cccccc" for p in pv]

    ypos = np.arange(len(sorted_feats))
    ax.barh(ypos, vals, color=colors, edgecolor="none", height=0.75)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xlabel("Spearman ρ", fontsize=9)
    ax.set_title(f"{title}\n(n={n})", fontsize=10, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels(sorted_feats, fontsize=6.5)

    # colour y-tick labels for new features
    for tick, new in zip(ax.get_yticklabels(), sorted_new):
        tick.set_color("#c0392b" if new else "black")

    for y, p, v in zip(ypos, pv, vals):
        if p < 0.05:
            ax.text(v + (0.02 if v >= 0 else -0.02), y, "*",
                    ha="left" if v >= 0 else "right", va="center",
                    fontsize=7, color="black")

sig_patch   = mpatches.Patch(color="steelblue", label="p < 0.05")
insig_patch = mpatches.Patch(color="#cccccc",   label="p ≥ 0.05")
fig.legend(handles=[sig_patch, insig_patch],
           loc="lower right", fontsize=9, framealpha=0.8)

plt.tight_layout()
out2 = FIG_DIR / "feature_delay_corr_extended.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")

print("\nDone.")

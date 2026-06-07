"""
12_feature_analysis.py — Feature correlation analysis

1. Spearman correlation matrix of all z-normalised features
2. Per-population bar plots: Spearman correlation of each feature with
   delay_green_to_red within early / medium / late groups
   (GMM-derived cutoffs: 911 min and 2163 min)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CUT_EARLY_MED = 911    # GMM boundary: early | medium (min)
CUT_MED_LATE  = 2163   # GMM boundary: medium | late  (min)
CAT_NAMES     = ["early", "medium", "late"]
CAT_COLORS    = ["#2ecc71", "#e67e22", "#e74c3c"]

# ── load & filter ──────────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)
print(f"Cells after first-half filter: {len(df)}")

NON_FEAT  = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
             "green_onset_min", "track_start_min", "abs_gfp_onset_min",
             "movie_half_min", "y",
             "gfp_snr_mean", "bf_snr_mean"}   # 100% missing — excluded
feat_cols = [c for c in df.columns if c not in NON_FEAT]

# ── median-impute then z-normalise (identical to XGBoost / TabPFN pipeline) ───
X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
df_feat = pd.DataFrame(X, columns=feat_cols)

print(f"Features: {len(feat_cols)}")

# ── productive cells & group labels ───────────────────────────────────────────
delay      = df["delay_green_to_red"].values.astype(float)
productive = np.isfinite(delay)
df_prod    = df_feat[productive].copy()
delay_prod = delay[productive]

def to_cat(d):
    if d <= CUT_EARLY_MED: return "early"
    if d <= CUT_MED_LATE:  return "medium"
    return "late"

groups = np.array([to_cat(d) for d in delay_prod])
n_early  = (groups == "early").sum()
n_medium = (groups == "medium").sum()
n_late   = (groups == "late").sum()
print(f"Early: {n_early}  Medium: {n_medium}  Late: {n_late}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Spearman correlation matrix of all features
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing feature correlation matrix...")

# Use all cells (not just productive) for the feature–feature correlation
# Spearman = Pearson on ranks — fast vectorised approach
from scipy.stats import rankdata
ranks = np.apply_along_axis(rankdata, 0, df_feat[feat_cols].values.astype(float))
corr_vals = np.corrcoef(ranks.T)
corr_mat  = pd.DataFrame(corr_vals, index=feat_cols, columns=feat_cols)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(feat_cols)))
ax.set_xticklabels(feat_cols, rotation=90, fontsize=7)
ax.set_yticks(range(len(feat_cols)))
ax.set_yticklabels(feat_cols, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman ρ")
ax.set_title(
    f"Spearman correlation matrix — z-normalised features\n"
    f"All cells after first-half filter  (n={len(df_feat)})",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out = FIG_DIR / "feature_correlation_matrix.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-population bar plots: Spearman(feature, delay) within group
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing per-group feature–delay correlations...")

# Also compute for the full productive population as reference
def feat_delay_corrs(feat_df, delay_vals):
    rhos, pvals = [], []
    for col in feat_cols:
        r, p = spearmanr(feat_df[col].values, delay_vals)
        rhos.append(r)
        pvals.append(p)
    return np.array(rhos), np.array(pvals)

rho_all,  pval_all  = feat_delay_corrs(df_prod, delay_prod)

group_rhos  = {}
group_pvals = {}
for name in CAT_NAMES:
    mask = groups == name
    r, p = feat_delay_corrs(df_prod[mask], delay_prod[mask])
    group_rhos[name]  = r
    group_pvals[name] = p

# Sort features by overall (full productive) Spearman correlation magnitude
sort_idx = np.argsort(np.abs(rho_all))[::-1]
sorted_feats = [feat_cols[i] for i in sort_idx]

# ── panel layout: 1 column per population + 1 for full productive ─────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 9), sharey=True)
fig.suptitle(
    "Spearman correlation of each feature with green→red delay\n"
    f"within each timing group  (GMM cutoffs: {CUT_EARLY_MED} / {CUT_MED_LATE} min)",
    fontsize=13, fontweight="bold"
)

panel_data = [
    ("All productive",  rho_all,             pval_all,             "steelblue",    len(delay_prod)),
    ("Early",           group_rhos["early"],  group_pvals["early"],  CAT_COLORS[0], n_early),
    ("Medium",          group_rhos["medium"], group_pvals["medium"], CAT_COLORS[1], n_medium),
    ("Late",            group_rhos["late"],   group_pvals["late"],   CAT_COLORS[2], n_late),
]

SIG_ALPHA = 0.05  # mark significant bars with a star

for ax, (title, rhos, pvals, color, n) in zip(axes, panel_data):
    vals   = [rhos[i] for i in sort_idx]
    pv     = [pvals[i] for i in sort_idx]
    colors = [color if p < SIG_ALPHA else "#cccccc" for p in pv]

    y_pos = range(len(sorted_feats))
    ax.barh(y_pos, vals, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ", fontsize=9)
    ax.set_title(f"{title}\n(n={n})", fontsize=10, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats, fontsize=7)
    ax.tick_params(axis="x", labelsize=8)

    # star for significant
    for y, p, v in zip(y_pos, pv, vals):
        if p < SIG_ALPHA:
            ha = "left" if v >= 0 else "right"
            ax.text(v + (0.02 if v >= 0 else -0.02), y, "*",
                    ha=ha, va="center", fontsize=8, color="black")

# legend
sig_patch   = mpatches.Patch(color="steelblue", label="p < 0.05")
insig_patch = mpatches.Patch(color="#cccccc",   label="p ≥ 0.05")
fig.legend(handles=[sig_patch, insig_patch], loc="lower right",
           fontsize=9, framealpha=0.8)

plt.tight_layout()
out2 = FIG_DIR / "feature_delay_corr_by_group.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Between-group feature distributions (overlaid KDE, multi-page)
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting between-group feature distributions...")

from scipy.stats import gaussian_kde, kruskal
from matplotlib.backends.backend_pdf import PdfPages

FEATS_PER_PAGE = 6   # 2 rows × 3 cols per page
n_pages = int(np.ceil(len(feat_cols) / FEATS_PER_PAGE))

# data per group (use unscaled raw values so axes are interpretable)
X_raw_prod = X_raw[productive]
group_masks = {name: groups == name for name in CAT_NAMES}

pdf_path = FIG_DIR / "feature_distributions_by_group.pdf"
with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        feats_page = feat_cols[page * FEATS_PER_PAGE : (page + 1) * FEATS_PER_PAGE]
        n_feats    = len(feats_page)
        fig, axes  = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat  = axes.flatten()

        fig.suptitle(
            f"Feature distributions by timing group  "
            f"(GMM cutoffs: {CUT_EARLY_MED} / {CUT_MED_LATE} min) — page {page+1}/{n_pages}",
            fontsize=11, fontweight="bold"
        )

        for i, feat in enumerate(feats_page):
            ax   = axes_flat[i]
            fidx = feat_cols.index(feat)
            vals_all = X_raw_prod[:, fidx]

            # Kruskal-Wallis test across groups
            group_vals = [vals_all[group_masks[name]] for name in CAT_NAMES]
            try:
                _, kw_p = kruskal(*group_vals)
                p_str = f"KW p={kw_p:.3f}" if kw_p >= 0.001 else "KW p<0.001"
            except Exception:
                p_str = ""

            x_lo = np.nanpercentile(vals_all, 1)
            x_hi = np.nanpercentile(vals_all, 99)
            x_grid = np.linspace(x_lo, x_hi, 300)

            for name, color in zip(CAT_NAMES, CAT_COLORS):
                v = vals_all[group_masks[name]]
                v = v[np.isfinite(v)]
                if len(v) < 5:
                    continue
                try:
                    kde = gaussian_kde(v, bw_method="scott")
                    ax.plot(x_grid, kde(x_grid), color=color, lw=1.8,
                            label=f"{name} (n={len(v)})")
                    ax.axvline(np.median(v), color=color, lw=0.8, linestyle="--", alpha=0.7)
                except Exception:
                    pass

            ax.set_title(f"{feat}\n{p_str}", fontsize=8, pad=3)
            ax.set_xlabel("raw value", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right")

        # hide unused subplots
        for j in range(n_feats, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print(f"Saved {pdf_path}  ({n_pages} pages)")

print("\nDone.")

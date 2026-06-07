"""
06_regression_by_population.py
ElasticNet regression within each GMM population (early / medium / late)
separately.  Same 45-feature set and productive-only filter as 01_regression_en_xgb.py.
No half-movie filter.  5-fold CV within each subgroup.
"""

import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, norm as sp_norm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
EXT_DIR    = BASE / "results" / "elasticnet_extended2"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
FIG_DIR    = ANA_DIR / "figures"
SEED = 42

# ── data — 45-feature extended set (same as 01) ───────────────────────────────
ext_df  = pd.read_csv(EXT_DIR / "model_df_extended2.csv")
filt_df = pd.read_csv(EXPORT_DIR / "model_df.csv")[
    ["Track.ID", "abs_gfp_onset_min", "movie_half_min"]]
df = ext_df.merge(filt_df, on="Track.ID", how="left")

META_COLS = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
             "abs_gfp_onset_min","movie_half_min"}
EXTRAS_18 = {"cell_aspect_start","cell_aspect_mean","bfp_nuc_frac_start",
             "nuc_ratio_start","nuc_ratio_end",
             "bf_ctrst_start","bf_ctrst_end","bf_ctrst_slope"}
feat_cols = [c for c in ext_df.columns if c not in META_COLS and c not in EXTRAS_18]

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]
mask = np.isfinite(df["delay_blue_to_red"].values.astype(float))
df   = df[mask].reset_index(drop=True)

y     = df["delay_blue_to_red"].values.astype(float)
X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

print(f"Total: n={len(y)} cells | {len(feat_cols)} features | "
      f"delay_b2r: {y.min():.0f}–{y.max():.0f} min (median {np.median(y):.0f})")

# ── GMM G=3 to define early / medium / late ───────────────────────────────────
gmm = GaussianMixture(n_components=3, covariance_type="full",
                      random_state=42, n_init=10)
gmm.fit(y.reshape(-1, 1))
order = np.argsort(gmm.means_.ravel())
mu    = gmm.means_.ravel()[order]
sig   = np.sqrt(gmm.covariances_.ravel()[order])
pro   = gmm.weights_[order]

x_grid   = np.arange(0, y.max() + 1, 1.0)
dens_mat = np.column_stack([pro[i] * sp_norm.pdf(x_grid, mu[i], sig[i])
                            for i in range(3)])
cls_pred = dens_mat.argmax(axis=1)

idx_early = np.where(cls_pred == 0)[0]
cutoff1   = x_grid[idx_early[-1]] if len(idx_early) > 0 else mu[0] + sig[0]
idx_med   = np.where(cls_pred == 1)[0]
cutoff2   = x_grid[idx_med[-1]]   if len(idx_med)   > 0 else mu[1] + sig[1]

pop_labels = np.where(y <= cutoff1, "early",
             np.where(y <= cutoff2, "medium", "late"))

print(f"\nGMM cutoffs: {cutoff1:.0f} min  |  {cutoff2:.0f} min")
for pop in ["early", "medium", "late"]:
    n_pop = (pop_labels == pop).sum()
    vals  = y[pop_labels == pop]
    print(f"  {pop:8s}: n={n_pop:4d}  range {vals.min():.0f}–{vals.max():.0f} min"
          f"  median {np.median(vals):.0f} min")

# ── ElasticNet CV per population ──────────────────────────────────────────────
def run_en_cv(X_sub, y_sub, n_folds, label):
    n = len(y_sub)
    n_folds_use = min(n_folds, n)
    kf = KFold(n_splits=n_folds_use, shuffle=True, random_state=SEED)
    oof = np.zeros(n)
    y_mean, y_std = y_sub.mean(), y_sub.std() or 1.0
    y_s = (y_sub - y_mean) / y_std

    for fold, (tr, te) in enumerate(kf.split(X_sub)):
        sc = StandardScaler().fit(X_sub[tr])
        en = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=SEED,
                          max_iter=2000, n_alphas=100, tol=1e-4, n_jobs=-1)
        en.fit(sc.transform(X_sub[tr]), y_s[tr])
        oof[te] = en.predict(sc.transform(X_sub[te])) * y_std + y_mean

    r2  = r2_score(y_sub, oof)
    r   = float(pearsonr(y_sub, oof)[0])
    rho = float(spearmanr(y_sub, oof)[0])
    print(f"  {label:8s}: n={n:4d}  R²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}  "
          f"({n_folds_use}-fold CV)", flush=True)
    return r2, r, rho, oof

print(f"\n═══ ElasticNet CV by population ═══", flush=True)

# Individual groups + combined medium+late (mirrors binary classification target)
pop_masks = [
    ("early",       pop_labels == "early"),
    ("medium",      pop_labels == "medium"),
    ("late",        pop_labels == "late"),
    ("medium+late", pop_labels != "early"),
]

rows = []
oof_store = {}
for label, mask in pop_masks:
    idx = np.where(mask)[0]
    r2, r, rho, oof = run_en_cv(X_raw[idx], y[idx], n_folds=5, label=label)
    oof_store[label] = (idx, oof)
    rows.append({"population": label, "n": len(idx),
                 "R2": round(r2, 4), "pearson_r": round(r, 4),
                 "spearman_rho": round(rho, 4)})

# ── combined scatter coloured by population ───────────────────────────────────
cat_colors = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9",
              "medium+late": "#8e44ad"}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (label, mask) in zip(axes.ravel(), pop_masks):
    idx, y_p = oof_store[label]
    y_t = y[idx]
    r2  = r2_score(y_t, y_p)
    rho = float(spearmanr(y_t, y_p)[0])
    ax.scatter(y_t/60, y_p/60, color=cat_colors[label],
               alpha=0.55, s=20, edgecolors="none")
    lo = min(y_t.min(), y_p.min())/60*0.92
    hi = max(y_t.max(), y_p.max())/60*1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Actual BFP→mCherry (h)")
    ax.set_ylabel("CV predicted (h)")
    ax.set_title(f"{label.capitalize()}  (n={len(idx)})\nR²={r2:.3f}  ρ={rho:.3f}")
    ax.spines[["top","right"]].set_visible(False)
plt.suptitle("ElasticNet — per-population CV (45 features, no half-movie filter)",
             y=1.02, fontsize=11)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"reg_en_by_population.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {FIG_DIR}/reg_en_by_population.png")

# ── save metrics ──────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(RES_DIR/"reg_en_by_population_metrics.csv", index=False)
print(f"Metrics saved: {RES_DIR}/reg_en_by_population_metrics.csv")

print("\n══ Summary ══")
print(f"  {'Population':12s}  {'n':>5}  {'R²':>7}  {'r':>7}  {'ρ':>7}")
for row in rows:
    print(f"  {row['population']:12s}  {row['n']:>5}  "
          f"{row['R2']:>7.3f}  {row['pearson_r']:>7.3f}  "
          f"{row['spearman_rho']:>7.3f}")
print("\nDone.", flush=True)

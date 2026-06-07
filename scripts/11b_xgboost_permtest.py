"""
11b_xgboost_permtest.py  —  Permutation test for XGBoost regression

Tests whether the CV R² and Spearman ρ for predicting GFP→mCherry delay
(productive cells only) are above what would be expected by chance.

Procedure:
  1. Average the per-fold best hyperparameters from the real model.
  2. Re-evaluate the real model via 3-fold CV using these fixed params.
  3. For each of N_PERM permutations: shuffle delay labels, run same 3-fold CV.
  4. p-value = fraction of permuted metrics >= real metric.

Fixed hyperparams ensure the comparison is symmetric (no Optuna advantage
for the real model vs permutations).

Output: results/xgboost/permutation_test.png
        results/xgboost/permtest_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
RESULTS_DIR = BASE / "results" / "xgboost"

N_PERM  = 100
N_FOLDS = 3
SEED    = 42

# ── load data (same pipeline as 11_xgboost.py) ────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y",
            "gfp_snr_mean", "bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

X_raw  = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler()
X_all  = scaler.fit_transform(X_raw)

delay      = df["delay_green_to_red"].values.astype(float)
productive = np.isfinite(delay)
prod_idx   = np.where(productive)[0]
X          = X_all[prod_idx]
y          = delay[prod_idx]

print(f"Productive cells: {len(y)}  |  Features: {X.shape[1]}")

# ── build fixed hyperparameters (average across 3 folds from real model) ───────
bp_df = pd.read_csv(RESULTS_DIR / "best_params_regression.csv")

fixed_params = dict(
    max_depth        = int(round(bp_df["max_depth"].mean())),
    learning_rate    = float(bp_df["learning_rate"].mean()),
    subsample        = float(bp_df["subsample"].mean()),
    colsample_bytree = float(bp_df["colsample_bytree"].mean()),
    min_child_weight = int(round(bp_df["min_child_weight"].mean())),
    reg_alpha        = float(bp_df["reg_alpha"].mean()),
    reg_lambda       = float(bp_df["reg_lambda"].mean()),
    gamma            = float(bp_df["gamma"].mean()),
    n_estimators     = int(np.ceil(bp_df["n_estimators"].mean())),
    random_state     = SEED,
    verbosity        = 0,
    n_jobs           = -1,
)
print("Fixed hyperparameters (averaged over 3 folds):")
for k, v in fixed_params.items():
    print(f"  {k:20s} = {v}")

# ── helper: 3-fold CV with fixed params ───────────────────────────────────────
def cv_metrics(y_labels, seed_offset=0):
    kf    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_offset)
    preds = np.zeros(len(y_labels))
    for tr, te in kf.split(X):
        params = dict(**fixed_params)
        params["random_state"] = SEED + seed_offset
        m = XGBRegressor(**params)
        m.fit(X[tr], y_labels[tr], verbose=False)
        preds[te] = m.predict(X[te])
    r2  = r2_score(y_labels, preds)
    rho = spearmanr(y_labels, preds).statistic
    return r2, rho

# ── real model (fixed params) ──────────────────────────────────────────────────
print("\nEvaluating real model with fixed params ...")
real_r2, real_rho = cv_metrics(y, seed_offset=0)
print(f"Real CV  R²={real_r2:.4f}   Spearman ρ={real_rho:.4f}")
print(f"(Original Optuna-tuned model: R²=0.041  ρ=0.285)")

# ── permutation test ───────────────────────────────────────────────────────────
print(f"\nRunning {N_PERM} permutations ...")
rng      = np.random.default_rng(SEED)
perm_r2  = np.zeros(N_PERM)
perm_rho = np.zeros(N_PERM)

for p in range(N_PERM):
    if (p + 1) % 10 == 0:
        print(f"  {p+1}/{N_PERM}", flush=True)
    y_perm       = rng.permutation(y)
    perm_r2[p], perm_rho[p] = cv_metrics(y_perm, seed_offset=p + 1)

p_val_r2  = float(np.mean(perm_r2  >= real_r2))
p_val_rho = float(np.mean(perm_rho >= real_rho))
print(f"\np-value  R²  : {p_val_r2:.4f}   (real={real_r2:.4f}, null mean={perm_r2.mean():.4f})")
print(f"p-value  ρ   : {p_val_rho:.4f}   (real={real_rho:.4f}, null mean={perm_rho.mean():.4f})")

# ── save results ───────────────────────────────────────────────────────────────
pd.DataFrame({
    "perm_r2":  perm_r2,
    "perm_rho": perm_rho,
}).to_csv(RESULTS_DIR / "permtest_null_dist.csv", index=False)

pd.DataFrame([{
    "real_r2":    round(real_r2,  4),
    "real_rho":   round(real_rho, 4),
    "null_mean_r2":   round(perm_r2.mean(),  4),
    "null_mean_rho":  round(perm_rho.mean(), 4),
    "p_val_r2":   p_val_r2,
    "p_val_rho":  p_val_rho,
    "n_perm":     N_PERM,
    "n_cells":    len(y),
}]).to_csv(RESULTS_DIR / "permtest_results.csv", index=False)
print("Saved permtest_results.csv  +  permtest_null_dist.csv")

# ── plot ───────────────────────────────────────────────────────────────────────
def fmt_p(p, n):
    return f"< {1/n:.3f}" if p == 0 else f"= {p:.3f}"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"XGBoost permutation test — GFP→mCherry delay regression\n"
    f"A2+A3 productive cells  (n={len(y)}, {N_PERM} permutations, fixed hyperparams)",
    fontsize=11, fontweight="bold"
)

for ax, perm_vals, real_val, p_val, label, color in [
    (axes[0], perm_r2,  real_r2,  p_val_r2,  "R²",         "#2980b9"),
    (axes[1], perm_rho, real_rho, p_val_rho, "Spearman ρ", "#27ae60"),
]:
    h = ax.hist(perm_vals, bins=30, color=color, alpha=0.55, edgecolor="white")
    # highlight bins at or beyond the real value in red
    bins = np.array([p.get_x() for p in h[2]] + [h[2][-1].get_x() + h[2][-1].get_width()])
    for patch, left, right in zip(h[2], bins[:-1], bins[1:]):
        if left >= real_val:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.7)
    ax.axvline(real_val, color="#c0392b", lw=2.5, label=f"Real {label} = {real_val:.3f}")
    ax.set_xlabel(f"CV {label}  (permuted labels)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}   p {fmt_p(p_val, N_PERM)}")
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.97, 0.95, f"Null mean = {perm_vals.mean():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")

plt.tight_layout()
out = RESULTS_DIR / "permutation_test.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

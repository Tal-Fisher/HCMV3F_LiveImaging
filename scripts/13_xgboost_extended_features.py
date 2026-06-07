"""
13_xgboost_extended_features.py

XGBoost regression on A2+A3 productive cells with 6 new features added:
  1. gfp_at_f16         — GFP intensity at frame 16 (last frame before earliest red onset)
  2. bfp_at_f16         — BFP intensity at frame 16
  3. gfp_delta_mean     — mean GFP change per frame over frames 1-16
  4. bfp_delta_mean     — mean BFP change per frame over frames 1-16
  5. dist_nearest       — distance to nearest infected neighbour at onset (proximity analysis)
  6. n_within_100       — # infected cells within 100 px at onset (proximity analysis)

CV: stratified 4-fold (75/25) by early/mid/late category.
Hyperparameters: averaged from best_params_regression.csv (same as xgb_regression_A2A3.json).
Model saved as xgb_regression_A2A3_extended.json — does NOT overwrite the base model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
import json

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
RESULTS_DIR = BASE / "results" / "xgboost"
FIG_DIR     = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── load base features + outcome ──────────────────────────────────────────────
model_df = pd.read_csv(EXPORT_DIR / "model_df.csv")
model_df = model_df[model_df["abs_gfp_onset_min"] <= model_df["movie_half_min"]].reset_index(drop=True)
print(f"Base model_df after first-half filter: {len(model_df)} cells")

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y", "gfp_snr_mean", "bf_snr_mean"}
base_feat_cols = [c for c in model_df.columns if c not in NON_FEAT]

# ── load new features ─────────────────────────────────────────────────────────
f16   = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
prox  = pd.read_csv(BASE / "proximity_analysis/results/proximity_features.csv")
cat_df = pd.read_csv(EXPORT_DIR / "category_df.csv")

# merge all onto model_df (left join — keep all base cells)
df = model_df.merge(f16,  left_on="Track.ID", right_on="Track.ID", how="left")
df = df.merge(prox[["Track_ID", "dist_nearest", "n_within_100"]],
              left_on="Track.ID", right_on="Track_ID", how="left").drop(columns="Track_ID")
df = df.merge(cat_df[["Track.ID", "category"]], on="Track.ID", how="left")

new_feat_cols = ["gfp_at_f16", "bfp_at_f16", "gfp_delta_mean", "bfp_delta_mean",
                 "dist_nearest", "n_within_100"]
all_feat_cols = base_feat_cols + new_feat_cols
print(f"Total features: {len(all_feat_cols)}  ({len(base_feat_cols)} base + {len(new_feat_cols)} new)")

# ── restrict to productive cells with an early/mid/late category ──────────────
prod_mask = np.isfinite(df["delay_green_to_red"].values.astype(float))
cat_mask  = df["category"].isin(["early", "medium", "late"])
keep      = prod_mask & cat_mask
df_prod   = df[keep].reset_index(drop=True)
print(f"Productive cells (early/mid/late): {len(df_prod)}  "
      f"({df_prod['category'].value_counts().to_dict()})")

y      = df_prod["delay_green_to_red"].values.astype(float)
strata = df_prod["category"].values

# ── feature matrix with median imputation ────────────────────────────────────
X_raw = df_prod[all_feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# ── averaged hyperparameters from existing Optuna runs ───────────────────────
params_df = pd.read_csv(RESULTS_DIR / "best_params_regression.csv")
avg_params = {
    "max_depth":        int(round(params_df["max_depth"].mean())),
    "learning_rate":    float(params_df["learning_rate"].mean()),
    "subsample":        float(params_df["subsample"].mean()),
    "colsample_bytree": float(params_df["colsample_bytree"].mean()),
    "min_child_weight": int(round(params_df["min_child_weight"].mean())),
    "reg_alpha":        float(params_df["reg_alpha"].mean()),
    "reg_lambda":       float(params_df["reg_lambda"].mean()),
    "gamma":            float(params_df["gamma"].mean()),
    "n_estimators":     int(round(params_df["n_estimators"].mean())),
    "random_state":     42,
    "verbosity":        0,
    "n_jobs":           -1,
}
print(f"\nHyperparameters (averaged from {len(params_df)} Optuna folds):")
for k, v in avg_params.items():
    print(f"  {k}: {v}")

# ── stratified 4-fold CV (stratified by early/mid/late) ──────────────────────
print(f"\n── 4-fold stratified CV ──")
skf       = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cv_pred   = np.zeros(len(df_prod))
imp_folds = np.zeros((4, len(all_feat_cols)))

for fold, (tr, te) in enumerate(skf.split(X_raw, strata)):
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler().fit(X_raw[tr])
    X_tr    = scaler.transform(X_raw[tr])
    X_te    = scaler.transform(X_raw[te])

    model = XGBRegressor(**avg_params)
    model.fit(X_tr, y[tr], verbose=False)
    cv_pred[te]     = model.predict(X_te)
    imp_folds[fold] = model.feature_importances_

    r2_fold  = r2_score(y[te], cv_pred[te])
    r_fold   = float(np.corrcoef(y[te], cv_pred[te])[0, 1])
    rho_fold = float(spearmanr(y[te], cv_pred[te]).statistic)
    print(f"  Fold {fold+1}: n_test={len(te):3d}  R²={r2_fold:.3f}  r={r_fold:.3f}  ρ={rho_fold:.3f}")

r2_cv  = r2_score(y, cv_pred)
r_cv   = float(np.corrcoef(y, cv_pred)[0, 1])
rho_cv = float(spearmanr(y, cv_pred).statistic)
print(f"\nCV overall  R²={r2_cv:.3f}  r={r_cv:.3f}  ρ={rho_cv:.3f}")

# ── train final model on all productive cells, save ───────────────────────────
from sklearn.preprocessing import StandardScaler
scaler_final = StandardScaler().fit(X_raw)
X_all        = scaler_final.transform(X_raw)
model_final  = XGBRegressor(**avg_params)
model_final.fit(X_all, y, verbose=False)

model_final.save_model(str(RESULTS_DIR / "xgb_regression_A2A3_extended.json"))
np.save(str(RESULTS_DIR / "xgb_scaler_mean_extended.npy"),  scaler_final.mean_)
np.save(str(RESULTS_DIR / "xgb_scaler_scale_extended.npy"), scaler_final.scale_)
with open(RESULTS_DIR / "xgb_feat_cols_extended.json", "w") as f:
    json.dump(all_feat_cols, f)
print(f"Model saved → results/xgboost/xgb_regression_A2A3_extended.json")

# reference: base model CV (from existing summary)
r2_base, rho_base = 0.041, 0.285

# ── plot ──────────────────────────────────────────────────────────────────────
mean_imp = imp_folds.mean(axis=0)
top_n    = 15
top_idx  = np.argsort(mean_imp)[-top_n:]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"XGBoost extended features — A2+A3  (4-fold stratified CV by early/mid/late)\n"
    f"n={len(df_prod)} productive cells  |  {len(all_feat_cols)} features "
    f"({len(base_feat_cols)} base + {len(new_feat_cols)} new)",
    fontsize=11, fontweight="bold"
)

# panel 1: predicted vs actual
ax = axes[0]
actual_h = y / 60
pred_h   = cv_pred / 60
colors   = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}
for cat, grp in df_prod.groupby("category"):
    idx = grp.index
    ax.scatter(actual_h[idx], pred_h[idx],
               color=colors.get(cat, "grey"), alpha=0.5, s=14, label=cat)
lo = min(actual_h.min(), pred_h.min()) * 0.9
hi = max(actual_h.max(), pred_h.max()) * 1.05
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("Actual GFP→mCherry delay (h)")
ax.set_ylabel("CV predicted delay (h)")
ax.set_title(
    f"CV predictions  (n={len(df_prod)})\n"
    f"R²={r2_cv:.3f}  r={r_cv:.3f}  ρ={rho_cv:.3f}\n"
    f"(base model: R²={r2_base:.3f}  ρ={rho_base:.3f})"
)
ax.legend(fontsize=8, frameon=False)

# panel 2: top feature importances
ax = axes[1]
bar_colors = ["#e67e22" if all_feat_cols[i] in new_feat_cols else "#7f8c8d"
              for i in top_idx]
ax.barh(range(top_n), mean_imp[top_idx], color=bar_colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels([all_feat_cols[i] for i in top_idx], fontsize=8)
ax.set_xlabel("Mean feature importance (gain, averaged over folds)")
ax.set_title("Top 15 features\n(orange = new)")

plt.tight_layout()
fig.savefig(FIG_DIR / "xgboost_extended_features.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved figures/combined/xgboost_extended_features.png")

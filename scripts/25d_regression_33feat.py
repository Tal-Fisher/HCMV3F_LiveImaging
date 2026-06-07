"""
25d_regression_33feat.py

Elastic Net + XGBoost regression for green-to-red delay.
Old filtering: model_df.csv + first-half filter → 497 productive cells.
33 features: 29 base + 4 frame16, NO proximity, NO extended R features.

Isolates whether regression R² difference is driven by cell set or feature set.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
RESULTS_DIR = BASE / "results" / "regression_33feat"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── old filtering: model_df.csv + first-half filter ───────────────────────────
model_df = pd.read_csv(EXPORT_DIR / "model_df.csv")
model_df = model_df[model_df["abs_gfp_onset_min"] <= model_df["movie_half_min"]].reset_index(drop=True)
print(f"After first-half filter: {len(model_df)} cells")

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y", "gfp_snr_mean", "bf_snr_mean"}
base_feat_cols = [c for c in model_df.columns if c not in NON_FEAT]

f16    = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
cat_df = pd.read_csv(EXPORT_DIR / "category_df.csv")

df = model_df.merge(f16, on="Track.ID", how="left")
df = df.merge(cat_df[["Track.ID", "category"]], on="Track.ID", how="left")

frame16_cols = ["gfp_at_f16", "bfp_at_f16", "gfp_delta_mean", "bfp_delta_mean"]
feat_cols    = base_feat_cols + frame16_cols
print(f"Features: {len(feat_cols)}  ({len(base_feat_cols)} base + {len(frame16_cols)} frame16, no proximity)")

# ── productive cells with category for stratified CV ─────────────────────────
prod_mask = np.isfinite(df["delay_green_to_red"].values.astype(float))
cat_mask  = df["category"].isin(["early", "medium", "late"])
df_prod   = df[prod_mask & cat_mask].reset_index(drop=True)
print(f"Productive cells (early/mid/late): {len(df_prod)}  "
      f"({df_prod['category'].value_counts().to_dict()})")

y_g2r  = df_prod["delay_green_to_red"].values.astype(float)
strata = df_prod["category"].values

print(f"\nGreen→Red delay: {y_g2r.min():.0f}–{y_g2r.max():.0f} min  "
      f"(median {np.median(y_g2r):.0f})")

# ── feature matrix ─────────────────────────────────────────────────────────────
X_raw = df_prod[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# ── Elastic Net ────────────────────────────────────────────────────────────────
def run_elasticnet(X, y, label):
    print(f"\n── Elastic Net [{label}] ──")
    outer_kf      = KFold(n_splits=10, shuffle=True, random_state=SEED)
    cv_preds      = np.zeros(len(y))
    y_mean, y_std = y.mean(), y.std()
    y_s           = (y - y_mean) / y_std
    for fold, (tr, te) in enumerate(outer_kf.split(X)):
        scaler   = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        en = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=SEED,
                          max_iter=2000, n_alphas=100, tol=1e-4)
        en.fit(X_tr, y_s[tr])
        cv_preds[te] = en.predict(X_te) * y_std + y_mean
    r2  = r2_score(y, cv_preds)
    r   = float(pearsonr(y, cv_preds)[0])
    rho = float(spearmanr(y, cv_preds)[0])
    print(f"  n={len(y)}  R²={r2:.3f}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")
    return cv_preds, r2, r, rho

# ── XGBoost ───────────────────────────────────────────────────────────────────
def run_xgboost(X, y, strata, label):
    print(f"\n── XGBoost [{label}] ──")
    params_df  = pd.read_csv(BASE / "results/xgboost/best_params_regression.csv")
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
        "random_state": SEED, "verbosity": 0, "n_jobs": 4,
    }
    skf      = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    cv_preds = np.zeros(len(y))
    imp_fold = np.zeros((4, X.shape[1]))
    for fold, (tr, te) in enumerate(skf.split(X, strata)):
        scaler   = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        model    = XGBRegressor(**avg_params)
        model.fit(X_tr, y[tr], verbose=False)
        cv_preds[te]   = model.predict(X_te)
        imp_fold[fold] = model.feature_importances_
        r2_f = r2_score(y[te], cv_preds[te])
        r_f  = float(pearsonr(y[te], cv_preds[te])[0])
        print(f"  Fold {fold+1}: n_test={len(te):3d}  R²={r2_f:.3f}  r={r_f:.3f}")
    r2  = r2_score(y, cv_preds)
    r   = float(pearsonr(y, cv_preds)[0])
    rho = float(spearmanr(y, cv_preds)[0])
    print(f"  CV overall  R²={r2:.3f}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")
    return cv_preds, r2, r, rho, imp_fold.mean(axis=0)

preds_en,  r2_en,  r_en,  rho_en              = run_elasticnet(X_raw, y_g2r, "Green→Red")
preds_xgb, r2_xgb, r_xgb, rho_xgb, imp_mean  = run_xgboost(X_raw, y_g2r, strata, "Green→Red")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"REGRESSION SUMMARY — Green→Red delay")
print(f"n={len(y_g2r)} cells (old filter)  |  {len(feat_cols)} features (no proximity)")
print(f"{'='*60}")
print(f"{'Method':<12}  {'R²':>6}  {'Pearson r':>10}  {'Spearman ρ':>11}")
print(f"{'ElasticNet':<12}  {r2_en:>6.3f}  {r_en:>10.3f}  {rho_en:>11.3f}")
print(f"{'XGBoost':<12}  {r2_xgb:>6.3f}  {r_xgb:>10.3f}  {rho_xgb:>11.3f}")
print(f"{'='*60}")

# ── figure ────────────────────────────────────────────────────────────────────
colors = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Green→Red delay regression  |  n={len(y_g2r)} cells (old filter)  |  "
    f"{len(feat_cols)} features (no proximity)\n"
    f"Elastic Net: R²={r2_en:.3f}  r={r_en:.3f}   |   "
    f"XGBoost: R²={r2_xgb:.3f}  r={r_xgb:.3f}",
    fontsize=11, fontweight="bold"
)

def scatter(ax, y_true, y_pred, title):
    for cat, grp in df_prod.groupby("category"):
        idx = grp.index.tolist()
        ax.scatter(y_true[idx] / 60, y_pred[idx] / 60,
                   color=colors.get(cat, "grey"), alpha=0.45, s=14, label=cat)
    lo = min(y_true.min(), y_pred.min()) / 60 * 0.92
    hi = max(y_true.max(), y_pred.max()) / 60 * 1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Actual delay (h)"); ax.set_ylabel("CV predicted (h)")
    ax.set_title(title); ax.legend(fontsize=8, frameon=False)

scatter(axes[0], y_g2r, preds_en,
        f"Elastic Net\nR²={r2_en:.3f}  r={r_en:.3f}  ρ={rho_en:.3f}")
scatter(axes[1], y_g2r, preds_xgb,
        f"XGBoost\nR²={r2_xgb:.3f}  r={r_xgb:.3f}  ρ={rho_xgb:.3f}")

# feature importance
ax = axes[2]
top_idx = np.argsort(imp_mean)[-15:]
ax.barh(range(15), imp_mean[top_idx], color="#7f8c8d")
ax.set_yticks(range(15))
ax.set_yticklabels([feat_cols[i] for i in top_idx], fontsize=8)
ax.set_xlabel("Mean gain (avg over folds)")
ax.set_title("XGBoost feature importance (top 15)")

plt.tight_layout()
fig.savefig(FIG_DIR / "regression_33feat.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → figures/combined/regression_33feat.png")

# ── save ─────────────────────────────────────────────────────────────────────
pd.DataFrame([
    {"method": "ElasticNet", "target": "green_to_red", "n_cells": len(y_g2r),
     "n_features": len(feat_cols), "R2": round(r2_en, 4),
     "pearson_r": round(r_en, 4), "spearman_r": round(rho_en, 4)},
    {"method": "XGBoost",    "target": "green_to_red", "n_cells": len(y_g2r),
     "n_features": len(feat_cols), "R2": round(r2_xgb, 4),
     "pearson_r": round(r_xgb, 4), "spearman_r": round(rho_xgb, 4)},
]).to_csv(RESULTS_DIR / "metrics.csv", index=False)

pd.DataFrame({
    "Track.ID":    df_prod["Track.ID"].values,
    "category":    strata,
    "y_g2r":       y_g2r.round(2),
    "en_pred":     preds_en.round(2),
    "xgb_pred":    preds_xgb.round(2),
}).to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)

print(f"Results saved to {RESULTS_DIR}")

"""
25_blue_to_red_en_xgb.py

Elastic Net + XGBoost regression for BFP→mCherry delay (blue-to-red).
Uses the same 45-feature extended dataset as 19_elasticnet_productive_only.R.
Mirrors the green-to-red methodology exactly — same cells, same features, same CV.

Run with: python3 25_blue_to_red_en_xgb.py
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
EXT_DIR     = BASE / "results" / "elasticnet_extended2"
RESULTS_DIR = BASE / "results" / "blue_to_red"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── load 45-feature extended dataset ─────────────────────────────────────────
ext_df = pd.read_csv(EXT_DIR / "model_df_extended2.csv")
print(f"Extended dataset: {len(ext_df)} rows × {ext_df.shape[1]} cols")

META_COLS = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue"}
EXTRAS_18 = {
    "cell_aspect_start", "cell_aspect_mean", "bfp_nuc_frac_start",
    "nuc_ratio_start", "nuc_ratio_end",
    "bf_ctrst_start", "bf_ctrst_end", "bf_ctrst_slope",
}
feat_cols = [c for c in ext_df.columns if c not in META_COLS and c not in EXTRAS_18]
print(f"Features: {len(feat_cols)}")

cat_df = pd.read_csv(EXPORT_DIR / "category_df.csv")
df = ext_df.merge(cat_df[["Track.ID", "category"]], on="Track.ID", how="left")

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]

prod_mask = np.isfinite(df["delay_green_to_red"].values.astype(float))
cat_mask  = df["category"].isin(["early", "medium", "late"])
df_prod   = df[prod_mask & cat_mask].reset_index(drop=True)
print(f"Productive cells (early/mid/late): {len(df_prod)}  "
      f"({df_prod['category'].value_counts().to_dict()})")

y_g2r  = df_prod["delay_green_to_red"].values.astype(float)
y_b2r  = df_prod["delay_blue_to_red"].values.astype(float)
strata = df_prod["category"].values

print(f"\nGreen→Red delay: {y_g2r.min():.0f}–{y_g2r.max():.0f} min  "
      f"(median {np.median(y_g2r):.0f})")
print(f"Blue→Red delay:  {y_b2r.min():.0f}–{y_b2r.max():.0f} min  "
      f"(median {np.median(y_b2r):.0f})")

# ── feature matrix (median impute) ────────────────────────────────────────────
X_raw = df_prod[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# ── Elastic Net helper ─────────────────────────────────────────────────────────
def run_elasticnet(X, y, label):
    """10-fold outer CV; inner 5-fold via ElasticNetCV; l1_ratio=0.5 (matching R alpha=0.5).
    y is scaled internally so coordinate descent converges quickly regardless of y units."""
    print(f"\n── Elastic Net [{label}] ──")
    outer_kf  = KFold(n_splits=10, shuffle=True, random_state=SEED)
    cv_preds  = np.zeros(len(y))
    y_mean, y_std = y.mean(), y.std()
    y_s = (y - y_mean) / y_std
    for fold, (tr, te) in enumerate(outer_kf.split(X)):
        scaler = StandardScaler().fit(X[tr])
        X_tr   = scaler.transform(X[tr])
        X_te   = scaler.transform(X[te])
        en     = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=SEED,
                              max_iter=2000, n_alphas=100, tol=1e-4)
        en.fit(X_tr, y_s[tr])
        cv_preds[te] = en.predict(X_te) * y_std + y_mean
    r2  = r2_score(y, cv_preds)
    r   = float(pearsonr(y, cv_preds)[0])
    rho = float(spearmanr(y, cv_preds)[0])
    print(f"  n={len(y)}  R²={r2:.3f}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")
    return cv_preds, r2, r, rho

# ── XGBoost helper ────────────────────────────────────────────────────────────
def run_xgboost(X, y, strata, label):
    """4-fold stratified CV; hyperparameters averaged from Optuna green-to-red runs."""
    print(f"\n── XGBoost [{label}] ──")
    params_df = pd.read_csv(BASE / "results/xgboost/best_params_regression.csv")
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
    skf       = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    cv_preds  = np.zeros(len(y))
    imp_folds = np.zeros((4, X.shape[1]))
    for fold, (tr, te) in enumerate(skf.split(X, strata)):
        scaler = StandardScaler().fit(X[tr])
        X_tr   = scaler.transform(X[tr])
        X_te   = scaler.transform(X[te])
        model  = XGBRegressor(**avg_params)
        model.fit(X_tr, y[tr], verbose=False)
        cv_preds[te]     = model.predict(X_te)
        imp_folds[fold]  = model.feature_importances_
        r2_f = r2_score(y[te], cv_preds[te])
        r_f  = float(pearsonr(y[te], cv_preds[te])[0])
        print(f"  Fold {fold+1}: n_test={len(te):3d}  R²={r2_f:.3f}  r={r_f:.3f}")
    r2  = r2_score(y, cv_preds)
    r   = float(pearsonr(y, cv_preds)[0])
    rho = float(spearmanr(y, cv_preds)[0])
    print(f"  CV overall  R²={r2:.3f}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")
    return cv_preds, r2, r, rho, imp_folds.mean(axis=0)

# ── run all models ────────────────────────────────────────────────────────────
preds_en_g2r,  r2_en_g2r,  r_en_g2r,  rho_en_g2r  = run_elasticnet(X_raw, y_g2r, "Green→Red")
preds_en_b2r,  r2_en_b2r,  r_en_b2r,  rho_en_b2r  = run_elasticnet(X_raw, y_b2r, "Blue→Red")
preds_xgb_g2r, r2_xgb_g2r, r_xgb_g2r, rho_xgb_g2r, imp_g2r = run_xgboost(X_raw, y_g2r, strata, "Green→Red")
preds_xgb_b2r, r2_xgb_b2r, r_xgb_b2r, rho_xgb_b2r, imp_b2r = run_xgboost(X_raw, y_b2r, strata, "Blue→Red")

# ── save predictions ──────────────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "Track.ID":       df_prod["Track.ID"],
    "category":       strata,
    "y_green_to_red": y_g2r.round(2),
    "y_blue_to_red":  y_b2r.round(2),
    "en_pred_g2r":    preds_en_g2r.round(2),
    "en_pred_b2r":    preds_en_b2r.round(2),
    "xgb_pred_g2r":   preds_xgb_g2r.round(2),
    "xgb_pred_b2r":   preds_xgb_b2r.round(2),
})
pred_df.to_csv(RESULTS_DIR / "per_cell_predictions_en_xgb.csv", index=False)

# ── save metrics ──────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame([
    {"method": "ElasticNet", "target": "green_to_red", "R2": round(r2_en_g2r, 4),
     "pearson_r": round(r_en_g2r, 4), "spearman_r": round(rho_en_g2r, 4)},
    {"method": "ElasticNet", "target": "blue_to_red",  "R2": round(r2_en_b2r, 4),
     "pearson_r": round(r_en_b2r, 4), "spearman_r": round(rho_en_b2r, 4)},
    {"method": "XGBoost",    "target": "green_to_red", "R2": round(r2_xgb_g2r, 4),
     "pearson_r": round(r_xgb_g2r, 4), "spearman_r": round(rho_xgb_g2r, 4)},
    {"method": "XGBoost",    "target": "blue_to_red",  "R2": round(r2_xgb_b2r, 4),
     "pearson_r": round(r_xgb_b2r, 4), "spearman_r": round(rho_xgb_b2r, 4)},
])
metrics_df.to_csv(RESULTS_DIR / "metrics_en_xgb.csv", index=False)
print(f"\nSaved results → {RESULTS_DIR}")

# ── figures ───────────────────────────────────────────────────────────────────
colors  = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}
cat_arr = strata

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Blue→Red delay regression — A2+A3  (n={len(df_prod)} productive cells)\n"
    f"{len(feat_cols)} features  |  top row: Green→Red (reference)  |  "
    f"bottom row: Blue→Red",
    fontsize=12, fontweight="bold"
)


def scatter_panel(ax, y_true, y_pred, title, df_prod, colors):
    for cat, grp in df_prod.groupby("category"):
        idx = grp.index.tolist()
        ax.scatter(y_true[idx] / 60, y_pred[idx] / 60,
                   color=colors.get(cat, "grey"), alpha=0.45, s=14, label=cat)
    lo = min(y_true.min(), y_pred.min()) / 60 * 0.92
    hi = max(y_true.max(), y_pred.max()) / 60 * 1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Actual delay (h)")
    ax.set_ylabel("CV predicted delay (h)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)


def feat_imp_panel(ax, mean_imp, title, feat_cols):
    top_n   = 15
    top_idx = np.argsort(mean_imp)[-top_n:]
    ax.barh(range(top_n), mean_imp[top_idx], color="#7f8c8d")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_cols[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("Mean gain (avg over folds)")
    ax.set_title(title)


scatter_panel(axes[0, 0], y_g2r, preds_en_g2r,
              f"Elastic Net — Green→Red (ref)\nR²={r2_en_g2r:.3f}  r={r_en_g2r:.3f}  ρ={rho_en_g2r:.3f}",
              df_prod, colors)
scatter_panel(axes[0, 1], y_g2r, preds_xgb_g2r,
              f"XGBoost — Green→Red (ref)\nR²={r2_xgb_g2r:.3f}  r={r_xgb_g2r:.3f}  ρ={rho_xgb_g2r:.3f}",
              df_prod, colors)
feat_imp_panel(axes[0, 2], imp_g2r, "XGBoost feature importance\nGreen→Red", feat_cols)

scatter_panel(axes[1, 0], y_b2r, preds_en_b2r,
              f"Elastic Net — Blue→Red\nR²={r2_en_b2r:.3f}  r={r_en_b2r:.3f}  ρ={rho_en_b2r:.3f}",
              df_prod, colors)
scatter_panel(axes[1, 1], y_b2r, preds_xgb_b2r,
              f"XGBoost — Blue→Red\nR²={r2_xgb_b2r:.3f}  r={r_xgb_b2r:.3f}  ρ={rho_xgb_b2r:.3f}",
              df_prod, colors)
feat_imp_panel(axes[1, 2], imp_b2r, "XGBoost feature importance\nBlue→Red", feat_cols)

plt.tight_layout()
fig.savefig(FIG_DIR / "blue_to_red_en_xgb.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → figures/combined/blue_to_red_en_xgb.png")

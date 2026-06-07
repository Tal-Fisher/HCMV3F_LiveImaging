"""
Extract EN betas and XGB SHAP values for the blue-to-red models.
Regression: 45-feature extended set, EN + XGB.
Classification: 33-feature set (model_df + frame16), EN + XGB.
Fits on full data using saved/averaged Optuna params.
"""

import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from scipy.stats import norm as sp_norm
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor, XGBClassifier
import shap
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
EXT_DIR    = BASE / "results" / "elasticnet_extended2"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
SEED = 42

# ── helpers ───────────────────────────────────────────────────────────────────
def impute_median(X):
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        X[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
    return X

def avg_params(df):
    p = {}
    for k in df.columns:
        vals = df[k].values
        p[k] = int(round(vals.mean())) if vals.dtype.kind == 'i' else float(vals.mean())
    return p

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION — 45 features
# ══════════════════════════════════════════════════════════════════════════════
print("=== REGRESSION ===")
ext_df  = pd.read_csv(EXT_DIR / "model_df_extended2.csv")
filt_df = pd.read_csv(EXPORT_DIR / "model_df.csv")[["Track.ID","abs_gfp_onset_min","movie_half_min"]]
reg_df  = ext_df.merge(filt_df, on="Track.ID", how="left")

META_COLS = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
             "abs_gfp_onset_min","movie_half_min"}
EXTRAS_18 = {"cell_aspect_start","cell_aspect_mean","bfp_nuc_frac_start",
             "nuc_ratio_start","nuc_ratio_end","bf_ctrst_start","bf_ctrst_end","bf_ctrst_slope"}
feat_reg = [c for c in ext_df.columns if c not in META_COLS and c not in EXTRAS_18]

reg_df["delay_blue_to_red"] = reg_df["delay_green_to_red"] - reg_df["delay_green_to_blue"]
mask = np.isfinite(reg_df["delay_blue_to_red"].values.astype(float))
reg_df = reg_df[mask].reset_index(drop=True)

y_reg  = reg_df["delay_blue_to_red"].values.astype(float)
X_reg  = impute_median(reg_df[feat_reg].values.astype(float))
print(f"  n={len(y_reg)}, {len(feat_reg)} features")

sc_reg = StandardScaler().fit(X_reg)
Xs_reg = sc_reg.transform(X_reg)

# EN: fit on full data, extract betas
print("  Fitting EN regression...")
en_reg = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=SEED,
                      max_iter=2000, n_alphas=100, tol=1e-4, n_jobs=-1)
y_s = (y_reg - y_reg.mean()) / y_reg.std()
en_reg.fit(Xs_reg, y_s)
reg_en_betas = pd.DataFrame({
    "feature": feat_reg,
    "reg_en_beta": en_reg.coef_,
    "reg_en_abs_beta": np.abs(en_reg.coef_),
}).sort_values("reg_en_abs_beta", ascending=False)
print(f"  EN nonzero: {(en_reg.coef_ != 0).sum()}")

# XGB: use averaged best params, compute SHAP
print("  Fitting XGB regression...")
bp_reg = pd.read_csv(RES_DIR / "best_params_b2r_regression.csv").drop(columns=["fold"])
avg_xgb_reg = avg_params(bp_reg)
avg_xgb_reg.update({"random_state": SEED, "verbosity": 0, "n_jobs": 4, "n_estimators": 600})
xgb_reg = XGBRegressor(**avg_xgb_reg)
xgb_reg.fit(Xs_reg, y_reg)

print("  Computing SHAP for regression...")
explainer_reg = shap.TreeExplainer(xgb_reg)
shap_reg = explainer_reg.shap_values(Xs_reg)
reg_xgb_shap = pd.DataFrame({
    "feature": feat_reg,
    "reg_xgb_mean_abs_shap": np.abs(shap_reg).mean(axis=0),
}).sort_values("reg_xgb_mean_abs_shap", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — 33 features (model_df + frame16)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== CLASSIFICATION ===")
cls_df = pd.read_csv(EXPORT_DIR / "model_df.csv")
f16    = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
cls_df = cls_df.merge(f16, on="Track.ID", how="left")

NON_FEAT = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
            "green_onset_min","track_start_min","abs_gfp_onset_min",
            "movie_half_min","y","gfp_snr_mean","bf_snr_mean"}
feat_cls = [c for c in cls_df.columns if c not in NON_FEAT]

cls_df["delay_blue_to_red"] = cls_df["delay_green_to_red"] - cls_df["delay_green_to_blue"]
mask = np.isfinite(cls_df["delay_blue_to_red"].values.astype(float))
cls_df = cls_df[mask].reset_index(drop=True)

delays = cls_df["delay_blue_to_red"].values.astype(float)
X_cls  = impute_median(cls_df[feat_cls].values.astype(float))
print(f"  n={len(delays)}, {len(feat_cls)} features")

# GMM cutoff
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=10)
gmm.fit(delays.reshape(-1, 1))
order = np.argsort(gmm.means_.ravel())
mu = gmm.means_.ravel()[order]; sig = np.sqrt(gmm.covariances_.ravel()[order]); pro = gmm.weights_[order]
x_grid   = np.arange(0, delays.max() + 1, 1.0)
dens_mat = np.column_stack([pro[i] * sp_norm.pdf(x_grid, mu[i], sig[i]) for i in range(3)])
cls_pred = dens_mat.argmax(axis=1)
idx_early = np.where(cls_pred == 0)[0]
cutoff1   = x_grid[idx_early[-1]] if len(idx_early) > 0 else mu[0] + sig[0]
y_cls = (delays <= cutoff1).astype(int)
spw   = float((y_cls==0).sum()) / max(1, y_cls.sum())
print(f"  GMM cutoff: {cutoff1:.0f} min | early={y_cls.sum()} | rest={( y_cls==0).sum()}")

sc_cls = StandardScaler().fit(X_cls)
Xs_cls = sc_cls.transform(X_cls)

# EN: fit on full data
print("  Fitting EN classification...")
en_cls = LogisticRegressionCV(
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0], penalty="elasticnet",
    solver="saga", max_iter=2000, random_state=SEED, n_jobs=-1, cv=5)
en_cls.fit(Xs_cls, y_cls)
cls_en_betas = pd.DataFrame({
    "feature": feat_cls,
    "cls_en_beta": en_cls.coef_[0],
    "cls_en_abs_beta": np.abs(en_cls.coef_[0]),
}).sort_values("cls_en_abs_beta", ascending=False)
print(f"  EN nonzero: {(en_cls.coef_[0] != 0).sum()}")

# XGB: use averaged best params, compute SHAP
print("  Fitting XGB classification...")
bp_cls = pd.read_csv(RES_DIR / "cls_xgb_best_params.csv")
avg_xgb_cls = avg_params(bp_cls)
avg_xgb_cls.update({"random_state": SEED, "verbosity": 0, "n_jobs": 4,
                    "n_estimators": 600, "scale_pos_weight": spw, "eval_metric": "auc"})
xgb_cls = XGBClassifier(**avg_xgb_cls)
xgb_cls.fit(Xs_cls, y_cls)

print("  Computing SHAP for classification...")
explainer_cls = shap.TreeExplainer(xgb_cls)
shap_cls = explainer_cls.shap_values(Xs_cls)
cls_xgb_shap = pd.DataFrame({
    "feature": feat_cls,
    "cls_xgb_mean_abs_shap": np.abs(shap_cls).mean(axis=0),
}).sort_values("cls_xgb_mean_abs_shap", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# MERGE and SAVE
# ══════════════════════════════════════════════════════════════════════════════

# Regression table: all 45 features
reg_table = reg_en_betas.merge(reg_xgb_shap, on="feature", how="outer")
reg_table["reg_en_rank"]  = reg_table["reg_en_abs_beta"].rank(ascending=False).astype(int)
reg_table["reg_xgb_rank"] = reg_table["reg_xgb_mean_abs_shap"].rank(ascending=False).astype(int)
reg_table = reg_table.sort_values("reg_en_abs_beta", ascending=False)
reg_table.to_csv(RES_DIR / "reg_feature_importance.csv", index=False)

# Classification table: all 33 features
cls_table = cls_en_betas.merge(cls_xgb_shap, on="feature", how="outer")
cls_table["cls_en_rank"]  = cls_table["cls_en_abs_beta"].rank(ascending=False).astype(int)
cls_table["cls_xgb_rank"] = cls_table["cls_xgb_mean_abs_shap"].rank(ascending=False).astype(int)
cls_table = cls_table.sort_values("cls_en_abs_beta", ascending=False)
cls_table.to_csv(RES_DIR / "cls_feature_importance.csv", index=False)

# Print top 15 for each
print("\n=== TOP 15 REGRESSION (sorted by |EN beta|) ===")
cols_reg = ["feature","reg_en_beta","reg_en_abs_beta","reg_xgb_mean_abs_shap"]
print(reg_table[cols_reg].head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== TOP 15 CLASSIFICATION (sorted by |EN beta|) ===")
cols_cls = ["feature","cls_en_beta","cls_en_abs_beta","cls_xgb_mean_abs_shap"]
print(cls_table[cols_cls].head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nDone. Saved to results/reg_feature_importance.csv and cls_feature_importance.csv")

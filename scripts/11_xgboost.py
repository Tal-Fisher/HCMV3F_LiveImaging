"""
11_xgboost.py  —  XGBoost on HCMV live imaging features (A2+A3 combined)

Two cross-validated tasks:
  1. Regression  : predict GFP→mCherry delay (productive cells only)
  2. Binary      : productive vs non-productive

Hyperparameters tuned per outer fold via Optuna (TPE sampler).
Nested CV: outer 3-fold evaluation, inner 20% hold-out for Optuna + early stopping.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, roc_curve

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    sys.exit("xgboost not found — run:  pip install --user xgboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit("optuna not found — run:  pip install --user optuna")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "xgboost"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
N_FOLDS  = 3
N_TRIALS = 50   # Optuna trials per outer fold
SEED     = 42

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
print(f"Loaded {len(df)} cells  (A2={( df['dataset']=='A2').sum()}  A3={(df['dataset']=='A3').sum()})")

before = len(df)
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)
print(f"After first-half filter: {len(df)} / {before} cells")

# ── features ───────────────────────────────────────────────────────────────────
NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y",
            "gfp_snr_mean", "bf_snr_mean"}   # 100% missing — excluded
feat_cols = [c for c in df.columns if c not in NON_FEAT]

X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")

# ── outcomes ───────────────────────────────────────────────────────────────────
delay      = df["delay_green_to_red"].values.astype(float)
productive = np.isfinite(delay)
y_bin      = productive.astype(int)
prod_idx   = np.where(productive)[0]
X_prod     = X[prod_idx]
delay_prod = delay[prod_idx]

print(f"Productive: {productive.sum()}  Non-productive: {(~productive).sum()}")

# ── Optuna search space ────────────────────────────────────────────────────────
def suggest_params(trial):
    return dict(
        max_depth          = trial.suggest_int("max_depth", 3, 8),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight   = trial.suggest_int("min_child_weight", 1, 20),
        reg_alpha          = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda         = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma              = trial.suggest_float("gamma", 0.0, 1.0),
        n_estimators       = 600,
        early_stopping_rounds = 40,
        random_state       = SEED,
        verbosity          = 0,
        n_jobs             = -1,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Task 1: Regression ──")

kf               = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_pred_reg      = np.zeros(len(prod_idx))
imp_reg          = np.zeros((N_FOLDS, len(feat_cols)))
best_params_reg  = []

for fold, (tr, te) in enumerate(kf.split(X_prod)):
    print(f"  Fold {fold+1}/{N_FOLDS} — Optuna ({N_TRIALS} trials)...", flush=True)

    n_val    = max(10, len(tr) // 5)
    inner_tr = tr[:-n_val]
    inner_va = tr[-n_val:]

    def reg_obj(trial):
        p = suggest_params(trial)
        m = XGBRegressor(**p, eval_metric="rmse")
        m.fit(X_prod[inner_tr], delay_prod[inner_tr],
              eval_set=[(X_prod[inner_va], delay_prod[inner_va])],
              verbose=False)
        return r2_score(delay_prod[inner_va], m.predict(X_prod[inner_va]))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED + fold))
    study.optimize(reg_obj, n_trials=N_TRIALS)

    bp = study.best_params
    bp.update(random_state=SEED, verbosity=0, n_jobs=-1)

    # Fit on inner split with early stopping to find best n_estimators
    m_iter = XGBRegressor(**bp, n_estimators=600, early_stopping_rounds=40, eval_metric="rmse")
    m_iter.fit(X_prod[inner_tr], delay_prod[inner_tr],
               eval_set=[(X_prod[inner_va], delay_prod[inner_va])],
               verbose=False)
    n_best = m_iter.best_iteration + 1
    # Scale up proportionally since full tr has more data than inner_tr
    n_scaled = max(n_best, int(n_best * len(tr) / len(inner_tr)))

    # Retrain on the full training fold without early stopping
    m_final = XGBRegressor(**bp, n_estimators=n_scaled, eval_metric="rmse")
    m_final.fit(X_prod[tr], delay_prod[tr], verbose=False)
    cv_pred_reg[te] = m_final.predict(X_prod[te])
    imp_reg[fold]   = m_final.feature_importances_

    bp_save = dict(**bp, n_estimators=n_scaled)
    best_params_reg.append(bp_save)
    print(f"    inner R²={study.best_value:.3f}  test R²={r2_score(delay_prod[te], cv_pred_reg[te]):.3f}  n_est={n_scaled}")

r2_cv  = r2_score(delay_prod, cv_pred_reg)
rho_cv = spearmanr(delay_prod, cv_pred_reg).statistic
print(f"Regression CV  R²={r2_cv:.3f}   Spearman ρ={rho_cv:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Binary
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Task 2: Binary ──")

skf              = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_prob_bin      = np.zeros(len(df))
imp_bin          = np.zeros((N_FOLDS, len(feat_cols)))
best_params_bin  = []

for fold, (tr, te) in enumerate(skf.split(X, y_bin)):
    print(f"  Fold {fold+1}/{N_FOLDS} — Optuna ({N_TRIALS} trials)...", flush=True)

    n_val    = max(10, len(tr) // 5)
    inner_tr = tr[:-n_val]
    inner_va = tr[-n_val:]

    spw = (y_bin[tr] == 0).sum() / max((y_bin[tr] == 1).sum(), 1)

    def bin_obj(trial):
        p = suggest_params(trial)
        m = XGBClassifier(**p, scale_pos_weight=spw, eval_metric="auc")
        m.fit(X[inner_tr], y_bin[inner_tr],
              eval_set=[(X[inner_va], y_bin[inner_va])],
              verbose=False)
        return roc_auc_score(y_bin[inner_va], m.predict_proba(X[inner_va])[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED + fold))
    study.optimize(bin_obj, n_trials=N_TRIALS)

    bp = study.best_params
    bp.update(random_state=SEED, verbosity=0, n_jobs=-1)

    # Fit on inner split with early stopping to find best n_estimators
    m_iter = XGBClassifier(**bp, n_estimators=600, early_stopping_rounds=40,
                           scale_pos_weight=spw, eval_metric="auc")
    m_iter.fit(X[inner_tr], y_bin[inner_tr],
               eval_set=[(X[inner_va], y_bin[inner_va])],
               verbose=False)
    n_best = m_iter.best_iteration + 1
    n_scaled = max(n_best, int(n_best * len(tr) / len(inner_tr)))

    # Retrain on the full training fold without early stopping
    m_final = XGBClassifier(**bp, n_estimators=n_scaled,
                            scale_pos_weight=spw, eval_metric="auc")
    m_final.fit(X[tr], y_bin[tr], verbose=False)
    cv_prob_bin[te] = m_final.predict_proba(X[te])[:, 1]
    imp_bin[fold]   = m_final.feature_importances_

    bp_save = dict(**bp, n_estimators=n_scaled)
    best_params_bin.append(bp_save)
    print(f"    inner AUC={study.best_value:.3f}  test AUC={roc_auc_score(y_bin[te], cv_prob_bin[te]):.3f}  n_est={n_scaled}")

auc_cv = roc_auc_score(y_bin, cv_prob_bin)
print(f"Binary CV AUC: {auc_cv:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
mean_imp_reg = imp_reg.mean(axis=0)
mean_imp_bin = imp_bin.mean(axis=0)
top_n = 15

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(
    f"XGBoost + Optuna — HCMV live imaging  (A2+A3, first-half filter)\n"
    f"n={len(df)} cells  |  {N_FOLDS}-fold outer CV  |  {N_TRIALS} Optuna trials/fold",
    fontsize=13, fontweight="bold"
)

# panel 1: regression scatter
ax = axes[0, 0]
ax.scatter(cv_pred_reg, delay_prod, c="steelblue", alpha=0.4, s=14)
lo = min(cv_pred_reg.min(), delay_prod.min())
hi = max(cv_pred_reg.max(), delay_prod.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("CV predicted delay (min)")
ax.set_ylabel("True delay (min)")
ax.set_title(f"Regression  (productive only,  n={len(prod_idx)})\n"
             f"R²={r2_cv:.3f}   Spearman ρ={rho_cv:.3f}")

# panel 2: ROC
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_bin, cv_prob_bin)
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_cv:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Binary — Productive vs non-productive\n{N_FOLDS}-fold CV AUC = {auc_cv:.3f}")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# panel 3: regression feature importance
ax = axes[1, 0]
idx_r = np.argsort(mean_imp_reg)[-top_n:]
colors_r = ["steelblue"] * top_n
ax.barh(range(top_n), mean_imp_reg[idx_r], color=colors_r)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feat_cols[i] for i in idx_r], fontsize=8)
ax.set_xlabel("Mean importance (gain, averaged over folds)")
ax.set_title("Top features — Regression")

# panel 4: binary feature importance
ax = axes[1, 1]
idx_b = np.argsort(mean_imp_bin)[-top_n:]
ax.barh(range(top_n), mean_imp_bin[idx_b], color="tomato")
ax.set_yticks(range(top_n))
ax.set_yticklabels([feat_cols[i] for i in idx_b], fontsize=8)
ax.set_xlabel("Mean importance (gain, averaged over folds)")
ax.set_title("Top features — Binary")

plt.tight_layout()
fig.savefig(FIG_DIR / "xgboost_results.png", dpi=150, bbox_inches="tight")
print(f"\nSaved figures/combined/xgboost_results.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE TABLES
# ══════════════════════════════════════════════════════════════════════════════
pd.DataFrame([
    {"task": "Regression (productive only)", "metric": "R²",           "value": round(r2_cv,  3)},
    {"task": "Regression (productive only)", "metric": "Spearman rho", "value": round(rho_cv, 3)},
    {"task": "Binary (productive vs not)",   "metric": "AUC-ROC",      "value": round(auc_cv, 3)},
]).to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

pd.DataFrame({
    "feature":        feat_cols,
    "importance_reg": mean_imp_reg.round(5),
    "importance_bin": mean_imp_bin.round(5),
}).sort_values("importance_reg", ascending=False).to_csv(
    RESULTS_DIR / "feature_importances.csv", index=False)

cell_res = df[["Track.ID", "dataset", "green_onset_min", "delay_green_to_red"]].copy()
cell_res["productive"]      = productive
cell_res["prob_productive"]  = cv_prob_bin.round(4)
cell_res["pred_productive"]  = (cv_prob_bin >= 0.5).astype(int)
pred_delay = np.full(len(df), np.nan)
pred_delay[prod_idx] = cv_pred_reg
cell_res["pred_delay"] = pred_delay
cell_res.to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)

pd.DataFrame(best_params_reg).assign(fold=range(1, N_FOLDS+1)).to_csv(
    RESULTS_DIR / "best_params_regression.csv", index=False)
pd.DataFrame(best_params_bin).assign(fold=range(1, N_FOLDS+1)).to_csv(
    RESULTS_DIR / "best_params_binary.csv", index=False)

pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(RESULTS_DIR / "roc_curve.csv", index=False)

print(f"Results saved to results/xgboost/")
print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Regression  R²={r2_cv:.3f}   Spearman ρ={rho_cv:.3f}   n={len(prod_idx)} productive cells")
print(f"Binary      AUC={auc_cv:.3f}                          n={len(df)} total cells")

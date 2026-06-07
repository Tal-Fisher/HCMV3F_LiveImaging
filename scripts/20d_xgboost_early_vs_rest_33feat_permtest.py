"""
20d_xgboost_early_vs_rest_33feat_permtest.py

XGBoost classification: early vs medium+late.
Old filtering: model_df.csv + first-half filter → 497 productive cells.
33 features: 29 base + 4 frame16, NO proximity.

Label permutation test: training labels are shuffled N_PERM times using the
averaged best hyperparameters from the real run, giving a null distribution
of Spearman ρ(prob_early, true_label). The real ρ is compared against this
null to compute an empirical p-value.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             balanced_accuracy_score, roc_curve)

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit("xgboost not found")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit("optuna not found")

try:
    import shap
except ImportError:
    sys.exit("shap not found")

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "xgboost_early_33feat"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUT_EARLY_MED = 911
N_FOLDS       = 5
N_TRIALS      = 50
N_PERM        = 500
SEED          = 42
THRESHOLD     = 0.15

# ── old filtering ──────────────────────────────────────────────────────────────
model_df = pd.read_csv(EXPORT_DIR / "model_df.csv")
model_df = model_df[model_df["abs_gfp_onset_min"] <= model_df["movie_half_min"]].reset_index(drop=True)
print(f"After first-half filter: {len(model_df)} cells")

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y", "gfp_snr_mean", "bf_snr_mean"}
base_feat_cols = [c for c in model_df.columns if c not in NON_FEAT]

f16 = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
df  = model_df.merge(f16, on="Track.ID", how="left")

frame16_cols = ["gfp_at_f16", "bfp_at_f16", "gfp_delta_mean", "bfp_delta_mean"]
feat_cols    = base_feat_cols + frame16_cols
print(f"Features: {len(feat_cols)}  ({len(base_feat_cols)} base + {len(frame16_cols)} frame16)")

delay      = df["delay_green_to_red"].values.astype(float)
productive = np.isfinite(delay)
df_prod    = df[productive].reset_index(drop=True)
delay_prod = delay[productive]
print(f"Productive cells: {len(df_prod)}")

X_raw = df_prod[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)

y       = (delay_prod <= CUT_EARLY_MED).astype(int)
n_early = y.sum()
n_rest  = (y == 0).sum()
spw     = n_rest / n_early
print(f"Early (1): {n_early}   Medium+Late (0): {n_rest}   scale_pos_weight={spw:.2f}")

# ── Optuna search space ───────────────────────────────────────────────────────
def suggest_params(trial):
    return dict(
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma            = trial.suggest_float("gamma", 0.0, 1.0),
    )

# ── STEP 1: real nested CV with Optuna ───────────────────────────────────────
outer_cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_proba    = np.zeros(len(y))
cv_pred     = np.zeros(len(y), dtype=int)
cv_shap     = np.zeros((len(y), len(feat_cols)))
best_models = []
best_params = []

print(f"\n── STEP 1: Real CV (Optuna {N_TRIALS} trials/fold) ──")
for fold, (tr, te) in enumerate(outer_cv.split(X, y)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", flush=True)

    n_val    = max(10, len(tr) // 5)
    inner_tr = tr[:-n_val]
    inner_va = tr[-n_val:]

    def clf_obj(trial):
        p = suggest_params(trial)
        m = XGBClassifier(
            **p, n_estimators=600, early_stopping_rounds=40,
            scale_pos_weight=spw, eval_metric="auc",
            random_state=SEED, verbosity=0, n_jobs=4
        )
        m.fit(X[inner_tr], y[inner_tr],
              eval_set=[(X[inner_va], y[inner_va])], verbose=False)
        return roc_auc_score(y[inner_va], m.predict_proba(X[inner_va])[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED + fold))
    study.optimize(clf_obj, n_trials=N_TRIALS)
    bp = study.best_params

    m_iter = XGBClassifier(
        **bp, n_estimators=600, early_stopping_rounds=40,
        scale_pos_weight=spw, eval_metric="auc",
        random_state=SEED, verbosity=0, n_jobs=4
    )
    m_iter.fit(X[inner_tr], y[inner_tr],
               eval_set=[(X[inner_va], y[inner_va])], verbose=False)
    n_best   = m_iter.best_iteration + 1
    n_scaled = max(n_best, int(n_best * len(tr) / len(inner_tr)))

    m_final = XGBClassifier(
        **bp, n_estimators=n_scaled,
        scale_pos_weight=spw, eval_metric="auc",
        random_state=SEED, verbosity=0, n_jobs=4
    )
    m_final.fit(X[tr], y[tr], verbose=False)
    best_models.append(m_final)
    best_params.append({**bp, "n_estimators": n_scaled})

    proba = m_final.predict_proba(X[te])[:, 1]
    cv_proba[te] = proba
    cv_pred[te]  = (proba >= THRESHOLD).astype(int)
    cv_shap[te]  = shap.TreeExplainer(m_final).shap_values(X[te])

    tp_ = ((cv_pred[te] == 1) & (y[te] == 1)).sum()
    fn_ = ((cv_pred[te] == 0) & (y[te] == 1)).sum()
    tn_ = ((cv_pred[te] == 0) & (y[te] == 0)).sum()
    fp_ = ((cv_pred[te] == 1) & (y[te] == 0)).sum()
    s   = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    sp  = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
    print(f"    inner AUC={study.best_value:.3f}  test AUC={roc_auc_score(y[te], proba):.3f}  "
          f"Sens={s:.3f}  Spec={sp:.3f}  n_est={n_scaled}")

# real metrics
tp = ((cv_pred == 1) & (y == 1)).sum()
fn = ((cv_pred == 0) & (y == 1)).sum()
tn = ((cv_pred == 0) & (y == 0)).sum()
fp = ((cv_pred == 1) & (y == 0)).sum()
sensitivity  = tp / (tp + fn)
specificity  = tn / (tn + fp)
auc_cv       = roc_auc_score(y, cv_proba)
bal_acc      = balanced_accuracy_score(y, cv_pred)
real_rho     = spearmanr(y, cv_proba).statistic

print(f"\n{'='*60}")
print(f"REAL CV SUMMARY")
print(f"AUC-ROC={auc_cv:.3f}  Spearman ρ={real_rho:.3f}  "
      f"Sens={sensitivity:.3f}  Spec={specificity:.3f}  BalAcc={bal_acc:.3f}")
print(f"{'='*60}")

# ── STEP 2: label permutation test ────────────────────────────────────────────
# Use averaged best HPs from real run — no Optuna needed for permutations
avg_params = {}
for k in best_params[0]:
    vals = [p[k] for p in best_params]
    avg_params[k] = int(round(np.mean(vals))) if isinstance(vals[0], int) else float(np.mean(vals))

print(f"\n── STEP 2: Label permutation test ({N_PERM} permutations) ──")
print(f"  Using averaged HPs: n_est={avg_params['n_estimators']}  "
      f"lr={avg_params['learning_rate']:.4f}  depth={avg_params['max_depth']}")

rng       = np.random.default_rng(SEED + 999)
perm_rhos = np.zeros(N_PERM)

for perm_i in range(N_PERM):
    perm_proba = np.zeros(len(y))
    for tr, te in outer_cv.split(X, y):
        y_tr_shuffled = rng.permutation(y[tr])   # shuffle training labels only
        scaler_p = StandardScaler().fit(X[tr])
        X_tr_p   = scaler_p.transform(X[tr])
        X_te_p   = scaler_p.transform(X[te])
        m_perm   = XGBClassifier(
            **avg_params, scale_pos_weight=spw, eval_metric="auc",
            random_state=SEED, verbosity=0, n_jobs=4
        )
        m_perm.fit(X_tr_p, y_tr_shuffled, verbose=False)
        perm_proba[te] = m_perm.predict_proba(X_te_p)[:, 1]
    perm_rhos[perm_i] = spearmanr(y, perm_proba).statistic
    if (perm_i + 1) % 50 == 0:
        print(f"  {perm_i+1}/{N_PERM} permutations done  "
              f"(null ρ mean={perm_rhos[:perm_i+1].mean():.3f})", flush=True)

p_value = (np.sum(perm_rhos >= real_rho) + 1) / (N_PERM + 1)
print(f"\nReal Spearman ρ = {real_rho:.4f}")
print(f"Null ρ:  mean={perm_rhos.mean():.4f}  SD={perm_rhos.std():.4f}  "
      f"max={perm_rhos.max():.4f}")
print(f"Empirical p-value = {p_value:.4f}  ({np.sum(perm_rhos >= real_rho)}/{N_PERM} "
      f"permutations ≥ real ρ)")

# ── FIGURES ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"XGBoost — Early vs Med+Late  |  n={len(y)} cells (old filter)  |  {len(feat_cols)} features (no proximity)\n"
    f"{N_FOLDS}-fold nested CV  |  AUC={auc_cv:.3f}  Sens={sensitivity:.2f}  "
    f"Spec={specificity:.2f}  BalAcc={bal_acc:.2f}",
    fontsize=11, fontweight="bold"
)

# confusion matrix
ax = axes[0, 0]
cm_disp = confusion_matrix(y, cv_pred, labels=[1, 0])
cm_norm = cm_disp.astype(float) / cm_disp.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Early", "Pred Med+Late"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True Early", "True Med+Late"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{100*cm_norm[i,j]:.0f}%\n(n={cm_disp[i,j]})",
                ha="center", va="center", fontsize=12,
                color="white" if cm_norm[i, j] > 0.55 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f"Confusion matrix (% of true class)\n"
             f"Sens={sensitivity:.3f}   Spec={specificity:.3f}   Bal acc={bal_acc:.3f}")

# ROC curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y, cv_proba)
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"XGBoost  AUC={auc_cv:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve"); ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# SHAP feature importance
ax = axes[1, 0]
mean_abs_shap = np.abs(cv_shap).mean(axis=0)
order_shap    = np.argsort(mean_abs_shap)[-20:]
from scipy.stats import spearmanr as _sp
corr = np.array([_sp(X[:, i], cv_shap[:, i]).statistic for i in order_shap])
colors_s = ["#e74c3c" if c > 0 else "#2980b9" for c in corr]
ax.barh(range(len(order_shap)), mean_abs_shap[order_shap], color=colors_s, height=0.7)
ax.set_yticks(range(len(order_shap)))
ax.set_yticklabels([feat_cols[i] for i in order_shap], fontsize=7)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP feature importance (top 20)\nred = high → early  |  blue = high → med+late")

# label permutation test
ax = axes[1, 1]
ax.hist(perm_rhos, bins=40, color="#95a5a6", edgecolor="white", linewidth=0.4,
        label=f"Null distribution\n(n={N_PERM} label permutations)")
ax.axvline(real_rho, color="#e74c3c", lw=2.5,
           label=f"Real Spearman ρ = {real_rho:.3f}\np = {p_value:.4f}")
ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Spearman ρ (prob_early vs true label)")
ax.set_ylabel("Count")
ax.set_title(f"Label permutation test\n"
             f"Real ρ={real_rho:.3f}  |  null mean={perm_rhos.mean():.3f}  |  p={p_value:.4f}")
ax.legend(fontsize=9)

plt.tight_layout()
out = FIG_DIR / "xgboost_early_vs_rest_33feat_permtest.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

# beeswarm SHAP
fig_b, ax_b = plt.subplots(figsize=(9, 8))
shap.summary_plot(cv_shap, X, feature_names=feat_cols, max_display=20,
                  show=False, plot_type="dot", color_bar=True)
plt.title(
    f"SHAP beeswarm — XGBoost early vs med+late\n"
    f"n={len(y)} cells (old filter)  |  {len(feat_cols)} features (no proximity)  |  {N_FOLDS}-fold CV\n"
    f"positive SHAP → predicts early",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "xgboost_early_vs_rest_33feat_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# ── save ─────────────────────────────────────────────────────────────────────
pd.DataFrame([
    {"metric": "AUC-ROC",            "value": round(auc_cv,      3)},
    {"metric": "Spearman rho",        "value": round(real_rho,    4)},
    {"metric": "Permutation p-value", "value": round(p_value,     4)},
    {"metric": "Sensitivity",        "value": round(sensitivity, 3)},
    {"metric": "Specificity",        "value": round(specificity, 3)},
    {"metric": "Balanced acc",       "value": round(bal_acc,     3)},
    {"metric": "n_cells",            "value": int(len(y))},
    {"metric": "n_features",         "value": int(len(feat_cols))},
    {"metric": "TP (early correct)", "value": int(tp)},
    {"metric": "FN (early missed)",  "value": int(fn)},
    {"metric": "TN (rest correct)",  "value": int(tn)},
    {"metric": "FP (rest as early)", "value": int(fp)},
]).to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

pd.DataFrame(best_params).to_csv(RESULTS_DIR / "best_params.csv", index=False)
pd.DataFrame({"perm_rho": perm_rhos}).to_csv(RESULTS_DIR / "permutation_null_rhos.csv", index=False)

pd.DataFrame({
    "feature":       feat_cols,
    "mean_abs_shap": np.abs(cv_shap).mean(axis=0).round(5),
    "mean_shap":     cv_shap.mean(axis=0).round(5),
}).sort_values("mean_abs_shap", ascending=False).to_csv(
    RESULTS_DIR / "feature_importance.csv", index=False)

pd.DataFrame({
    "Track.ID":   df_prod["Track.ID"].values,
    "true_label": y,
    "pred_label": cv_pred,
    "prob_early": cv_proba.round(4),
}).to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(RESULTS_DIR / "roc_curve.csv", index=False)

print(f"Results saved to {RESULTS_DIR}")

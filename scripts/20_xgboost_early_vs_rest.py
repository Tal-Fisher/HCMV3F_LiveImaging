"""
20_xgboost_early_vs_rest.py

XGBoost binary classification: early vs medium+late (productive cells only).
GMM cutoffs: early ≤ 911 min, medium 911–2163 min, late > 2163 min.

Uses the same 45-feature extended dataset as the regression scripts.
No proximity features.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
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
EXT_DIR     = BASE / "results" / "elasticnet_extended2"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "xgboost_early"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUT_EARLY_MED = 911
N_FOLDS       = 5
N_TRIALS      = 50
SEED          = 42
THRESHOLD     = 0.15

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

prod_mask = np.isfinite(df["delay_green_to_red"].values.astype(float))
cat_mask  = df["category"].isin(["early", "medium", "late"])
df_prod   = df[prod_mask & cat_mask].reset_index(drop=True)
print(f"Productive cells: {len(df_prod)}")

# ── feature matrix + binary label ────────────────────────────────────────────
X_raw = df_prod[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)

delay   = df_prod["delay_green_to_red"].values.astype(float)
y       = (delay <= CUT_EARLY_MED).astype(int)
n_early = y.sum()
n_rest  = (y == 0).sum()
spw     = n_rest / n_early
print(f"Early (1): {n_early}   Medium+Late (0): {n_rest}   scale_pos_weight={spw:.2f}")

# ── Optuna search space ────────────────────────────────────────────────────────
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

# ── nested CV ─────────────────────────────────────────────────────────────────
outer_cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_proba    = np.zeros(len(y))
cv_pred     = np.zeros(len(y), dtype=int)
cv_shap     = np.zeros((len(y), len(feat_cols)))
best_models = []
best_params = []

print(f"\nRunning {N_FOLDS}-fold nested CV (Optuna {N_TRIALS} trials/fold)...")
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

    explainer_fold = shap.TreeExplainer(m_final)
    cv_shap[te]    = explainer_fold.shap_values(X[te])

    tp_ = ((cv_pred[te] == 1) & (y[te] == 1)).sum()
    fn_ = ((cv_pred[te] == 0) & (y[te] == 1)).sum()
    tn_ = ((cv_pred[te] == 0) & (y[te] == 0)).sum()
    fp_ = ((cv_pred[te] == 1) & (y[te] == 0)).sum()
    s   = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    sp  = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
    print(f"    inner AUC={study.best_value:.3f}  test AUC={roc_auc_score(y[te], proba):.3f}  "
          f"Sens={s:.3f}  Spec={sp:.3f}  n_est={n_scaled}")

# ── overall metrics ────────────────────────────────────────────────────────────
tp = ((cv_pred == 1) & (y == 1)).sum()
fn = ((cv_pred == 0) & (y == 1)).sum()
tn = ((cv_pred == 0) & (y == 0)).sum()
fp = ((cv_pred == 1) & (y == 0)).sum()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc_cv      = roc_auc_score(y, cv_proba)
bal_acc     = balanced_accuracy_score(y, cv_pred)

print(f"\n{'='*50}")
print(f"CV SUMMARY ({N_FOLDS}-fold, threshold={THRESHOLD})")
print(f"{'='*50}")
print(f"AUC-ROC       : {auc_cv:.3f}")
print(f"Sensitivity   : {sensitivity:.3f}  ({tp}/{tp+fn} early cells correctly identified)")
print(f"Specificity   : {specificity:.3f}  ({tn}/{tn+fp} med+late cells correctly identified)")
print(f"Balanced acc  : {bal_acc:.3f}")

print(f"\nThreshold sweep:")
print(f"{'Thresh':>8}  {'Sens':>6}  {'Spec':>6}  {'BalAcc':>8}")
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    p = (cv_proba >= thr).astype(int)
    tp_ = ((p == 1) & (y == 1)).sum()
    fn_ = ((p == 0) & (y == 1)).sum()
    tn_ = ((p == 0) & (y == 0)).sum()
    fp_ = ((p == 1) & (y == 0)).sum()
    s  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    sp = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
    ba = (s + sp) / 2
    print(f"{thr:>8.2f}  {s:>6.3f}  {sp:>6.3f}  {ba:>8.3f}")

# ── figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"XGBoost — Early vs Medium+Late  (n_early={n_early}, n_rest={n_rest})\n"
    f"{N_FOLDS}-fold nested CV  |  scale_pos_weight={spw:.1f}  |  threshold={THRESHOLD}  |  {len(feat_cols)} features",
    fontsize=12, fontweight="bold"
)

# panel 1: confusion matrix
ax = axes[0, 0]
cm_disp = confusion_matrix(y, cv_pred, labels=[1, 0])
cm_norm = cm_disp.astype(float) / cm_disp.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Early", "Pred Med+Late"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True Early", "True Med+Late"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{100*cm_norm[i,j]:.0f}%\n(n={cm_disp[i,j]})",
                ha="center", va="center", fontsize=11,
                color="white" if cm_norm[i, j] > 0.55 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f"Confusion matrix\nSens={sensitivity:.2f}  Spec={specificity:.2f}  "
             f"AUC={auc_cv:.3f}  Bal={bal_acc:.3f}")

# panel 2: ROC curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y, cv_proba)
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"XGBoost  AUC={auc_cv:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# panel 3: SHAP feature importance
ax = axes[1, 0]
mean_abs_shap = np.abs(cv_shap).mean(axis=0)
order = np.argsort(mean_abs_shap)[-20:]
feat_names_top = [feat_cols[i] for i in order]
vals_top       = mean_abs_shap[order]
from scipy.stats import spearmanr as _sp
corr_feat_shap = np.array([_sp(X[:, i], cv_shap[:, i]).statistic for i in order])
colors_shap    = ["#e74c3c" if c > 0 else "#2980b9" for c in corr_feat_shap]
y_pos = range(len(feat_names_top))
ax.barh(y_pos, vals_top, color=colors_shap, height=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels(feat_names_top, fontsize=7)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP feature importance (top 20)\nred=high value → early  |  blue=high value → med+late")

# panel 4: threshold sweep
ax = axes[1, 1]
thresholds = np.linspace(0.01, 0.99, 200)
sens_arr, spec_arr, bal_arr = [], [], []
for thr in thresholds:
    p = (cv_proba >= thr).astype(int)
    tp_ = ((p == 1) & (y == 1)).sum()
    fn_ = ((p == 0) & (y == 1)).sum()
    tn_ = ((p == 0) & (y == 0)).sum()
    fp_ = ((p == 1) & (y == 0)).sum()
    s  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    sp = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
    sens_arr.append(s); spec_arr.append(sp); bal_arr.append((s + sp) / 2)
ax.plot(thresholds, sens_arr, label="Sensitivity", color="#e74c3c")
ax.plot(thresholds, spec_arr, label="Specificity", color="#2980b9")
ax.plot(thresholds, bal_arr,  label="Balanced acc", color="#27ae60", lw=2)
ax.axvline(THRESHOLD, color="black", ls="--", lw=0.9, label=f"Threshold={THRESHOLD}")
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold sweep")
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
out = FIG_DIR / "xgboost_early_vs_rest.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

# ── beeswarm SHAP ─────────────────────────────────────────────────────────────
print("Saving beeswarm SHAP figure...")
fig_bees, ax_bees = plt.subplots(figsize=(9, 10))
shap.summary_plot(cv_shap, X, feature_names=feat_cols, max_display=20,
                  show=False, plot_type="dot", color_bar=True)
plt.title(
    f"SHAP beeswarm — XGBoost early vs medium+late\n"
    f"n={len(y)} productive cells  |  each dot = one cell  |  "
    f"colour = feature value (red=high, blue=low)\n"
    f"positive SHAP → predicts early  |  {N_FOLDS}-fold CV",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
bees_out = FIG_DIR / "xgboost_early_vs_rest_shap_beeswarm.png"
plt.savefig(bees_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {bees_out}")

# ── save results ───────────────────────────────────────────────────────────────
pd.DataFrame([
    {"metric": "AUC-ROC",            "value": round(auc_cv,      3)},
    {"metric": "Sensitivity",        "value": round(sensitivity, 3)},
    {"metric": "Specificity",        "value": round(specificity, 3)},
    {"metric": "Balanced acc",       "value": round(bal_acc,     3)},
    {"metric": "TP (early correct)", "value": int(tp)},
    {"metric": "FN (early missed)",  "value": int(fn)},
    {"metric": "TN (rest correct)",  "value": int(tn)},
    {"metric": "FP (rest as early)", "value": int(fp)},
]).to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

pd.DataFrame(best_params).to_csv(RESULTS_DIR / "best_params.csv", index=False)

pd.DataFrame({
    "feature":       feat_cols,
    "mean_abs_shap": np.abs(cv_shap).mean(axis=0).round(5),
    "mean_shap":     cv_shap.mean(axis=0).round(5),
    "gain":          best_models[-1].feature_importances_.round(5),
}).sort_values("mean_abs_shap", ascending=False).to_csv(
    RESULTS_DIR / "feature_importance.csv", index=False)

pd.DataFrame({
    "Track.ID":   df_prod["Track.ID"].values,
    "true_label": y,
    "pred_label": cv_pred,
    "prob_early": cv_proba.round(4),
}).to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)

pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
    RESULTS_DIR / "roc_curve.csv", index=False)

# ── save final model ──────────────────────────────────────────────────────────
print("\nTraining final model on all productive cells...")
avg_params = {}
for k in best_params[0]:
    vals = [p[k] for p in best_params]
    avg_params[k] = int(round(np.mean(vals))) if isinstance(vals[0], int) else float(np.mean(vals))

m_save = XGBClassifier(
    **avg_params, scale_pos_weight=spw, eval_metric="auc",
    random_state=SEED, verbosity=0, n_jobs=4
)
m_save.fit(X, y, verbose=False)
m_save.save_model(str(RESULTS_DIR / "xgb_early_vs_rest.json"))
np.save(str(RESULTS_DIR / "scaler_mean.npy"),  scaler.mean_)
np.save(str(RESULTS_DIR / "scaler_scale.npy"), scaler.scale_)
with open(RESULTS_DIR / "feat_cols.json", "w") as f:
    json.dump(feat_cols, f)

print(f"Model saved → results/xgboost_early/xgb_early_vs_rest.json")
print(f"Results saved to results/xgboost_early/")

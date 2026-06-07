"""
13_elasticnet_early_vs_rest.py

Binary elastic net: early vs medium+late (productive cells only).
GMM cutoffs: early ≤ 911 min, medium 911–2163, late > 2163.

Uses the same 45-feature extended dataset as the regression scripts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             balanced_accuracy_score, roc_curve)

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
EXT_DIR     = BASE / "results" / "elasticnet_extended2"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "elasticnet_early"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUT_EARLY_MED = 911
N_FOLDS       = 5
SEED          = 42

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

delay = df_prod["delay_green_to_red"].values.astype(float)
y     = (delay <= CUT_EARLY_MED).astype(int)   # 1=early, 0=medium+late
print(f"Early (1): {y.sum()}   Medium+Late (0): {(y==0).sum()}")

# ── hyperparameter grid ────────────────────────────────────────────────────────
param_grid = {
    "C":        [0.001, 0.01, 0.1, 1, 10],
    "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
}

# ── nested CV ─────────────────────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)

cv_proba    = np.zeros(len(y))
cv_pred     = np.zeros(len(y), dtype=int)
best_models = []

print(f"\nRunning {N_FOLDS}-fold nested CV...")
for fold, (tr, te) in enumerate(outer_cv.split(X, y)):
    gs = GridSearchCV(
        LogisticRegression(
            penalty="elasticnet", solver="saga",
            class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=4
    )
    gs.fit(X[tr], y[tr])
    best_models.append(gs.best_estimator_)
    cv_proba[te] = gs.predict_proba(X[te])[:, 1]
    cv_pred[te]  = gs.predict(X[te])

    tp_ = ((cv_pred[te] == 1) & (y[te] == 1)).sum()
    fn_ = ((cv_pred[te] == 0) & (y[te] == 1)).sum()
    tn_ = ((cv_pred[te] == 0) & (y[te] == 0)).sum()
    fp_ = ((cv_pred[te] == 1) & (y[te] == 0)).sum()
    sens = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    spec = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y[te], cv_proba[te]):.3f}  "
          f"Sens={sens:.3f}  Spec={spec:.3f}  "
          f"best C={gs.best_params_['C']}  l1r={gs.best_params_['l1_ratio']}")

# ── overall CV metrics ─────────────────────────────────────────────────────────
tp = ((cv_pred == 1) & (y == 1)).sum()
fn = ((cv_pred == 0) & (y == 1)).sum()
tn = ((cv_pred == 0) & (y == 0)).sum()
fp = ((cv_pred == 1) & (y == 0)).sum()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc_cv      = roc_auc_score(y, cv_proba)
bal_acc     = balanced_accuracy_score(y, cv_pred)

print(f"\n{'='*50}")
print(f"CV SUMMARY ({N_FOLDS}-fold, early=1 vs medium+late=0)")
print(f"{'='*50}")
print(f"AUC-ROC       : {auc_cv:.3f}")
print(f"Sensitivity   : {sensitivity:.3f}  ({tp}/{tp+fn} early cells correctly identified)")
print(f"Specificity   : {specificity:.3f}  ({tn}/{tn+fp} med+late cells correctly identified)")
print(f"Balanced acc  : {bal_acc:.3f}")

# ── figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Elastic Net — Early vs Medium+Late  (n_early={y.sum()}, n_rest={(y==0).sum()})\n"
    f"{N_FOLDS}-fold CV  |  class_weight='balanced'  |  {len(feat_cols)} features",
    fontsize=12, fontweight="bold"
)

# panel 1: confusion matrix
ax = axes[0]
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
ax.set_title(f"Confusion matrix (% of true class)\n"
             f"Sens={sensitivity:.2f}   Spec={specificity:.2f}\n"
             f"AUC={auc_cv:.3f}   Balanced acc={bal_acc:.3f}")

# panel 2: ROC curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y, cv_proba)
ax.plot(fpr, tpr, color="#95a5a6", lw=2, label=f"Elastic Net  AUC={auc_cv:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC curve\n{N_FOLDS}-fold CV")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# panel 3: mean coefficients across folds
ax = axes[2]
coef_mat  = np.vstack([m.coef_[0] for m in best_models])
mean_coef = coef_mat.mean(axis=0)
sd_coef   = coef_mat.std(axis=0)
nonzero   = np.abs(mean_coef) > 1e-6
if nonzero.sum() == 0:
    nonzero = np.ones(len(mean_coef), dtype=bool)
idx_sort  = np.argsort(mean_coef[nonzero])
feat_nz   = [feat_cols[i] for i, keep in enumerate(nonzero) if keep]
vals_nz   = mean_coef[nonzero][idx_sort]
errs_nz   = sd_coef[nonzero][idx_sort]
feats_nz  = [feat_nz[i] for i in idx_sort]
colors_c  = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals_nz]
y_pos     = range(len(feats_nz))
ax.barh(y_pos, vals_nz, xerr=errs_nz, color=colors_c,
        error_kw={"elinewidth": 0.8, "capsize": 2}, height=0.7)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels(feats_nz, fontsize=7)
ax.set_xlabel("Mean coefficient (positive = predicts early)")
ax.set_title("Elastic net coefficients\n(mean ± SD across 5 folds)")

plt.tight_layout()
out = FIG_DIR / "elasticnet_early_vs_rest.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

# ── save results ───────────────────────────────────────────────────────────────
pd.DataFrame([
    {"metric": "AUC-ROC",             "value": round(auc_cv,      3)},
    {"metric": "Sensitivity",         "value": round(sensitivity, 3)},
    {"metric": "Specificity",         "value": round(specificity, 3)},
    {"metric": "Balanced acc",        "value": round(bal_acc,     3)},
    {"metric": "TP (early correct)",  "value": int(tp)},
    {"metric": "FN (early missed)",   "value": int(fn)},
    {"metric": "TN (rest correct)",   "value": int(tn)},
    {"metric": "FP (rest as early)",  "value": int(fp)},
]).to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

pd.DataFrame({
    "feature":   feat_cols,
    "mean_coef": mean_coef.round(5),
    "sd_coef":   sd_coef.round(5),
}).sort_values("mean_coef", key=abs, ascending=False).to_csv(
    RESULTS_DIR / "coefficients.csv", index=False)

pd.DataFrame({
    "Track.ID":   df_prod["Track.ID"].values,
    "true_label": y,
    "pred_label": cv_pred,
    "prob_early": cv_proba.round(4),
}).to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)

pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
    RESULTS_DIR / "roc_curve.csv", index=False)

print(f"Results saved to results/elasticnet_early/")

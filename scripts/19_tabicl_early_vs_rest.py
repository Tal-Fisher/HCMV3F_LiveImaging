"""
19_tabicl_early_vs_rest.py

TabICL binary classification: early vs medium+late (productive cells only).
GMM cutoffs: early ≤ 911 min, medium 911–2163 min, late > 2163 min.

Uses the same 45-feature extended dataset as the regression scripts.
Run with: /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12 19_tabicl_early_vs_rest.py
"""

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
    from tabicl import TabICLClassifier
except ImportError:
    raise ImportError(
        "tabicl not found — run with: "
        "/home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12"
    )

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
EXT_DIR     = BASE / "results" / "elasticnet_extended2"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "tabicl_early"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUT_EARLY_MED = 911
N_FOLDS       = 5
N_EST         = 8
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

delay = df_prod["delay_green_to_red"].values.astype(float)
y     = (delay <= CUT_EARLY_MED).astype(int)
print(f"Early (1): {y.sum()}   Medium+Late (0): {(y==0).sum()}")

# ── 5-fold stratified CV ───────────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_proba = np.zeros(len(y))
cv_pred  = np.zeros(len(y), dtype=int)

print(f"\nRunning {N_FOLDS}-fold stratified CV with TabICLClassifier (n_estimators={N_EST})...")
print(f"  No oversampling  |  Threshold: {THRESHOLD}")
for fold, (tr, te) in enumerate(outer_cv.split(X, y)):
    clf = TabICLClassifier(n_estimators=N_EST, random_state=SEED)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    cv_proba[te] = proba
    cv_pred[te]  = (proba >= THRESHOLD).astype(int)

    tp_ = ((cv_pred[te] == 1) & (y[te] == 1)).sum()
    fn_ = ((cv_pred[te] == 0) & (y[te] == 1)).sum()
    tn_ = ((cv_pred[te] == 0) & (y[te] == 0)).sum()
    fp_ = ((cv_pred[te] == 1) & (y[te] == 0)).sum()
    sens = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    spec = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y[te], cv_proba[te]):.3f}  "
          f"Sens={sens:.3f}  Spec={spec:.3f}")

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
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"TabICL — Early vs Medium+Late  (n_early={y.sum()}, n_rest={(y==0).sum()})\n"
    f"{N_FOLDS}-fold stratified CV  |  n_estimators={N_EST}  |  threshold={THRESHOLD}  |  {len(feat_cols)} features",
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
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"TabICL  AUC={auc_cv:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC curve  |  {N_FOLDS}-fold CV")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
out = FIG_DIR / "tabicl_early_vs_rest.png"
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
    "Track.ID":   df_prod["Track.ID"].values,
    "true_label": y,
    "pred_label": cv_pred,
    "prob_early": cv_proba.round(4),
}).to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)

pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
    RESULTS_DIR / "roc_curve.csv", index=False)

print(f"Results saved to results/tabicl_early/")

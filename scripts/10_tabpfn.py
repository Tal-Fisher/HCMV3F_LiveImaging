"""
10_tabpfn.py  —  TabPFN v2 on HCMV live imaging features (A2+A3 combined)

Three cross-validated tasks, restricted to cells with GFP onset in the first
half of the movie (≤ 2160 min = half of 4320-min movies):

  1. Regression   : predict GFP→mCherry delay (productive cells only)
  2. Binary       : productive vs non-productive
  3. 3-class      : early / medium / late  (fixed GMM cutoffs: 911, 2108 min)

All tasks use 3-fold CV on the shuffled combined dataset.
Features are z-normalised and median-imputed (matching the R pipeline).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, r2_score, roc_curve)

try:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    _token = os.environ.get("TABPFN_TOKEN", "")
    if not _token:
        sys.exit("Set TABPFN_TOKEN environment variable before running")
    tabpfn_client.set_access_token(_token)
    tabpfn_client.init(use_server=True)
except ImportError:
    sys.exit("tabpfn-client not found — run:  pip install --user tabpfn-client")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
CUT_EARLY_MED = 911    # GMM boundary: early | medium (min)
CUT_MED_LATE  = 2163   # GMM boundary: medium | late  (min)
N_FOLDS       = 3
N_EST         = 8
SEED          = 42
CAT_NAMES     = ["early", "medium", "late"]

# ── load exported data ─────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")

print(f"Loaded {len(df)} cells from combined cache")
print(f"  A2: {(df['dataset']=='A2').sum()}   A3: {(df['dataset']=='A3').sum()}")

# ── first-half filter ──────────────────────────────────────────────────────────
# Keep cells whose GFP onset occurred in the absolute first half of the movie.
# Uses abs_gfp_onset_min = track_start_min + green_onset_min (from movie start).
before = len(df)
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)
print(f"After first-half filter: {len(df)} / {before} cells")

# ── feature matrix ─────────────────────────────────────────────────────────────
NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y",
            "gfp_snr_mean", "bf_snr_mean"}   # 100% missing — excluded
feat_cols = [c for c in df.columns if c not in NON_FEAT]

X_raw = df[feat_cols].values.astype(float)

# median impute
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# z-normalise
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Feature matrix: {X.shape[0]} cells × {X.shape[1]} features")
print(f"Features: {feat_cols}")

# ── outcome variables ──────────────────────────────────────────────────────────
delay = df["delay_green_to_red"].values.astype(float)
productive = np.isfinite(delay)

# binary labels
y_bin = productive.astype(int)

# productive cells subset
prod_idx   = np.where(productive)[0]
X_prod     = X[prod_idx]
delay_prod = delay[prod_idx]

def to_cat(d):
    if d <= CUT_EARLY_MED: return 0
    if d <= CUT_MED_LATE:  return 1
    return 2

y_cat = np.array([to_cat(d) for d in delay_prod])

print(f"\nOutcomes:")
print(f"  Productive: {productive.sum()}  Non-productive: {(~productive).sum()}")
print(f"  3-class — early: {(y_cat==0).sum()}  medium: {(y_cat==1).sum()}  late: {(y_cat==2).sum()}")

# ── TabPFN v2 helpers ──────────────────────────────────────────────────────────
def make_clf(**kw):
    return TabPFNClassifier(n_estimators=N_EST, random_state=SEED, **kw)

def make_reg(**kw):
    return TabPFNRegressor(n_estimators=N_EST, random_state=SEED, **kw)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Regression: predict delay (productive cells only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Task 1: Regression (productive cells only) ──")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_pred_reg = np.zeros(len(prod_idx))

for fold, (tr, te) in enumerate(kf.split(X_prod)):
    reg = make_reg()
    reg.fit(X_prod[tr], delay_prod[tr])
    cv_pred_reg[te] = reg.predict(X_prod[te])
    print(f"  fold {fold+1}/{N_FOLDS}")

r2_prod  = r2_score(delay_prod, cv_pred_reg)
rho_prod = spearmanr(delay_prod, cv_pred_reg).statistic

print(f"Productive cells  R²={r2_prod:.3f}   Spearman ρ={rho_prod:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Binary: productive vs non-productive
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Task 2: Binary classification ──")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_prob_bin = np.zeros(len(df))

for fold, (tr, te) in enumerate(skf.split(X, y_bin)):
    clf = make_clf()
    clf.fit(X[tr], y_bin[tr])
    cv_prob_bin[te] = clf.predict_proba(X[te])[:, 1]
    print(f"  fold {fold+1}/{N_FOLDS}")

auc_bin = roc_auc_score(y_bin, cv_prob_bin)
print(f"Binary CV AUC: {auc_bin:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — 3-class: early / medium / late (productive cells only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Task 3: 3-class early/medium/late ──")

skf3 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_pred_cat = np.zeros(len(prod_idx), dtype=int)
cv_prob_cat = np.zeros((len(prod_idx), 3))

for fold, (tr, te) in enumerate(skf3.split(X_prod, y_cat)):
    clf3 = make_clf()
    clf3.fit(X_prod[tr], y_cat[tr])
    cv_pred_cat[te]  = clf3.predict(X_prod[te])
    cv_prob_cat[te]  = clf3.predict_proba(X_prod[te])
    print(f"  fold {fold+1}/{N_FOLDS}")

acc_cat  = accuracy_score(y_cat, cv_pred_cat)
chance   = np.bincount(y_cat).max() / len(y_cat)
conf_mat = confusion_matrix(y_cat, cv_pred_cat)

print(f"3-class CV accuracy: {100*acc_cat:.1f}%  (chance={100*chance:.1f}%)")
print(pd.DataFrame(conf_mat,
                   index=[f"true_{c}" for c in CAT_NAMES],
                   columns=[f"pred_{c}" for c in CAT_NAMES]))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    "TabPFN v2 — HCMV live imaging  (A2+A3, GFP onset ≤ first half of movie)\n"
    f"n={len(df)} cells total  |  {N_FOLDS}-fold CV  |  n_estimators={N_EST}",
    fontsize=12, fontweight="bold"
)

# ── panel 1: regression scatter (productive only) ─────────────────────────────
ax = axes[0]
ax.scatter(cv_pred_reg, delay_prod,
           c="steelblue", alpha=0.45, s=14,
           label=f"Productive cells (n={len(prod_idx)})")
lo = min(cv_pred_reg.min(), delay_prod.min())
hi = max(cv_pred_reg.max(), delay_prod.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("CV predicted delay (min)")
ax.set_ylabel("True delay (min)")
ax.set_title(
    f"Task 1 — Regression (productive only)\n"
    f"R²={r2_prod:.3f}   Spearman ρ={rho_prod:.3f}\n"
    f"n={len(prod_idx)}  |  each dot = held-out test cell"
)
ax.legend(fontsize=8, loc="upper left")

# ── panel 2: ROC curve ────────────────────────────────────────────────────────
ax = axes[1]
fpr, tpr, _ = roc_curve(y_bin, cv_prob_bin)
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_bin:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Task 2 — Binary\nProductive vs non-productive\n{N_FOLDS}-fold CV AUC = {auc_bin:.3f}")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── panel 3: confusion matrix ─────────────────────────────────────────────────
ax = axes[2]
cm_norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(CAT_NAMES)
ax.set_yticks(range(3)); ax.set_yticklabels(CAT_NAMES)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(
    f"Task 3 — 3-class (early/medium/late)\n"
    f"GMM cutoffs: {CUT_EARLY_MED} / {CUT_MED_LATE} min\n"
    f"{N_FOLDS}-fold CV acc = {100*acc_cat:.1f}%  (chance = {100*chance:.1f}%)"
)
for i in range(3):
    for j in range(3):
        ax.text(j, i,
                f"{100*cm_norm[i,j]:.0f}%\n(n={conf_mat[i,j]})",
                ha="center", va="center", fontsize=9,
                color="white" if cm_norm[i, j] > 0.55 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out_path = FIG_DIR / "tabpfn_results.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS AS TABLES
# ══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = BASE / "results" / "tabpfn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Summary metrics
summary = pd.DataFrame([
    {"task": "Regression (productive only)", "metric": "R²",           "value": round(r2_prod,  3)},
    {"task": "Regression (productive only)", "metric": "Spearman rho", "value": round(rho_prod, 3)},
    {"task": "Binary (productive vs not)",   "metric": "AUC-ROC",      "value": round(auc_bin,  3)},
    {"task": "3-class (early/med/late)",     "metric": "Accuracy",     "value": round(acc_cat,  3)},
    {"task": "3-class (early/med/late)",     "metric": "Chance",       "value": round(chance,   3)},
])
summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

# 2. Confusion matrix (counts)
cm_df = pd.DataFrame(conf_mat,
                     index=[f"true_{c}"  for c in CAT_NAMES],
                     columns=[f"pred_{c}" for c in CAT_NAMES])
cm_df.to_csv(RESULTS_DIR / "confusion_matrix_counts.csv")

# 3. Confusion matrix (% of true class)
cm_pct = (cm_df.div(cm_df.sum(axis=1), axis=0) * 100).round(1)
cm_pct.to_csv(RESULTS_DIR / "confusion_matrix_pct.csv")

# 4. Per-cell predictions
cell_results = df[["Track.ID", "dataset", "green_onset_min", "delay_green_to_red"]].copy()
cell_results["productive"]        = productive
cell_results["pred_productive"]   = (cv_prob_bin >= 0.5).astype(int)
cell_results["prob_productive"]   = cv_prob_bin.round(4)

pred_delay_col = np.full(len(df), np.nan)
pred_delay_col[prod_idx] = cv_pred_reg
cell_results["pred_delay"] = pred_delay_col

cat_col = np.full(len(df), np.nan, dtype=object)
cat_col[prod_idx] = [CAT_NAMES[c] for c in y_cat]
pred_cat_col = np.full(len(df), np.nan, dtype=object)
pred_cat_col[prod_idx] = [CAT_NAMES[c] for c in cv_pred_cat]
prob_cat = np.full((len(df), 3), np.nan)
prob_cat[prod_idx] = cv_prob_cat

cell_results["true_category"]     = cat_col
cell_results["pred_category"]     = pred_cat_col
for i, name in enumerate(CAT_NAMES):
    cell_results[f"prob_{name}"]  = prob_cat[:, i].round(4)

cell_results.to_csv(RESULTS_DIR / "per_cell_predictions.csv", index=False)

# 5. ROC curve points
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_df.to_csv(RESULTS_DIR / "roc_curve.csv", index=False)

print(f"\nResults tables saved to results/tabpfn/:")
print(f"  summary_metrics.csv          — top-line metrics for all three tasks")
print(f"  confusion_matrix_counts.csv  — 3-class confusion matrix (n)")
print(f"  confusion_matrix_pct.csv     — 3-class confusion matrix (% of true class)")
print(f"  per_cell_predictions.csv     — per-cell predicted delays, probs, categories")
print(f"  roc_curve.csv                — ROC curve points for binary task")

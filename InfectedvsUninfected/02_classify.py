#!/usr/bin/env python3
"""
02_classify.py

Binary classification: infected (label=1) vs uninfected (label=0) cells,
using 256-dim Cellpose SAM BF embeddings produced by 01_extract_embeddings.py.

Pass A: all 256 embedding dimensions
  - Stratified 5-fold CV with logistic regression (elastic net)
  - Standardize features per fold (fit on train, apply to val)
  - Report ROC-AUC, PR-AUC, confusion matrix at Youden threshold
  - Save mean |beta| per feature -> top20_features.csv

Pass B: top-20 features by mean |beta| from Pass A
  - Same CV protocol, report same metrics

Null control: shuffled labels -> should give AUC ~0.50

Outputs:
  results/pass_A_metrics.csv
  results/pass_B_metrics.csv
  results/top20_features.csv
  figures/roc_pass_A.png
  figures/roc_pass_B.png
  figures/beta_barplot.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, confusion_matrix)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent
EMB_CSV = BASE / 'embeddings' / 'A2_infected_vs_uninfected.csv'
RES_DIR = BASE / 'results'
FIG_DIR = BASE / 'figures'

# ── Parameters ─────────────────────────────────────────────────────────────────
N_FOLDS          = 5
L1_RATIOS        = [0.1, 0.5, 0.9]
MAX_ITER         = 5000
RANDOM_SEED      = 42
TOP_N            = 20
N_UNINFECTED_SUB = 500   # subsample uninfected to this many cells

# ── Load embeddings ────────────────────────────────────────────────────────────
print('Loading embeddings...', flush=True)
df = pd.read_csv(EMB_CSV)
emb_cols = [c for c in df.columns if c.startswith('emb_')]
X_all = df[emb_cols].values.astype(np.float32)
y     = df['label'].values.astype(int)

print(f'  Before subsampling: {len(y)} cells  |  '
      f'infected: {(y==1).sum()}  |  uninfected: {(y==0).sum()}', flush=True)

# ── Subsample uninfected to N_UNINFECTED_SUB ───────────────────────────────────
rng_sub = np.random.default_rng(RANDOM_SEED)
uninf_idx = np.where(y == 0)[0]
inf_idx   = np.where(y == 1)[0]
n_sub     = min(N_UNINFECTED_SUB, len(uninf_idx))
uninf_sub = rng_sub.choice(uninf_idx, size=n_sub, replace=False)
keep_idx  = np.concatenate([inf_idx, uninf_sub])
keep_idx  = np.sort(keep_idx)
X_all     = X_all[keep_idx]
y         = y[keep_idx]

n_inf   = int((y == 1).sum())
n_uninf = int((y == 0).sum())
print(f'  After subsampling:  {len(y)} cells  |  '
      f'infected: {n_inf}  |  uninfected: {n_uninf}', flush=True)
print(f'  class_weight=balanced will use ratio {n_uninf/n_inf:.2f}:1 '
      f'(uninfected:infected)', flush=True)
print(f'  Features: {X_all.shape[1]}', flush=True)

# ── CV helper ──────────────────────────────────────────────────────────────────
def run_cv(X, y, label='pass', rng_seed=RANDOM_SEED):
    """
    Run stratified 5-fold CV with LogisticRegressionCV (elastic net).
    Returns dict with per-fold metrics, mean betas (N_features,), and
    out-of-fold predicted probabilities.
    """
    cv_outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=rng_seed)
    fold_aucs  = []
    fold_pr    = []
    all_betas  = []
    oof_probs  = np.zeros(len(y))
    oof_preds  = np.zeros(len(y), dtype=int)

    for fold_i, (train_idx, val_idx) in enumerate(cv_outer.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        clf = LogisticRegressionCV(
            penalty='elasticnet',
            l1_ratios=L1_RATIOS,
            solver='saga',
            class_weight='balanced',
            max_iter=MAX_ITER,
            cv=3,
            scoring='roc_auc',
            random_state=rng_seed,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)

        prob_val = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = prob_val

        auc = roc_auc_score(y_val, prob_val)
        pr  = average_precision_score(y_val, prob_val)
        fold_aucs.append(auc)
        fold_pr.append(pr)
        all_betas.append(clf.coef_[0])

        # Youden threshold on this fold
        fpr, tpr, thresholds = roc_curve(y_val, prob_val)
        youden_idx = np.argmax(tpr - fpr)
        thresh     = thresholds[youden_idx]
        oof_preds[val_idx] = (prob_val >= thresh).astype(int)

        print(f'  Fold {fold_i+1}: AUC={auc:.3f}  PR-AUC={pr:.3f}  '
              f'best_C={clf.C_[0]:.4f}  l1_ratio={clf.l1_ratio_[0]:.2f}  '
              f'[n_train={len(train_idx)} n_val={len(val_idx)}]', flush=True)

    mean_betas = np.mean(all_betas, axis=0)  # (n_features,)

    # Overall OOF metrics
    oof_auc = roc_auc_score(y, oof_probs)
    oof_pr  = average_precision_score(y, oof_probs)

    # Youden threshold on full OOF
    fpr_full, tpr_full, thresh_full = roc_curve(y, oof_probs)
    youden_full = np.argmax(tpr_full - fpr_full)
    thresh_oof  = thresh_full[youden_full]
    oof_pred_final = (oof_probs >= thresh_oof).astype(int)
    cm = confusion_matrix(y, oof_pred_final)

    print(f'  OOF AUC={oof_auc:.3f}  PR-AUC={oof_pr:.3f}  '
          f'(Youden thresh={thresh_oof:.3f})')
    print(f'  OOF confusion matrix (rows=true, cols=pred):\n{cm}')

    metrics_rows = []
    for i, (a, p) in enumerate(zip(fold_aucs, fold_pr)):
        metrics_rows.append({'fold': i+1, 'roc_auc': a, 'pr_auc': p})
    metrics_rows.append({'fold': 'OOF_mean',
                         'roc_auc': oof_auc, 'pr_auc': oof_pr})

    return {
        'metrics':    pd.DataFrame(metrics_rows),
        'mean_betas': mean_betas,
        'oof_probs':  oof_probs,
        'roc_fpr':    fpr_full,
        'roc_tpr':    tpr_full,
        'oof_auc':    oof_auc,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PASS A: all 256 features
# ══════════════════════════════════════════════════════════════════════════════

print('\n=== PASS A: all 256 features ===')
res_A = run_cv(X_all, y, label='A')

res_A['metrics'].to_csv(RES_DIR / 'pass_A_metrics.csv', index=False)
print(f"Saved: {RES_DIR / 'pass_A_metrics.csv'}")

# Top-20 features by mean |beta|
abs_betas = np.abs(res_A['mean_betas'])
top20_idx = np.argsort(abs_betas)[::-1][:TOP_N]
top20_df  = pd.DataFrame({
    'feature_index': top20_idx,
    'feature_name':  [emb_cols[i] for i in top20_idx],
    'mean_abs_beta': abs_betas[top20_idx],
    'mean_beta':     res_A['mean_betas'][top20_idx],
})
top20_df.to_csv(RES_DIR / 'top20_features.csv', index=False)
print(f"Saved: {RES_DIR / 'top20_features.csv'}")
print(f"  Top 5 features: {top20_df['feature_name'].tolist()[:5]}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS B: top-20 features
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n=== PASS B: top-{TOP_N} features by |beta| from Pass A ===')
X_top20 = X_all[:, top20_idx]
res_B = run_cv(X_top20, y, label='B')

res_B['metrics'].to_csv(RES_DIR / 'pass_B_metrics.csv', index=False)
print(f"Saved: {RES_DIR / 'pass_B_metrics.csv'}")

# ══════════════════════════════════════════════════════════════════════════════
# NULL CONTROL: shuffled labels
# ══════════════════════════════════════════════════════════════════════════════

print('\n=== NULL CONTROL: shuffled labels (pass A) ===')
rng_null = np.random.default_rng(RANDOM_SEED + 99)
y_shuffled = y.copy()
rng_null.shuffle(y_shuffled)
res_null = run_cv(X_all, y_shuffled, label='null', rng_seed=RANDOM_SEED + 99)
null_auc = res_null['oof_auc']
if null_auc > 0.55:
    print(f'  WARNING: null AUC={null_auc:.3f} > 0.55 — check for data leakage!')
else:
    print(f'  Null AUC={null_auc:.3f} OK (< 0.55)')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ROC curves
for res, label, out_name in [
        (res_A, f'Pass A — all 256 (AUC={res_A["oof_auc"]:.3f})',
         'roc_pass_A.png'),
        (res_B, f'Pass B — top-{TOP_N} (AUC={res_B["oof_auc"]:.3f})',
         'roc_pass_B.png')]:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(res['roc_fpr'], res['roc_tpr'], lw=2,
            label=label)
    ax.plot(res_null['roc_fpr'], res_null['roc_tpr'], lw=1.5,
            linestyle='--', color='gray',
            label=f'Null shuffle (AUC={res_null["oof_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Infected vs Uninfected — BF embedding')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / out_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIG_DIR / out_name}")

# Beta barplot (top-20 from Pass A)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#d62728' if b > 0 else '#1f77b4'
          for b in top20_df['mean_beta']]
ax.barh(top20_df['feature_name'][::-1],
        top20_df['mean_abs_beta'][::-1],
        color=colors[::-1])
ax.set_xlabel('Mean |beta| across folds')
ax.set_title(f'Top-{TOP_N} embedding dimensions by logistic regression |beta|')
plt.tight_layout()
fig.savefig(str(FIG_DIR / 'beta_barplot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {FIG_DIR / 'beta_barplot.png'}")

# Summary
print('\n=== Summary ===')
print(f'  Pass A  AUC: {res_A["oof_auc"]:.3f}')
print(f'  Pass B  AUC: {res_B["oof_auc"]:.3f}')
print(f'  Null    AUC: {null_auc:.3f}')
print('\nDone.', flush=True)

#!/usr/bin/env python3
"""
04_bf_classify_stratified.py

Focused classification analysis addressing class imbalance (22 early / 145 med+late).

Three strategies compared on the same 5-fold CV:
  A) Balanced class weights (current approach, threshold=0.5)
  B) Balanced class weights + optimal threshold (maximise F1 on OOF)
  C) Random undersampling of majority in training, threshold=0.5

Metrics reported at each strategy's threshold:
  AUC, Precision, Recall (=Sensitivity), Specificity, F1, Accuracy, MCC

Also produces:
  - Precision-Recall curve (more informative than ROC for imbalanced data)
  - ROC curve
  - Predicted probability histograms: early vs non-early
  - Threshold vs Precision/Recall/F1 sweep

Outputs:
  figures/classify_pr_curve.png
  figures/classify_roc_curve.png
  figures/classify_prob_hist.png
  figures/classify_threshold_sweep.png
  results/classify_stratified_metrics.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, matthews_corrcoef,
                             balanced_accuracy_score)
from sklearn.utils import resample
warnings.filterwarnings('ignore')

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ     = BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
TOP20_CSV   = BASE / 'results' / 'top20_dims.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'

CUT_EARLY = 911
SEED      = 42

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['gfp_track_ids']
embeddings    = d['embeddings'].astype(np.float64)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}

top20_dims = pd.read_csv(TOP20_CSV)['dim'].tolist()
print(f'  Top-20 dims: {top20_dims}')

df = pd.read_csv(MODEL_DF)
df_cls = df[df['abs_gfp_onset_min'] <= df['movie_half_min']].copy()
df_cls = df_cls[df_cls['dataset'] == 'A2'].copy()
df_cls['track_id'] = df_cls['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_cls = df_cls[np.isfinite(df_cls['delay_green_to_red'])].copy()
df_cls = df_cls[df_cls['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows   = [emb_id_to_row[tid] for tid in df_cls['track_id']]

X = embeddings[rows][:, top20_dims]
y = (df_cls['delay_green_to_red'].values <= CUT_EARLY).astype(int)
n_early = y.sum()
n_total = len(y)
print(f'  n={n_total}  early={n_early}  non-early={n_total-n_early}  '
      f'base_rate={n_early/n_total:.3f}')

# ── CV setup ───────────────────────────────────────────────────────────────────
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

LR_PARAMS = dict(
    penalty='elasticnet', solver='saga',
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    Cs=np.logspace(-3, 1, 20),
    class_weight='balanced', scoring='roc_auc',
    max_iter=2000, random_state=SEED, n_jobs=-1,
)

# ── Strategy A + B: balanced class weights, save OOF probs ────────────────────
print('\nStrategy A/B — balanced class weights...', flush=True)
oof_balanced = np.zeros(n_total)
for tr, te in outer.split(X, y):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    Xte = sc.transform(X[te])
    m   = LogisticRegressionCV(**{**LR_PARAMS, 'cv': inner})
    m.fit(Xtr, y[tr])
    oof_balanced[te] = m.predict_proba(Xte)[:, 1]

auc_bal = roc_auc_score(y, oof_balanced)
ap_bal  = average_precision_score(y, oof_balanced)
print(f'  AUC={auc_bal:.3f}  AP={ap_bal:.3f}')

# Threshold sweep to find optimal F1
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores  = [f1_score(y, (oof_balanced >= t).astype(int), zero_division=0)
              for t in thresholds]
opt_thresh_bal = thresholds[np.argmax(f1_scores)]
print(f'  Optimal threshold (F1): {opt_thresh_bal:.3f}')

def metrics_at_threshold(y_true, probs, thresh, label):
    pred = (probs >= thresh).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum())
    fp = int(((pred==1)&(y_true==0)).sum())
    tn = int(((pred==0)&(y_true==0)).sum())
    fn = int(((pred==0)&(y_true==1)).sum())
    prec  = tp/(tp+fp) if (tp+fp)>0 else 0
    rec   = tp/(tp+fn) if (tp+fn)>0 else 0
    spec  = tn/(tn+fp) if (tn+fp)>0 else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    acc   = (tp+tn)/len(y_true)
    mcc   = matthews_corrcoef(y_true, pred)
    bal   = balanced_accuracy_score(y_true, pred)
    auc   = roc_auc_score(y_true, probs)
    ap    = average_precision_score(y_true, probs)
    print(f'  [{label}] thresh={thresh:.3f}  '
          f'Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  '
          f'Acc={acc:.3f}  Spec={spec:.3f}  MCC={mcc:.3f}  BalAcc={bal:.3f}')
    return dict(strategy=label, threshold=round(thresh,3), auc=round(auc,3),
                ap=round(ap,3), precision=round(prec,3), recall=round(rec,3),
                specificity=round(spec,3), f1=round(f1,3),
                accuracy=round(acc,3), mcc=round(mcc,3), bal_acc=round(bal,3),
                tp=tp, fp=fp, tn=tn, fn=fn)

print('\nMetrics:')
row_a = metrics_at_threshold(y, oof_balanced, 0.5,           'A: balanced_weights thresh=0.5')
row_b = metrics_at_threshold(y, oof_balanced, opt_thresh_bal, 'B: balanced_weights opt_thresh')

# ── Strategy C: random undersampling of majority in training ──────────────────
print('\nStrategy C — undersampling majority in training...', flush=True)
oof_under = np.zeros(n_total)
rng = np.random.default_rng(SEED)

LR_PARAMS_UNDER = {**LR_PARAMS, 'class_weight': None}  # no class weights needed

for fold_i, (tr, te) in enumerate(outer.split(X, y)):
    X_tr, y_tr = X[tr], y[tr]

    # undersample majority to match minority count
    pos_idx = tr[y_tr == 1]
    neg_idx = tr[y_tr == 0]
    neg_idx_down = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    tr_balanced  = np.concatenate([pos_idx, neg_idx_down])
    rng.shuffle(tr_balanced)

    sc  = StandardScaler()
    sc.fit(X[tr])                    # fit scaler on full training fold
    Xtr = sc.transform(X[tr_balanced])
    Xte = sc.transform(X[te])
    y_tr_bal = y[tr_balanced]

    m = LogisticRegressionCV(**{**LR_PARAMS_UNDER, 'cv':
            StratifiedKFold(n_splits=min(5, len(pos_idx)), shuffle=True,
                            random_state=SEED + fold_i)})
    m.fit(Xtr, y_tr_bal)
    oof_under[te] = m.predict_proba(Xte)[:, 1]

print('\nMetrics:')
row_c = metrics_at_threshold(y, oof_under, 0.5, 'C: undersampling thresh=0.5')

# ── Save metrics ───────────────────────────────────────────────────────────────
results = pd.DataFrame([row_a, row_b, row_c])
results.to_csv(RESULTS_DIR / 'classify_stratified_metrics.csv', index=False)
print(f'\nSaved results/classify_stratified_metrics.csv')

# ── Precision-Recall curve ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
base_rate = n_early / n_total

for probs, label, colour in [
    (oof_balanced, f'Balanced weights (AP={average_precision_score(y,oof_balanced):.3f})', '#9C27B0'),
    (oof_under,    f'Undersampling    (AP={average_precision_score(y,oof_under):.3f})',    '#FF5722'),
]:
    prec_c, rec_c, _ = precision_recall_curve(y, probs)
    ax.plot(rec_c, prec_c, lw=2, color=colour, label=label)

ax.axhline(base_rate, color='gray', lw=1, ls='--',
           label=f'Random baseline (precision={base_rate:.2f})')
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall curve — BF top-20 dims\n(n={n_total}, {n_early} early)')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'classify_pr_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/classify_pr_curve.png')

# ── ROC curve ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
for probs, label, colour in [
    (oof_balanced, f'Balanced weights (AUC={roc_auc_score(y,oof_balanced):.3f})', '#9C27B0'),
    (oof_under,    f'Undersampling    (AUC={roc_auc_score(y,oof_under):.3f})',    '#FF5722'),
]:
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, lw=2, color=colour, label=label)

ax.axhline(0.742, color='#FF9800', lw=1.5, ls='--', label='GFP onset top-20  AUC=0.742')
ax.axhline(0.684, color='#2196F3', lw=1,   ls=':',  label='Tabular (A2)  AUC=0.684')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random  AUC=0.500')
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'ROC — BF top-20 dims\n(n={n_total}, {n_early} early)')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'classify_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/classify_roc_curve.png')

# ── Probability histogram ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, probs, label in [
    (axes[0], oof_balanced, 'Balanced weights'),
    (axes[1], oof_under,    'Undersampling'),
]:
    ax.hist(probs[y==0], bins=25, alpha=0.6, color='#2196F3', label='Non-early', density=True)
    ax.hist(probs[y==1], bins=25, alpha=0.7, color='#F44336', label='Early',     density=True)
    ax.set_xlabel('Predicted probability of early')
    ax.set_ylabel('Density')
    ax.set_title(label)
    ax.legend(fontsize=9)
fig.suptitle('Predicted probability distributions — BF top-20 dims', fontsize=11)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'classify_prob_hist.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/classify_prob_hist.png')

# ── Threshold sweep ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, probs, label in [
    (axes[0], oof_balanced, 'Balanced weights'),
    (axes[1], oof_under,    'Undersampling'),
]:
    ts = np.linspace(0.01, 0.99, 200)
    precs = []; recs = []; f1s = []; accs = []
    for t in ts:
        p = (probs >= t).astype(int)
        precs.append(f1_score(y, p, average=None, zero_division=0)[1]
                     if p.sum()>0 else 0)
        recs.append(((p==1)&(y==1)).sum() / y.sum())
        f1s.append(f1_score(y, p, zero_division=0))
        accs.append(accuracy_score(y, p))
    opt_t = ts[np.argmax(f1s)]
    ax.plot(ts, precs, label='Precision', color='#E91E63')
    ax.plot(ts, recs,  label='Recall',    color='#2196F3')
    ax.plot(ts, f1s,   label='F1',        color='#4CAF50', lw=2)
    ax.axvline(opt_t, color='gray', ls='--', lw=1, label=f'Opt threshold={opt_t:.2f}')
    ax.axvline(0.5,   color='black', ls=':', lw=1, label='threshold=0.50')
    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'{label}')
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
fig.suptitle('Threshold sweep — BF top-20 dims (early vs rest)', fontsize=11)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'classify_threshold_sweep.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/classify_threshold_sweep.png')

# ── Summary table ──────────────────────────────────────────────────────────────
print('\n── Summary ──')
print(f'  Base rate (early): {n_early}/{n_total} = {n_early/n_total:.1%}')
print(f'  {"Strategy":<40}  {"Prec":>6}  {"Rec":>6}  {"F1":>6}  {"Acc":>6}  {"AUC":>6}')
for row in [row_a, row_b, row_c]:
    print(f'  {row["strategy"]:<40}  {row["precision"]:>6.3f}  '
          f'{row["recall"]:>6.3f}  {row["f1"]:>6.3f}  '
          f'{row["accuracy"]:>6.3f}  {row["auc"]:>6.3f}')
print(f'\n  {"Random baseline (predict all negative)":<40}  {"0.000":>6}  {"0.000":>6}  '
      f'{"0.000":>6}  {1-n_early/n_total:>6.3f}')

print('\nDone.', flush=True)

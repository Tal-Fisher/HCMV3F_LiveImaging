#!/usr/bin/env python3
"""
05_bf_comparison_figure.py

Comparison figure: BF top-20 embeddings vs tabular features vs random baseline.
Target: delay_blue_to_red (BFP→mCherry). No half-movie filter.

Strategy: balanced class weights, threshold optimised for F1 on OOF predictions.
Metrics reported: AUC, Precision, Recall, F1, Accuracy.
Random baseline: 100 label-permutation runs → mean ± std.

Output:
  figures/comparison_figure.png
  results/comparison_metrics.csv
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
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             average_precision_score, precision_recall_curve)
warnings.filterwarnings('ignore')

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ   = BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
MODEL_DF  = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
F16_CSV   = LIVEIMG / 'cache' / 'python_export' / 'frame16_features.csv'
TOP20_CSV = BASE / 'results' / 'top20_dims.csv'
RESULTS   = BASE / 'results'
FIGURES   = BASE / 'figures'

CUT_EARLY  = 1094  # GMM Bayes-optimal cutoff for delay_blue_to_red (A2+A3)
SEED       = 42
N_PERM     = 20

# ── Load shared cell set ───────────────────────────────────────────────────────
print('Building BF-matched dataset...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['gfp_track_ids']
embeddings    = d['embeddings'].astype(np.float64)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
top20_dims    = pd.read_csv(TOP20_CSV)['dim'].tolist()

df = pd.read_csv(MODEL_DF)
f16 = pd.read_csv(F16_CSV)
df = df.merge(f16, on='Track.ID', how='left')
df['delay_blue_to_red'] = df['delay_green_to_red'] - df['delay_green_to_blue']

df_cls = df[df['dataset'] == 'A2'].copy()
df_cls['track_id'] = df_cls['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_cls = df_cls[np.isfinite(df_cls['delay_blue_to_red'])].copy()
df_cls = df_cls[df_cls['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows = [emb_id_to_row[tid] for tid in df_cls['track_id']]

y = (df_cls['delay_blue_to_red'].values <= CUT_EARLY).astype(int)
n_total, n_early = len(y), int(y.sum())
base_rate = n_early / n_total
print(f'  n={n_total}  early={n_early}  base_rate={base_rate:.3f}')

NON_FEAT = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
            "delay_blue_to_red","green_onset_min","track_start_min","abs_gfp_onset_min",
            "movie_half_min","y","gfp_snr_mean","bf_snr_mean","track_id"}
feat_cols = [c for c in df_cls.columns if c not in NON_FEAT]
X_tab = df_cls[feat_cols].values.astype(float)
X_bf  = embeddings[rows][:, top20_dims]
print(f'  Tabular features: {len(feat_cols)}  |  BF dims: {len(top20_dims)}')

# ── CV runner ──────────────────────────────────────────────────────────────────
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

LR = dict(penalty='elasticnet', solver='saga',
          l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
          Cs=np.logspace(-3, 1, 20),
          class_weight='balanced', scoring='roc_auc',
          max_iter=2000, random_state=SEED, n_jobs=-1, cv=inner)

def run_cv(X, y_true, seed_offset=0):
    oof = np.zeros(len(y_true))
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + seed_offset)
    for tr, te in cv.split(X, y_true):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR)
        m.fit(sc.fit_transform(X[tr]), y_true[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof

def best_threshold_metrics(y_true, probs, label):
    ts   = np.linspace(0.01, 0.99, 300)
    f1s  = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in ts]
    opt  = ts[np.argmax(f1s)]
    pred = (probs >= opt).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum())
    fp = int(((pred==1)&(y_true==0)).sum())
    fn = int(((pred==0)&(y_true==1)).sum())
    tn = int(((pred==0)&(y_true==0)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    acc  = (tp+tn)/len(y_true)
    auc  = roc_auc_score(y_true, probs)
    ap   = average_precision_score(y_true, probs)
    print(f'  [{label}] thresh={opt:.3f}  AUC={auc:.3f}  AP={ap:.3f}  '
          f'Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Acc={acc:.3f}')
    return dict(label=label, auc=auc, ap=ap, precision=prec, recall=rec,
                f1=f1, accuracy=acc, threshold=opt)

# ── BF embeddings ──────────────────────────────────────────────────────────────
print('\nBF top-20 embeddings...', flush=True)
oof_bf = run_cv(X_bf, y)
m_bf   = best_threshold_metrics(y, oof_bf, 'BF top-20')

# ── Tabular features — train on full A2 (no half-movie filter), evaluate on BF subset ──
print('\nTabular 33 features (full A2 CV, no half-movie filter)...', flush=True)

df_a2_full = df[(df['dataset']=='A2') &
                np.isfinite(df['delay_blue_to_red'])].copy()
df_a2_full['track_id'] = df_a2_full['Track.ID'].str.replace('A2_','',regex=False).astype(int)
df_a2_full = df_a2_full.sort_values('track_id').reset_index(drop=True)

feat_cols_full = [c for c in df_a2_full.columns if c not in NON_FEAT]
X_tab_full = df_a2_full[feat_cols_full].values.astype(float)
y_full     = (df_a2_full['delay_blue_to_red'].values <= CUT_EARLY).astype(int)

# impute missing with column median
col_med = np.nanmedian(X_tab_full, axis=0)
for j in range(X_tab_full.shape[1]):
    bad = ~np.isfinite(X_tab_full[:, j])
    X_tab_full[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

print(f'  Full A2: n={len(y_full)}, early={y_full.sum()}')

# OOF on all 274
oof_tab_full = run_cv(X_tab_full, y_full)

# Subset OOF to the 167 BF-matched cells
bf_track_set = set(df_cls['track_id'].values)
bf_mask      = df_a2_full['track_id'].isin(bf_track_set).values
oof_tab      = oof_tab_full[bf_mask]   # 167 OOF probs
y_tab_sub    = y_full[bf_mask]         # should equal y

assert np.array_equal(y_tab_sub, y), "y mismatch between BF subset and full A2 selection"
print(f'  BF subset extracted: n={bf_mask.sum()}, early={y_tab_sub.sum()}')

m_tab = best_threshold_metrics(y, oof_tab, 'Tabular 33 b2r (A2 full CV)')

# ── Random baseline (label permutation) ────────────────────────────────────────
print(f'\nRandom baseline ({N_PERM} permutations)...', flush=True)
rng = np.random.default_rng(SEED)
perm_aucs, perm_precs, perm_recs, perm_f1s, perm_accs = [], [], [], [], []

for i in range(N_PERM):
    y_perm = rng.permutation(y)
    oof_r  = run_cv(X_bf, y_perm, seed_offset=i+100)
    ts     = np.linspace(0.01, 0.99, 100)
    f1s_r  = [f1_score(y_perm, (oof_r >= t).astype(int), zero_division=0) for t in ts]
    opt_r  = ts[np.argmax(f1s_r)]
    pred_r = (oof_r >= opt_r).astype(int)
    tp = ((pred_r==1)&(y_perm==1)).sum()
    fp = ((pred_r==1)&(y_perm==0)).sum()
    fn = ((pred_r==0)&(y_perm==1)).sum()
    tn = ((pred_r==0)&(y_perm==0)).sum()
    prec_r = tp/(tp+fp) if (tp+fp)>0 else 0
    rec_r  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1_r   = 2*prec_r*rec_r/(prec_r+rec_r) if (prec_r+rec_r)>0 else 0
    perm_aucs.append(roc_auc_score(y_perm, oof_r))
    perm_precs.append(prec_r)
    perm_recs.append(rec_r)
    perm_f1s.append(f1_r)
    perm_accs.append((tp+tn)/len(y_perm))
    if (i+1) % 20 == 0:
        print(f'  {i+1}/{N_PERM}', flush=True)

m_rand = dict(label='Random', auc=np.mean(perm_aucs), ap=base_rate,
              precision=np.mean(perm_precs), recall=np.mean(perm_recs),
              f1=np.mean(perm_f1s), accuracy=np.mean(perm_accs), threshold=np.nan)
m_rand_std = dict(auc=np.std(perm_aucs), precision=np.std(perm_precs),
                  recall=np.std(perm_recs), f1=np.std(perm_f1s),
                  accuracy=np.std(perm_accs))
print(f'  [Random mean] AUC={m_rand["auc"]:.3f}±{m_rand_std["auc"]:.3f}  '
      f'Prec={m_rand["precision"]:.3f}±{m_rand_std["precision"]:.3f}  '
      f'Rec={m_rand["recall"]:.3f}±{m_rand_std["recall"]:.3f}  '
      f'Acc={m_rand["accuracy"]:.3f}±{m_rand_std["accuracy"]:.3f}')

# ── Save metrics ───────────────────────────────────────────────────────────────
rows_out = []
for m in [m_bf, m_tab, m_rand]:
    row = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
    rows_out.append(row)
pd.DataFrame(rows_out).to_csv(RESULTS / 'comparison_metrics.csv', index=False)

# ── Figure ─────────────────────────────────────────────────────────────────────
metrics_to_plot = ['AUC', 'Precision', 'Recall', 'F1', 'Accuracy']
keys            = ['auc', 'precision', 'recall', 'f1', 'accuracy']
stds_rand       = [m_rand_std['auc'], m_rand_std['precision'],
                   m_rand_std['recall'], m_rand_std['f1'], m_rand_std['accuracy']]

models = [
    ('Random',        m_rand, '#B0BEC5', stds_rand),
    ('Tabular\n33 feat (b2r)', m_tab,  '#2196F3', [0]*5),
    ('BF embeddings\ntop-20 dims', m_bf, '#9C27B0', [0]*5),
]

x     = np.arange(len(metrics_to_plot))
width = 0.22
n_mod = len(models)
offsets = np.linspace(-(n_mod-1)/2, (n_mod-1)/2, n_mod) * width

fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, m, colour, stds) in enumerate(models):
    vals = [m[k] for k in keys]
    bars = ax.bar(x + offsets[i], vals, width, label=name,
                  color=colour, alpha=0.85, edgecolor='white', linewidth=0.5)
    # error bars for random only
    if name == 'Random':
        ax.errorbar(x + offsets[i], vals, yerr=stds,
                    fmt='none', color='#546E7A', capsize=3, lw=1.2)
    # value labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

# Base rate line for reference on precision panel
ax.axhline(base_rate, color='#E53935', lw=1, ls='--', alpha=0.6,
           label=f'Base rate (precision if random = {base_rate:.2f})')

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title(
    f'BF embeddings vs tabular features vs random (target: delay BFP→mCherry)\n'
    f'(n={n_total}, {n_early} early cells, {n_total-n_early} non-early, '
    f'threshold optimised for F1)',
    fontsize=11
)
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(FIGURES / 'comparison_figure.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('\nSaved figures/comparison_figure.png')

# ── PR curves on same axes ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for probs, label, colour in [
    (oof_bf,  f'BF top-20  (AP={average_precision_score(y, oof_bf):.3f})',  '#9C27B0'),
    (oof_tab, f'Tabular 33 b2r (AP={average_precision_score(y, oof_tab):.3f})', '#2196F3'),
]:
    prec_c, rec_c, _ = precision_recall_curve(y, probs)
    ax.plot(rec_c, prec_c, lw=2, color=colour, label=label)

ax.axhline(base_rate, color='gray', lw=1, ls='--',
           label=f'Random (AP={base_rate:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall — BF embeddings vs tabular (b2r)\n(n={n_total}, {n_early} early)')
ax.legend(fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
fig.savefig(FIGURES / 'comparison_pr_curve.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('Saved figures/comparison_pr_curve.png')

print('\nDone.', flush=True)

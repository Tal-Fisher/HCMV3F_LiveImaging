#!/usr/bin/env python3
"""
03_classify_top20_cv.py

5-fold stratified CV classification of fast vs slow b2r outcome using
top-20 Cellpose embedding dims from three feature sets:

  A) GFP alone        — top-20 dims from GFP embedding at onset
  B) BF alone         — top-20 dims from BF embedding at onset
  C) GFP + BF concat  — top-20 GFP dims + top-20 BF dims (40-dim input)

Classifier: LogisticRegressionCV with ElasticNet penalty (two-layer linear model).
Dim selection: global |beta| rank averaged across ElasticNet regression and
               LogReg classification (fit once on full dataset before CV).

Label: fast = delay_blue_to_red <= 1094 min  (GMM Bayes-optimal cutoff)
Filters: finite b2r, half-movie filter, inner join on available embeddings.

Datasets: A2 (A3 added automatically once embeddings are available).

Outputs:
  results/classify_top20_metrics.csv
  figures/classify_top20_roc.png
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score,
                             balanced_accuracy_score,
                             matthews_corrcoef)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')
CPE     = LIVEIMG / 'CellposeEmbedding' / 'embeddings'

MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', default=['A2', 'A3'],
                    choices=['A2', 'A3'], help='Datasets to include (default: A2 A3)')
args = parser.parse_args()
DATASETS_REQUESTED = args.datasets

CUT_B2R  = 1094   # minutes — GMM Bayes-optimal cutoff
SEED     = 42
TOP_N    = 20
N_SPLITS = 5

LR_PARAMS = dict(
    penalty='elasticnet', solver='saga',
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    Cs=np.logspace(-3, 1, 20),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
    class_weight='balanced', scoring='roc_auc',
    max_iter=2000, random_state=SEED, n_jobs=-1,
)

# ── Load embeddings for all available datasets ─────────────────────────────────
print('Loading embeddings...', flush=True)
gfp_emb_files = {
    'A2': CPE / 'A2_cell_embeddings.npz',
    'A3': CPE / 'A3_cell_embeddings.npz',
}
bf_emb_files = {
    'A2': BASE / 'embeddings' / 'A2_bf_at_gfp_onset.npz',
    'A3': BASE / 'embeddings' / 'A3_bf_at_gfp_onset.npz',
}

gfp_ids_all, gfp_embs_all = [], []
bf_ids_all,  bf_embs_all  = [], []
datasets_used = []

for ds in DATASETS_REQUESTED:
    gfp_f = gfp_emb_files[ds]
    bf_f  = bf_emb_files[ds]
    if not gfp_f.exists() or not bf_f.exists():
        print(f'  {ds}: embeddings not yet available — skipping')
        continue
    d_gfp = np.load(str(gfp_f))
    d_bf  = np.load(str(bf_f))
    gfp_ids_all.append(pd.DataFrame({'track_id': d_gfp['track_ids'].astype(int), 'dataset': ds}))
    gfp_embs_all.append(d_gfp['embeddings'].astype(np.float32))
    bf_ids_all.append(pd.DataFrame({'track_id': d_bf['track_ids'].astype(int), 'dataset': ds}))
    bf_embs_all.append(d_bf['embeddings'].astype(np.float32))
    datasets_used.append(ds)
    print(f'  {ds}: GFP {d_gfp["embeddings"].shape}  BF {d_bf["embeddings"].shape}')

gfp_id_df = pd.concat(gfp_ids_all).reset_index(drop=True)
bf_id_df  = pd.concat(bf_ids_all).reset_index(drop=True)
GFP_EMB   = np.vstack(gfp_embs_all)
BF_EMB    = np.vstack(bf_embs_all)

# ── Load labels and apply filters ─────────────────────────────────────────────
print('Loading labels...', flush=True)
mdf = pd.read_csv(MODEL_DF)
rows = []
for ds in datasets_used:
    sub = mdf[mdf['dataset'] == ds].copy()
    sub['track_id'] = sub['Track.ID'].str.replace(f'{ds}_', '', regex=False).astype(int)
    sub['b2r']      = sub['delay_green_to_red'] - sub['delay_green_to_blue']
    rows.append(sub)
meta = pd.concat(rows).reset_index(drop=True)

gfp_key = gfp_id_df.set_index(['dataset', 'track_id']).index
bf_key  = bf_id_df.set_index(['dataset', 'track_id']).index
meta_key = pd.MultiIndex.from_arrays([meta['dataset'], meta['track_id']])

eligible = meta[
    meta_key.isin(gfp_key) &
    meta_key.isin(bf_key)  &
    meta['b2r'].notna()    &
    (meta['abs_gfp_onset_min'] <= meta['movie_half_min'])
].sort_values(['dataset', 'track_id']).reset_index(drop=True)

# Gather embedding rows in the same order as eligible
gfp_index = {(r.dataset, r.track_id): i for i, r in gfp_id_df.iterrows()}
bf_index  = {(r.dataset, r.track_id): i for i, r in bf_id_df.iterrows()}
gfp_rows  = [gfp_index[(r.dataset, r.track_id)] for _, r in eligible.iterrows()]
bf_rows   = [bf_index[(r.dataset, r.track_id)]  for _, r in eligible.iterrows()]

X_GFP = GFP_EMB[gfp_rows]   # (n, 256)
X_BF  = BF_EMB[bf_rows]      # (n, 256)
y     = (eligible['b2r'].values <= CUT_B2R).astype(int)
y_reg = eligible['b2r'].values.astype(float)

n       = len(y)
n_fast  = int(y.sum())
n_slow  = n - n_fast
print(f'  Datasets: {datasets_used}')
print(f'  n={n}  fast={n_fast}  slow={n_slow}  base_rate={n_fast/n:.3f}')

# ── Top-N dim selection (full-data, same method as existing pipeline) ──────────
def select_top_dims(X, y_reg, y_cls, label='', top_n=TOP_N):
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    en  = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                       cv=5, max_iter=10000, n_jobs=-1, random_state=SEED)
    lr  = LogisticRegressionCV(**LR_PARAMS)
    en.fit(Xs, y_reg)
    lr.fit(Xs, y_cls)
    rank_r   = pd.Series(np.abs(en.coef_),    index=np.arange(X.shape[1])).rank(ascending=False)
    rank_c   = pd.Series(np.abs(lr.coef_[0]), index=np.arange(X.shape[1])).rank(ascending=False)
    avg_rank = (rank_r + rank_c) / 2
    dims     = avg_rank.sort_values().head(top_n).index.tolist()
    if label:
        print(f'    {label} top-{top_n} dims: {dims}', flush=True)
    return dims

print('\nSelecting top dims (full-data fit)...', flush=True)
gfp_top = select_top_dims(X_GFP, y_reg, y, label='GFP')
bf_top  = select_top_dims(X_BF,  y_reg, y, label='BF ')

# ── 5-fold stratified LogRegCV ────────────────────────────────────────────────
def run_cv(X, y, label=''):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR_PARAMS)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    auc  = roc_auc_score(y, oof)
    ap   = average_precision_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    sens = ((pred == 1) & (y == 1)).sum() / max(y.sum(), 1)
    spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    bal  = balanced_accuracy_score(y, pred)
    mcc  = matthews_corrcoef(y, pred)
    print(f'  {label:<30}  AUC={auc:.3f}  AP={ap:.3f}  '
          f'Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}  MCC={mcc:.3f}', flush=True)
    return oof, dict(model=label, n=n, n_fast=n_fast, datasets='+'.join(datasets_used),
                     auc=round(auc, 3), ap=round(ap, 3),
                     sens=round(sens, 3), spec=round(spec, 3),
                     bal_acc=round(bal, 3), mcc=round(mcc, 3))

print('\n5-fold stratified CV results:', flush=True)
X_gfp_t  = X_GFP[:, gfp_top]
X_bf_t   = X_BF[:, bf_top]
X_concat = np.concatenate([X_gfp_t, X_bf_t], axis=1)

oof_gfp,    row_gfp    = run_cv(X_gfp_t,  y, label=f'GFP alone      (top-{TOP_N})')
oof_bf,     row_bf     = run_cv(X_bf_t,   y, label=f'BF alone       (top-{TOP_N})')
oof_concat, row_concat = run_cv(X_concat, y, label=f'GFP+BF concat  (top-{TOP_N*2})')

# ── Save results ───────────────────────────────────────────────────────────────
results = pd.DataFrame([row_gfp, row_bf, row_concat])
out_csv = RESULTS_DIR / 'classify_top20_metrics.csv'
results.to_csv(out_csv, index=False)
print(f'\nSaved {out_csv}')

# ── ROC comparison figure ──────────────────────────────────────────────────────
colours = {
    f'GFP alone (top-{TOP_N})':          ('#4CAF50', oof_gfp),
    f'BF alone (top-{TOP_N})':           ('#2196F3', oof_bf),
    f'GFP+BF concat (top-{TOP_N*2})':   ('#FF9800', oof_concat),
}

fig, ax = plt.subplots(figsize=(6, 6))
for label, (colour, oof) in colours.items():
    auc = roc_auc_score(y, oof)
    fpr, tpr, _ = roc_curve(y, oof)
    ax.plot(fpr, tpr, lw=2, color=colour, label=f'{label}  AUC={auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random  AUC=0.500')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title(
    f'GFP vs BF vs Fusion — b2r classification\n'
    f'Datasets: {"+".join(datasets_used)}  |  n={n}, {n_fast} fast  |  5-fold OOF',
    fontsize=10
)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
png_path = FIGURES_DIR / 'classify_top20_roc.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {png_path}')

# ── Summary table ──────────────────────────────────────────────────────────────
print(f'\n{"─"*75}')
print(f'  {"Model":<30}  {"AUC":>6}  {"AP":>6}  {"Sens":>6}  {"Spec":>6}  '
      f'{"BalAcc":>7}  {"MCC":>6}')
print(f'{"─"*75}')
for row in [row_gfp, row_bf, row_concat]:
    print(f'  {row["model"]:<30}  {row["auc"]:>6.3f}  {row["ap"]:>6.3f}  '
          f'{row["sens"]:>6.3f}  {row["spec"]:>6.3f}  '
          f'{row["bal_acc"]:>7.3f}  {row["mcc"]:>6.3f}')
print(f'{"─"*75}')
print(f'  Random baseline (all negative):  AUC=0.500  base_rate={n_fast/n:.3f}')
print('\nDone.', flush=True)

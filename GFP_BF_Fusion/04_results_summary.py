#!/usr/bin/env python3
"""
04_results_summary.py

Compile and visualise GFP+BF fusion classification results across datasets.
Produces a summary figure comparing GFP / BF / Fusion AUC for A2, A3, A2+A3.

Outputs:
  figures/results_auc_comparison.png
  figures/results_roc_all.png
  results/results_all_conditions.csv
"""

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

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')
CPE     = LIVEIMG / 'CellposeEmbedding' / 'embeddings'

MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_B2R  = 1094
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

GFP_FILES = {
    'A2': CPE / 'A2_cell_embeddings.npz',
    'A3': CPE / 'A3_cell_embeddings.npz',
}
BF_FILES = {
    'A2': BASE / 'embeddings' / 'A2_bf_at_gfp_onset.npz',
    'A3': BASE / 'embeddings' / 'A3_bf_at_gfp_onset.npz',
}

mdf = pd.read_csv(MODEL_DF)


def load_dataset(ds_list):
    gfp_ids, gfp_embs, bf_ids, bf_embs = [], [], [], []
    for ds in ds_list:
        d_gfp = np.load(str(GFP_FILES[ds]))
        d_bf  = np.load(str(BF_FILES[ds]))
        gfp_ids.append(pd.DataFrame({'track_id': d_gfp['track_ids'].astype(int), 'dataset': ds}))
        gfp_embs.append(d_gfp['embeddings'].astype(np.float32))
        bf_ids.append(pd.DataFrame({'track_id': d_bf['track_ids'].astype(int), 'dataset': ds}))
        bf_embs.append(d_bf['embeddings'].astype(np.float32))

    gfp_id_df = pd.concat(gfp_ids).reset_index(drop=True)
    bf_id_df  = pd.concat(bf_ids).reset_index(drop=True)
    GFP_EMB   = np.vstack(gfp_embs)
    BF_EMB    = np.vstack(bf_embs)

    rows = []
    for ds in ds_list:
        sub = mdf[mdf['dataset'] == ds].copy()
        sub['track_id'] = sub['Track.ID'].str.replace(f'{ds}_', '', regex=False).astype(int)
        sub['b2r']      = sub['delay_green_to_red'] - sub['delay_green_to_blue']
        rows.append(sub)
    meta = pd.concat(rows).reset_index(drop=True)

    gfp_key  = gfp_id_df.set_index(['dataset', 'track_id']).index
    bf_key   = bf_id_df.set_index(['dataset', 'track_id']).index
    meta_key = pd.MultiIndex.from_arrays([meta['dataset'], meta['track_id']])

    eligible = meta[
        meta_key.isin(gfp_key) &
        meta_key.isin(bf_key)  &
        meta['b2r'].notna()    &
        (meta['abs_gfp_onset_min'] <= meta['movie_half_min'])
    ].sort_values(['dataset', 'track_id']).reset_index(drop=True)

    gfp_index = {(r.dataset, r.track_id): i for i, r in gfp_id_df.iterrows()}
    bf_index  = {(r.dataset, r.track_id): i for i, r in bf_id_df.iterrows()}
    gfp_rows  = [gfp_index[(r.dataset, r.track_id)] for _, r in eligible.iterrows()]
    bf_rows   = [bf_index[(r.dataset, r.track_id)]  for _, r in eligible.iterrows()]

    X_GFP = GFP_EMB[gfp_rows]
    X_BF  = BF_EMB[bf_rows]
    y     = (eligible['b2r'].values <= CUT_B2R).astype(int)
    y_reg = eligible['b2r'].values.astype(float)
    return X_GFP, X_BF, y, y_reg, eligible


def select_top_dims(X, y_reg, y_cls, top_n=TOP_N):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                      cv=5, max_iter=10000, n_jobs=-1, random_state=SEED)
    lr = LogisticRegressionCV(**LR_PARAMS)
    en.fit(Xs, y_reg)
    lr.fit(Xs, y_cls)
    rank_r   = pd.Series(np.abs(en.coef_),    index=np.arange(X.shape[1])).rank(ascending=False)
    rank_c   = pd.Series(np.abs(lr.coef_[0]), index=np.arange(X.shape[1])).rank(ascending=False)
    return ((rank_r + rank_c) / 2).sort_values().head(top_n).index.tolist()


def run_cv(X, y):
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
    return oof, dict(auc=round(auc,3), ap=round(ap,3), sens=round(sens,3),
                     spec=round(spec,3), bal_acc=round(bal,3), mcc=round(mcc,3))


# ── Run all conditions ─────────────────────────────────────────────────────────
conditions = [(['A2'], 'A2'), (['A3'], 'A3'), (['A2', 'A3'], 'A2+A3')]
all_records = []
oof_store   = {}   # (ds_label, model_label) → (oof, y)

for ds_list, ds_label in conditions:
    print(f'\n{"═"*55}')
    print(f'Dataset: {ds_label}', flush=True)
    X_GFP, X_BF, y, y_reg, eligible = load_dataset(ds_list)
    n = len(y)
    print(f'  n={n}  fast={y.sum()}  slow={(y==0).sum()}')

    gfp_top = select_top_dims(X_GFP, y_reg, y)
    bf_top  = select_top_dims(X_BF,  y_reg, y)

    X_gfp_t  = X_GFP[:, gfp_top]
    X_bf_t   = X_BF[:, bf_top]
    X_concat = np.concatenate([X_gfp_t, X_bf_t], axis=1)

    for X, model_label in [
        (X_gfp_t,  'GFP alone'),
        (X_bf_t,   'BF alone'),
        (X_concat, 'GFP+BF'),
    ]:
        oof, metrics = run_cv(X, y)
        metrics.update(dict(dataset=ds_label, model=model_label, n=n,
                            n_fast=int(y.sum()), n_slow=int((y==0).sum())))
        all_records.append(metrics)
        oof_store[(ds_label, model_label)] = (oof, y)
        print(f'  {model_label:<12}  AUC={metrics["auc"]:.3f}  '
              f'AP={metrics["ap"]:.3f}  BalAcc={metrics["bal_acc"]:.3f}  '
              f'MCC={metrics["mcc"]:.3f}', flush=True)

results = pd.DataFrame(all_records)
results.to_csv(RESULTS_DIR / 'results_all_conditions.csv', index=False)
print(f'\nSaved results/results_all_conditions.csv')

# ── Figure 1: AUC bar chart across datasets and models ────────────────────────
ds_labels    = ['A2', 'A3', 'A2+A3']
model_labels = ['GFP alone', 'BF alone', 'GFP+BF']
colours      = {'GFP alone': '#4CAF50', 'BF alone': '#2196F3', 'GFP+BF': '#FF9800'}

fig, ax = plt.subplots(figsize=(8, 5))
x      = np.arange(len(ds_labels))
width  = 0.25
offsets = [-width, 0, width]

for offset, model in zip(offsets, model_labels):
    aucs = [results.loc[(results['dataset'] == ds) & (results['model'] == model),
                        'auc'].values[0] for ds in ds_labels]
    bars = ax.bar(x + offset, aucs, width * 0.9, label=model,
                  color=colours[model], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6, label='Random (AUC=0.500)')
ax.set_xticks(x)
ns = {ds: results.loc[results['dataset'] == ds, 'n'].iloc[0] for ds in ds_labels}
ax.set_xticklabels([f'{ds}\n(n={ns[ds]})' for ds in ds_labels], fontsize=11)
ax.set_ylabel('AUC (5-fold OOF)', fontsize=11)
ax.set_title('GFP vs BF vs Fusion — b2r classification\n'
             f'top-{TOP_N} dims per embedding  |  cut={CUT_B2R} min', fontsize=11)
ax.set_ylim(0.4, 0.85)
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'results_auc_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/results_auc_comparison.png')

# ── Figure 2: ROC curves — one panel per dataset ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for ax, ds_label in zip(axes, ds_labels):
    for model in model_labels:
        oof, y = oof_store[(ds_label, model)]
        auc = roc_auc_score(y, oof)
        fpr, tpr, _ = roc_curve(y, oof)
        ax.plot(fpr, tpr, lw=2, color=colours[model],
                label=f'{model}  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    n     = oof_store[(ds_label, 'GFP alone')][1].shape[0]
    n_f   = int(oof_store[(ds_label, 'GFP alone')][1].sum())
    ax.set_title(f'{ds_label}  (n={n}, {n_f} fast)', fontsize=11)
    ax.set_xlabel('1 − Specificity')
    ax.legend(fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Sensitivity')
fig.suptitle(f'ROC curves — b2r classification  |  top-{TOP_N} dims  |  5-fold OOF',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'results_roc_all.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/results_roc_all.png')

# ── Print summary table ───────────────────────────────────────────────────────
print(f'\n{"─"*72}')
print(f'  {"Dataset":<8}  {"Model":<14}  {"n":>5}  {"Fast":>5}  '
      f'{"AUC":>6}  {"AP":>6}  {"Sens":>6}  {"Spec":>6}  {"BalAcc":>7}  {"MCC":>6}')
print(f'{"─"*72}')
for _, row in results.iterrows():
    print(f'  {row["dataset"]:<8}  {row["model"]:<14}  {row["n"]:>5}  '
          f'{row["n_fast"]:>5}  {row["auc"]:>6.3f}  {row["ap"]:>6.3f}  '
          f'{row["sens"]:>6.3f}  {row["spec"]:>6.3f}  '
          f'{row["bal_acc"]:>7.3f}  {row["mcc"]:>6.3f}')
print(f'{"─"*72}')
print('\nDone.', flush=True)

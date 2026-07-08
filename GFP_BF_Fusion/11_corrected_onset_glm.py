#!/usr/bin/env python3
"""
11_corrected_onset_glm.py

GLM classification of fast vs slow b2r using CORRECTED blue-to-red delays
(track-fragmentation correction from A2_b2r_corrected_delays.csv).

Fast/slow cut: 1094 min (GMM Bayes-optimal, same as previous scripts).
The corrected b2r = delay_corrected - delay_green_to_blue accounts for
predecessor tracks that were re-detected after GFP onset.

Feature sets tested:
  A) GFP alone        — top-20 dims of GFP Cellpose embeddings (256-dim)
  B) BF alone         — top-20 dims of BF Cellpose embeddings at GFP onset (256-dim)
  C) GFP+BF concat    — top-20 GFP + top-20 BF dims (40-dim)
  D) Handcrafted      — 29 features from model_df (first 16 frames)
  E) HC + GFP+BF      — handcrafted + top-20 GFP + top-20 BF

Classifier: LogisticRegressionCV with ElasticNet penalty, 5-fold stratified CV.
Output: OOF AUC and metrics saved to results/corrected_onset_glm_*.csv

Run on head node (no GPU needed):
  python GFP_BF_Fusion/11_corrected_onset_glm.py
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef)

warnings.filterwarnings('ignore')

BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')
CPE      = LIVEIMG / 'CellposeEmbedding' / 'embeddings'

MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
CORR_CSV    = LIVEIMG / 'CompleteImage' / 'A2_b2r_corrected_delays.csv'
OUT_LABELS  = LIVEIMG / 'CompleteImage' / 'A2_corrected_b2r_labels.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_B2R  = 1094   # min — GMM Bayes-optimal cutoff (same as prior scripts)
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

HC_COLS = [
    'gfp_corr_start', 'gfp_corr_mean', 'gfp_corr_sd', 'gfp_corr_slope',
    'nuc_bfp_start', 'nuc_bfp_mean', 'nuc_bfp_sd', 'nuc_bfp_slope',
    'nuc_area_mean', 'nuc_area_slope', 'nuc_circ_mean', 'nuc_circ_sd',
    'nuc_ratio_mean', 'nuc_ratio_slope',
    'area_start', 'area_mean', 'area_sd', 'area_slope',
    'solidity_mean', 'solidity_sd', 'shape_idx_mean',
    'bf_ctrst_mean', 'bf_ctrst_sd',
    'gfp_ratio_start', 'gfp_ratio_mean', 'gfp_ratio_sd',
    'gfp_ratio_slope', 'gfp_ratio_max',
    'gfp_snr_sd',
]  # gfp_snr_mean and bf_snr_mean excluded (all NaN)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FAST/SLOW TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print('Building corrected b2r labels...', flush=True)
corr = pd.read_csv(CORR_CSV)
mdf  = pd.read_csv(MODEL_DF)

a2 = mdf[mdf['dataset'] == 'A2'].copy()
a2['track_id'] = a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)

merged = a2.merge(corr[['track_id', 'delay_corrected', 'corrected_onset_frame',
                          'frames_gained', 'min_gained', 'verdict']],
                  on='track_id', how='left')

merged['b2r_orig_min']   = merged['delay_green_to_red'] - merged['delay_green_to_blue']
merged['b2r_corr_min']   = merged['delay_corrected']    - merged['delay_green_to_blue']
merged['corrected_onset_min'] = merged['corrected_onset_frame'] * 15.0  # 15 min/frame

labels = merged[['track_id', 'verdict', 'corrected_onset_frame', 'corrected_onset_min',
                  'min_gained', 'b2r_orig_min', 'b2r_corr_min', 'movie_half_min']].copy()
labels['fast_orig'] = np.where(labels['b2r_orig_min'].notna(), (labels['b2r_orig_min'] <= CUT_B2R).astype(int), np.nan)
labels['fast_corr'] = np.where(labels['b2r_corr_min'].notna(), (labels['b2r_corr_min'] <= CUT_B2R).astype(int), np.nan)
labels.to_csv(OUT_LABELS, index=False)
print(f'  Saved: {OUT_LABELS.name}', flush=True)

b2r = merged[merged['b2r_corr_min'].notna()].copy()
y_all = (b2r['b2r_corr_min'].values <= CUT_B2R).astype(int)

print(f'  b2r cells: {len(b2r)} total  |  {int(y_all.sum())} fast  {int((y_all==0).sum())} slow')


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

print('\nLoading embeddings...', flush=True)
d_gfp = np.load(str(CPE / 'A2_cell_embeddings_all.npz'))
d_bf  = np.load(str(BASE / 'embeddings' / 'A2_bf_at_gfp_onset_all.npz'))

gfp_id2row = {int(tid): i for i, tid in enumerate(d_gfp['track_ids'])}
bf_id2row  = {int(tid): i for i, tid in enumerate(d_bf['track_ids'])}
GFP_EMB = d_gfp['embeddings'].astype(np.float32)
BF_EMB  = d_bf['embeddings'].astype(np.float32)
print(f'  GFP embeddings: {GFP_EMB.shape}   BF embeddings: {BF_EMB.shape}')


def get_embedding_rows(df, id2row_dict):
    rows = []
    missing = []
    for tid in df['track_id']:
        if int(tid) in id2row_dict:
            rows.append(id2row_dict[int(tid)])
        else:
            rows.append(None)
            missing.append(tid)
    if missing:
        print(f'    WARNING: {len(missing)} track_ids not in embedding file', flush=True)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION & CV HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def select_top_dims(X, y_reg, y_cls, top_n=TOP_N, label=''):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                      cv=5, max_iter=10000, n_jobs=-1, random_state=SEED).fit(Xs, y_reg)
    lr = LogisticRegressionCV(**LR_PARAMS).fit(Xs, y_cls)
    rank_r = pd.Series(np.abs(en.coef_),    index=np.arange(X.shape[1])).rank(ascending=False)
    rank_c = pd.Series(np.abs(lr.coef_[0]), index=np.arange(X.shape[1])).rank(ascending=False)
    dims = ((rank_r + rank_c) / 2).sort_values().head(top_n).index.tolist()
    if label:
        print(f'    top-{top_n} {label}: {dims}', flush=True)
    return dims


def compute_metrics(y, oof):
    auc  = roc_auc_score(y, oof)
    ap   = average_precision_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    sens  = int(((pred == 1) & (y == 1)).sum()) / max(n_pos, 1)
    spec  = int(((pred == 0) & (y == 0)).sum()) / max(n_neg, 1)
    bal   = balanced_accuracy_score(y, pred)
    mcc   = matthews_corrcoef(y, pred)
    return dict(auc=round(auc, 3), ap=round(ap, 3),
                sens=round(sens, 3), spec=round(spec, 3),
                bal_acc=round(bal, 3), mcc=round(mcc, 3))


def run_cv_glm(X, y, label=''):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        imp = SimpleImputer(strategy='median')
        Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
        Xte = sc.transform(imp.transform(X[te]))
        m   = LogisticRegressionCV(**LR_PARAMS)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    m = compute_metrics(y, oof)
    if label:
        print(f'  {label:<35}  AUC={m["auc"]:.3f}  AP={m["ap"]:.3f}  '
              f'BalAcc={m["bal_acc"]:.3f}  MCC={m["mcc"]:.3f}', flush=True)
    return oof, m


# ═══════════════════════════════════════════════════════════════════════════════
# RUN CLASSIFICATION — for both all-b2r and half-movie-filter subsets
# ═══════════════════════════════════════════════════════════════════════════════

all_records = []
oof_store   = {}

for subset_label, df_sub, y_sub in [('all_b2r', b2r, y_all)]:
    print(f'\n{"═"*65}', flush=True)
    print(f'SUBSET: {subset_label}  n={len(df_sub)}  fast={int(y_sub.sum())}', flush=True)

    # Gather embedding rows
    gfp_rows = get_embedding_rows(df_sub, gfp_id2row)
    bf_rows  = get_embedding_rows(df_sub, bf_id2row)
    valid_mask = np.array([r is not None for r in gfp_rows]) & \
                 np.array([r is not None for r in bf_rows])
    df_emb = df_sub[valid_mask].reset_index(drop=True)
    y_emb  = y_sub[valid_mask]
    gfp_r  = [r for r, ok in zip(gfp_rows, valid_mask) if ok]
    bf_r   = [r for r, ok in zip(bf_rows,  valid_mask) if ok]

    X_GFP = GFP_EMB[gfp_r]   # (n, 256)
    X_BF  = BF_EMB[bf_r]      # (n, 256)
    y_reg = df_emb['b2r_corr_min'].values.astype(float)

    # Handcrafted features — use merged df (may differ from df_emb if embeddings missing)
    hc_df   = merged[merged['track_id'].isin(df_sub['track_id'])][['track_id'] + HC_COLS].copy()
    hc_df   = hc_df.merge(df_sub[['track_id']].assign(keep=True), on='track_id').drop(columns='keep')
    X_HC_raw = hc_df[HC_COLS].values.astype(np.float32)

    # For the embedding + handcrafted combo, use only cells with embeddings
    hc_df_emb = merged[merged['track_id'].isin(df_emb['track_id'])][['track_id'] + HC_COLS].copy()
    hc_df_emb = hc_df_emb.merge(df_emb[['track_id']], on='track_id')
    X_HC_emb  = hc_df_emb[HC_COLS].values.astype(np.float32)

    y_hc = (df_sub[df_sub['track_id'].isin(hc_df['track_id'])]['b2r_corr_min'].values <= CUT_B2R).astype(int)
    y_hc_emb = y_emb.copy()

    print(f'  Embedding subset: n={len(df_emb)}  fast={int(y_emb.sum())}')
    print(f'  Handcrafted subset: n={len(hc_df)}  fast={int(y_hc.sum())}')

    # Global top-dim selection for embeddings
    print(f'  Selecting top-{TOP_N} dims...', flush=True)
    gfp_top = select_top_dims(X_GFP, y_reg, y_emb, label='GFP')
    bf_top  = select_top_dims(X_BF,  y_reg, y_emb, label='BF')

    X_gfp_t = X_GFP[:, gfp_top]
    X_bf_t  = X_BF[:,  bf_top]
    X_top40 = np.concatenate([X_gfp_t, X_bf_t], axis=1)
    X_hc_gfp_bf = np.concatenate([X_HC_emb, X_gfp_t, X_bf_t], axis=1)

    print(f'\n  [GLM — LogisticRegression + ElasticNet]', flush=True)
    runs = [
        (X_gfp_t,    y_emb,    'GFP (top-20)'),
        (X_bf_t,     y_emb,    'BF (top-20)'),
        (X_top40,    y_emb,    'GFP+BF (top-40)'),
        (X_HC_raw,   y_hc,     'Handcrafted (29 feat)'),
        (X_hc_gfp_bf, y_hc_emb, 'HC + GFP+BF (top-40)'),
    ]
    for X_in, y_in, feat in runs:
        oof, m = run_cv_glm(X_in, y_in, label=feat)
        m.update(subset=subset_label, features=feat,
                 n=len(y_in), n_fast=int(y_in.sum()), cut_min=CUT_B2R)
        all_records.append(m)
        oof_store[(subset_label, feat)] = (oof, y_in.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

col_order = ['subset', 'features', 'n', 'n_fast', 'cut_min',
             'auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc']
results = pd.DataFrame(all_records)[col_order]
results.to_csv(RESULTS_DIR / 'corrected_onset_glm_metrics.csv', index=False)
print(f'\nSaved results/corrected_onset_glm_metrics.csv', flush=True)

oof_rows = []
for (subset, feat), (oof, y_arr) in oof_store.items():
    for p, lbl in zip(oof, y_arr):
        oof_rows.append(dict(subset=subset, features=feat,
                             y_true=int(lbl), oof_prob=round(float(p), 6)))
pd.DataFrame(oof_rows).to_csv(RESULTS_DIR / 'corrected_onset_glm_oof.csv', index=False)
print('Saved results/corrected_onset_glm_oof.csv', flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f'\n{"─"*85}', flush=True)
print(f'  {"Subset":<12}  {"Features":<30}  '
      f'{"n":>4}  {"Fast":>4}  {"AUC":>6}  {"AP":>6}  '
      f'{"Sens":>6}  {"Spec":>6}  {"BalAcc":>7}  {"MCC":>6}')
print(f'{"─"*85}', flush=True)
for _, row in results.iterrows():
    print(f'  {row["subset"]:<12}  {row["features"]:<30}  '
          f'{row["n"]:>4}  {row["n_fast"]:>4}  '
          f'{row["auc"]:>6.3f}  {row["ap"]:>6.3f}  '
          f'{row["sens"]:>6.3f}  {row["spec"]:>6.3f}  '
          f'{row["bal_acc"]:>7.3f}  {row["mcc"]:>6.3f}')
print(f'{"─"*85}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════════

FEAT_KEYS = ['GFP (top-20)', 'BF (top-20)', 'GFP+BF (top-40)',
             'Handcrafted (29 feat)', 'HC + GFP+BF (top-40)']
COLOURS = {
    'GFP (top-20)':          '#4CAF50',
    'BF (top-20)':           '#2196F3',
    'GFP+BF (top-40)':       '#FF9800',
    'Handcrafted (29 feat)': '#9C27B0',
    'HC + GFP+BF (top-40)':  '#E91E63',
}
fig, ax = plt.subplots(figsize=(7, 6))
for feat in FEAT_KEYS:
    if ('all_b2r', feat) not in oof_store:
        continue
    oof, y_arr = oof_store[('all_b2r', feat)]
    auc = roc_auc_score(y_arr, oof)
    fpr, tpr, _ = roc_curve(y_arr, oof)
    ax.plot(fpr, tpr, lw=2, color=COLOURS[feat], label=f'{feat}  AUC={auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
sub_res = results[results['subset'] == 'all_b2r']
n_tot  = sub_res.iloc[0]['n'] if len(sub_res) else 0
n_fast = sub_res.iloc[0]['n_fast'] if len(sub_res) else 0
ax.set_title(f'All b2r cells  (n={n_tot}, {n_fast} fast)\n'
             f'Corrected b2r  |  cut={CUT_B2R} min  |  5-fold OOF', fontsize=10)
ax.set_xlabel('1 − Specificity', fontsize=9)
ax.set_ylabel('Sensitivity', fontsize=9)
ax.legend(fontsize=8, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('ROC — corrected blue-to-red classification  |  A2  |  GLM + ElasticNet',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'corrected_onset_glm_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/corrected_onset_glm_roc.png', flush=True)

# AUC bar chart
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(FEAT_KEYS))
aucs = []
for feat in FEAT_KEYS:
    row = results[(results['subset'] == 'all_b2r') & (results['features'] == feat)]
    aucs.append(float(row['auc'].values[0]) if len(row) else np.nan)
bars = ax.bar(x, aucs, 0.6, color=[COLOURS[f] for f in FEAT_KEYS], alpha=0.85)
for bar, auc in zip(bars, aucs):
    if not np.isnan(auc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6, label='Chance')
ax.set_xticks(x)
ax.set_xticklabels(FEAT_KEYS, rotation=18, ha='right', fontsize=9)
ax.set_ylabel('AUC (5-fold OOF)', fontsize=10)
ax.set_title(f'b2r classification — corrected onset  |  A2  |  n={n_tot}, {n_fast} fast  |  cut={CUT_B2R} min',
             fontsize=10, fontweight='bold')
ax.set_ylim(0.35, 0.95)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'corrected_onset_glm_auc_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/corrected_onset_glm_auc_bars.png', flush=True)

print('\nDone.', flush=True)

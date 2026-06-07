#!/usr/bin/env python3
"""
03_bf_analysis.py

Downstream analysis of BF Cellpose embeddings (10 frames before GFP onset).

PART 1 — Raw 256-dim (5-fold CV)
  Regression:      delay_green_to_red    (ElasticNetCV)
  Classification:  early vs med+late     (LogisticRegressionCV, balanced)

PART 2 — Top-20 dims
  Select top-20 dims by average |beta| rank across both tasks from Part 1.
  Regression:      delay_green_to_red  +  delay_blue_to_red
  Classification:  early vs med+late

GFP embedding baseline (10-frame onset, top-20 dims):
  Regression delay_blue_to_red: r = 0.297
  Classification early vs rest: AUC = 0.742

Outputs:
  results/raw256_regression_metrics.csv / raw256_regression_betas.csv
  results/raw256_classify_metrics.csv   / raw256_classify_betas.csv
  results/top20_dims.csv
  results/top20_regression_metrics.csv  / top20_regression_betas.csv
  results/top20_classify_metrics.csv    / top20_classify_betas.csv
  figures/raw256_regression_betas.png
  figures/raw256_classify_betas.png
  figures/top20_regression_betas.png
  figures/top20_classify_betas.png
  figures/top20_scatter_regression.png
  figures/top20_roc.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error,
                             roc_auc_score, roc_curve, balanced_accuracy_score)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ     = BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_EARLY = 911   # minutes — same as GFP pipeline
SEED      = 42

# ── Load embeddings ────────────────────────────────────────────────────────────
print('Loading BF embeddings...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['gfp_track_ids']          # (511,)  int — GFP track IDs
embeddings    = d['embeddings'].astype(np.float64)  # (511, 256)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
print(f'  {embeddings.shape[0]} cells × {embeddings.shape[1]} dims')
print(f'  Embedding stats: mean={embeddings.mean():.4f}  std={embeddings.std():.4f}')

# ── Load model_df ──────────────────────────────────────────────────────────────
print('Loading model_df...', flush=True)
df = pd.read_csv(MODEL_DF)
df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)

# Regression: delay_green_to_red — all cells with finite outcome
df_reg = df_a2[np.isfinite(df_a2['delay_green_to_red'])].copy()
df_reg = df_reg[df_reg['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_reg = [emb_id_to_row[tid] for tid in df_reg['track_id']]
X_reg = embeddings[rows_reg]
y_reg = df_reg['delay_green_to_red'].values
print(f'  Regression (delay_green_to_red): n={len(y_reg)}')

# Regression: delay_blue_to_red — cells with both outcomes finite
df_b2r = df_a2[np.isfinite(df_a2['delay_green_to_red']) &
               np.isfinite(df_a2['delay_green_to_blue'])].copy()
df_b2r = df_b2r[df_b2r['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_b2r = [emb_id_to_row[tid] for tid in df_b2r['track_id']]
y_b2r = (df_b2r['delay_green_to_red'] - df_b2r['delay_green_to_blue']).values
print(f'  Regression (delay_blue_to_red):  n={len(y_b2r)}')

# Classification: early vs med+late — half-movie filter
df_cls = df[df['abs_gfp_onset_min'] <= df['movie_half_min']].copy()
df_cls = df_cls[df_cls['dataset'] == 'A2'].copy()
df_cls['track_id'] = df_cls['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_cls = df_cls[np.isfinite(df_cls['delay_green_to_red'])].copy()
df_cls = df_cls[df_cls['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_cls = [emb_id_to_row[tid] for tid in df_cls['track_id']]
X_cls = embeddings[rows_cls]
y_cls = (df_cls['delay_green_to_red'].values <= CUT_EARLY).astype(int)
print(f'  Classification: n={len(y_cls)}  ({y_cls.sum()} early / {(y_cls==0).sum()} med+late)')

# ── Helpers ────────────────────────────────────────────────────────────────────
def run_regression_cv(X, y, label, ref_r=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                          cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
        en.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = en.predict(sc.transform(X[te]))
    r2  = r2_score(y, oof)
    r   = pearsonr(y, oof)[0]
    mae = mean_absolute_error(y, oof)
    ref_str = f'  ref={ref_r:.3f}' if ref_r is not None else ''
    print(f'  {label}: R²={r2:.3f}  r={r:.3f}  MAE={mae:.1f} min{ref_str}')
    return oof, r2, r, mae

def fit_full_regression(X, y):
    sc = StandardScaler()
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                      cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
    en.fit(sc.fit_transform(X), y)
    print(f'  Full-data: alpha={en.alpha_:.5f}  l1={en.l1_ratio_:.2f}  '
          f'non-zero: {(en.coef_!=0).sum()}/{X.shape[1]}')
    return en.coef_

LR_PARAMS = dict(
    penalty='elasticnet', solver='saga',
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    Cs=np.logspace(-3, 1, 20),
    class_weight='balanced', scoring='roc_auc',
    max_iter=2000, random_state=SEED, n_jobs=-1,
)

def run_classify_cv(X, y, label, ref_auc=None):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    oof = np.zeros(len(y))
    for tr, te in outer.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**{**LR_PARAMS, 'cv': inner})
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    auc  = roc_auc_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fn = int(((pred==0)&(y==1)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fp = int(((pred==1)&(y==0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal  = balanced_accuracy_score(y, pred)
    ref_str = f'  ref_auc={ref_auc:.3f}' if ref_auc is not None else ''
    print(f'  {label}: AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}{ref_str}')
    return oof, auc, sens, spec, bal

def fit_full_classify(X, y):
    sc = StandardScaler()
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    m = LogisticRegressionCV(**{**LR_PARAMS, 'cv': inner})
    m.fit(sc.fit_transform(X), y)
    print(f'  Full-data: C={m.C_[0]:.5f}  l1={m.l1_ratio_[0]:.2f}  '
          f'non-zero: {(m.coef_[0]!=0).sum()}/{X.shape[1]}')
    return m.coef_[0]

def plot_betas_256(betas, title, path, top_n=30):
    nz   = np.where(betas != 0)[0]
    vals = betas[nz]
    ord_ = np.argsort(np.abs(vals))[::-1]
    nz, vals = nz[ord_], vals[ord_]
    show = min(top_n, len(vals))
    fig, ax = plt.subplots(figsize=(7, max(3, show * 0.28)))
    cols = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals[:show]]
    ax.barh(range(show), vals[:show], color=cols, height=0.7)
    ax.set_yticks(range(show))
    ax.set_yticklabels([f'dim_{i}' for i in nz[:show]], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Beta (scaled)')
    ax.set_title(f'{title}\n{len(nz)} non-zero / 256 dims  (top {show} shown)')
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path.name}')

def plot_betas_top20(betas, dims, title, path):
    order = np.argsort(np.abs(betas))[::-1]
    vals  = betas[order]
    labs  = [f'dim_{dims[i]}' for i in order]
    cols  = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(vals)), vals, color=cols, height=0.7)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Beta (scaled)')
    ax.set_title(f'{title}\n{(betas!=0).sum()} non-zero / {len(betas)} dims')
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path.name}')

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — RAW 256-DIM
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print('PART 1 — RAW 256-DIM')
print('═'*60, flush=True)

print('\n── Regression (delay_green_to_red) ──', flush=True)
_, r2_g, r_g, mae_g = run_regression_cv(X_reg, y_reg, '5-fold CV')
betas_reg = fit_full_regression(X_reg, y_reg)
n_nz_reg = (betas_reg != 0).sum()
beta_df_reg = pd.DataFrame({'dim': np.arange(256), 'beta': betas_reg,
                             'abs_beta': np.abs(betas_reg)}).sort_values('abs_beta', ascending=False)
beta_df_reg.to_csv(RESULTS_DIR / 'raw256_regression_betas.csv', index=False)
pd.DataFrame([dict(task='regression_g2r', n_dims=256, n=len(y_reg),
                   cv_r2=round(r2_g,4), cv_r=round(r_g,4), cv_mae=round(mae_g,1),
                   n_nonzero=int(n_nz_reg))
             ]).to_csv(RESULTS_DIR / 'raw256_regression_metrics.csv', index=False)
plot_betas_256(betas_reg, 'BF embeddings: regression delay_green_to_red (raw 256)',
               FIGURES_DIR / 'raw256_regression_betas.png')
top10_reg = beta_df_reg[beta_df_reg.abs_beta > 0].head(10)['dim'].tolist()
print(f'  Top 10 dims: {top10_reg}')

print('\n── Classification (early vs med+late) ──', flush=True)
_, auc_c, sens_c, spec_c, bal_c = run_classify_cv(X_cls, y_cls, '5-fold CV', ref_auc=0.742)
betas_cls = fit_full_classify(X_cls, y_cls)
n_nz_cls = (betas_cls != 0).sum()
beta_df_cls = pd.DataFrame({'dim': np.arange(256), 'beta': betas_cls,
                             'abs_beta': np.abs(betas_cls)}).sort_values('abs_beta', ascending=False)
beta_df_cls.to_csv(RESULTS_DIR / 'raw256_classify_betas.csv', index=False)
pd.DataFrame([dict(task='classify_early_vs_rest', n_dims=256, n=len(y_cls),
                   n_early=int(y_cls.sum()), auc=round(auc_c,3),
                   sens=round(sens_c,3), spec=round(spec_c,3), bal_acc=round(bal_c,3),
                   n_nonzero=int(n_nz_cls),
                   ref_gfp_auc=0.742)
             ]).to_csv(RESULTS_DIR / 'raw256_classify_metrics.csv', index=False)
plot_betas_256(betas_cls, 'BF embeddings: classification early vs rest (raw 256)',
               FIGURES_DIR / 'raw256_classify_betas.png')
top10_cls = beta_df_cls[beta_df_cls.abs_beta > 0].head(10)['dim'].tolist()
print(f'  Top 10 dims: {top10_cls}')

print(f'\n  Non-zero reg: {n_nz_reg}  |  Non-zero cls: {n_nz_cls}')
overlap_256 = set(np.where(betas_reg != 0)[0]) & set(np.where(betas_cls != 0)[0])
print(f'  Shared non-zero: {len(overlap_256)}  → {sorted(overlap_256)[:20]}{"..." if len(overlap_256)>20 else ""}')

# ══════════════════════════════════════════════════════════════════════════════
# TOP-20 DIM SELECTION
# Rank each non-zero dim by |beta| within each task, average ranks, take top 20
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Top-20 dim selection ──', flush=True)

rank_reg = pd.Series(np.abs(betas_reg), index=np.arange(256)).rank(ascending=False)
rank_cls = pd.Series(np.abs(betas_cls), index=np.arange(256)).rank(ascending=False)
avg_rank = (rank_reg + rank_cls) / 2
top20_dims = avg_rank.sort_values().head(20).index.tolist()
print(f'  Top-20 dims: {top20_dims}')

pd.DataFrame({'rank': range(1, 21),
              'dim': top20_dims,
              'avg_rank': avg_rank[top20_dims].values,
              'abs_beta_reg': np.abs(betas_reg)[top20_dims],
              'abs_beta_cls': np.abs(betas_cls)[top20_dims]}
             ).to_csv(RESULTS_DIR / 'top20_dims.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — TOP-20 DIMS
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print('PART 2 — TOP-20 DIMS')
print('═'*60, flush=True)

X_reg_t20  = embeddings[rows_reg][:, top20_dims]
X_b2r_t20  = embeddings[rows_b2r][:, top20_dims]
X_cls_t20  = embeddings[rows_cls][:, top20_dims]

print('\n── Regression delay_green_to_red (top-20) ──', flush=True)
oof_g, r2_g20, r_g20, mae_g20 = run_regression_cv(X_reg_t20, y_reg, '5-fold CV')
betas_reg20 = fit_full_regression(X_reg_t20, y_reg)

pd.DataFrame([dict(task='regression_g2r_top20', n_dims=20, n=len(y_reg),
                   cv_r2=round(r2_g20,4), cv_r=round(r_g20,4), cv_mae=round(mae_g20,1),
                   n_nonzero=int((betas_reg20!=0).sum()))
             ]).to_csv(RESULTS_DIR / 'top20_regression_g2r_metrics.csv', index=False)

plot_betas_top20(betas_reg20, top20_dims,
                 'BF embeddings: regression delay_green_to_red (top-20)',
                 FIGURES_DIR / 'top20_regression_g2r_betas.png')

print('\n── Regression delay_blue_to_red (top-20, GFP baseline r=0.297) ──', flush=True)
oof_b, r2_b20, r_b20, mae_b20 = run_regression_cv(X_b2r_t20, y_b2r, '5-fold CV', ref_r=0.297)
betas_b2r20 = fit_full_regression(X_b2r_t20, y_b2r)

pd.DataFrame([dict(task='regression_b2r_top20', n_dims=20, n=len(y_b2r),
                   cv_r2=round(r2_b20,4), cv_r=round(r_b20,4), cv_mae=round(mae_b20,1),
                   n_nonzero=int((betas_b2r20!=0).sum()),
                   ref_gfp_r=0.297)
             ]).to_csv(RESULTS_DIR / 'top20_regression_b2r_metrics.csv', index=False)

plot_betas_top20(betas_b2r20, top20_dims,
                 'BF embeddings: regression delay_blue_to_red (top-20)',
                 FIGURES_DIR / 'top20_regression_b2r_betas.png')

# Scatter plot — delay_green_to_red
tertile = np.array(pd.qcut(y_reg, q=3, labels=['early','medium','late']).astype(str))
cols_t  = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}
fig, ax = plt.subplots(figsize=(5, 5))
for t in ['early', 'medium', 'late']:
    m = tertile == t
    ax.scatter(y_reg[m], oof_g[m], c=cols_t[t], label=t, alpha=0.7, s=25, edgecolors='none')
lims = [y_reg.min() - 100, y_reg.max() + 100]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual delay_green_to_red (min)')
ax.set_ylabel('Predicted (min)')
ax.set_title(f'BF embeddings — delay_green_to_red\nCV R²={r2_g20:.3f}  r={r_g20:.3f}  (n={len(y_reg)})')
ax.legend(title='Tertile', fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'top20_scatter_regression.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved top20_scatter_regression.png')

print('\n── Classification early vs med+late (top-20, GFP baseline AUC=0.742) ──', flush=True)
oof_cls20, auc20, sens20, spec20, bal20 = run_classify_cv(X_cls_t20, y_cls, '5-fold CV', ref_auc=0.742)
betas_cls20 = fit_full_classify(X_cls_t20, y_cls)

pd.DataFrame([dict(task='classify_early_vs_rest_top20', n_dims=20, n=len(y_cls),
                   n_early=int(y_cls.sum()), auc=round(auc20,3),
                   sens=round(sens20,3), spec=round(spec20,3), bal_acc=round(bal20,3),
                   n_nonzero=int((betas_cls20!=0).sum()),
                   ref_gfp_auc=0.742)
             ]).to_csv(RESULTS_DIR / 'top20_classify_metrics.csv', index=False)

pd.DataFrame({'orig_dim': top20_dims, 'beta': betas_cls20,
              'abs_beta': np.abs(betas_cls20)}
             ).sort_values('abs_beta', ascending=False
             ).to_csv(RESULTS_DIR / 'top20_classify_betas.csv', index=False)

plot_betas_top20(betas_cls20, top20_dims,
                 'BF embeddings: classification early vs rest (top-20)',
                 FIGURES_DIR / 'top20_classify_betas.png')

# ROC curve
fpr, tpr, _ = roc_curve(y_cls, oof_cls20)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, color='#9C27B0', label=f'BF top-20 dims  AUC={auc20:.3f}')
ax.axhline(0.742, color='#FF5722', lw=1.5, ls='--', label='GFP onset top-20  AUC=0.742')
ax.axhline(0.674, color='#2196F3', lw=1,   ls=':',  label='Tabular (A2)  AUC=0.674')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random  AUC=0.500')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title(f'Early vs Med+Late — BF top-20 embedding dims\n(n={len(y_cls)}, {y_cls.sum()} early)')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'top20_roc.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved top20_roc.png')

# ── Final summary ──────────────────────────────────────────────────────────────
print('\n' + '═'*60)
print('SUMMARY')
print('═'*60)
print(f'  {"Source":<30}  {"Reg r (g2r)":>11}  {"AUC":>8}')
print(f'  {"BF raw 256-dim":<30}  {r_g:>11.3f}  {auc_c:>8.3f}')
print(f'  {"BF top-20 dims":<30}  {r_g20:>11.3f}  {auc20:>8.3f}')
print(f'  {"GFP top-20 (ref, b2r target)":<30}  {"0.297":>11}  {"0.742":>8}')
print(f'  {"Tabular (A2+A3, ref)":<30}  {"0.323":>11}  {"0.684":>8}')
print(f'\n  Top-20 dims: {top20_dims}')
print('\nDone.', flush=True)

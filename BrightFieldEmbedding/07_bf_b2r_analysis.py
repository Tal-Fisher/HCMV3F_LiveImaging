#!/usr/bin/env python3
"""
07_bf_b2r_analysis.py

Complete BF Cellpose embedding analysis with delay_blue_to_red as target.
No half-movie filter. Includes tabular (45 extended features) comparison
on the EXACT same cells.

Pipeline
--------
0. Cell statistics (early/med+late, match quality, temporal consistency)
1. Raw 256-dim BF embeddings  →  5-fold CV regression + classification
2. Top-20 dim selection by average |beta| rank (b2r-targeted)
3. Top-20 BF embeddings       →  5-fold CV regression + classification
4. Tabular 45 extended features (same cells)  →  5-fold CV

Classification cutoff: delay_blue_to_red <= 1094 min  (same as GFP pipeline)

Outputs
-------
  results/b2r_cell_stats.csv
  results/b2r_raw256_{regression,classify}_metrics.csv
  results/b2r_top20_dims.csv
  results/b2r_top20_{regression,classify}_metrics.csv
  results/b2r_tabular_{regression,classify}_metrics.csv
  results/b2r_summary.csv
  figures/b2r_roc_comparison.png
  figures/b2r_scatter_comparison.png
  figures/b2r_top20_emb_regression_betas.png
  figures/b2r_top20_emb_classify_betas.png
  figures/b2r_tabular_regression_betas.png
  figures/b2r_tabular_classify_betas.png
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
EXT_DF_CSV  = LIVEIMG / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'
MATCHES_CSV = BASE / 'bf_gfp_matches.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_B2R = 1094   # minutes — GMM Bayes-optimal cutoff (same as GFP pipeline)
SEED    = 42
TOP_N_DIMS = 20

# 45-feature extended set (same exclusions as BluetoRed analysis)
META_COLS  = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
              'abs_gfp_onset_min', 'movie_half_min'}
EXTRAS_18  = {'cell_aspect_start', 'cell_aspect_mean', 'bfp_nuc_frac_start',
              'nuc_ratio_start', 'nuc_ratio_end',
              'bf_ctrst_start', 'bf_ctrst_end', 'bf_ctrst_slope'}

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading BF embeddings...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['gfp_track_ids']
embeddings    = d['embeddings'].astype(np.float64)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
print(f'  {len(emb_id_to_row)} cells in BF embedding file')

print('Loading extended tabular features...', flush=True)
ext = pd.read_csv(EXT_DF_CSV)
tab_cols = [c for c in ext.columns if c not in META_COLS and c not in EXTRAS_18]
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id'] = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['delay_blue_to_red'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']
print(f'  A2 cells in tabular table: {len(ext)}')
print(f'  Tabular features: {len(tab_cols)}')

print('Loading BF↔GFP match table...', flush=True)
matches = pd.read_csv(MATCHES_CSV)

# ── Build analysis dataset ─────────────────────────────────────────────────────
# Cells with BF embedding AND finite delay_blue_to_red. No half-movie filter.
eligible = ext[
    ext['track_id'].isin(emb_id_to_row) &
    np.isfinite(ext['delay_blue_to_red'])
].sort_values('track_id').reset_index(drop=True)

y_reg = eligible['delay_blue_to_red'].values
y_cls = (y_reg <= CUT_B2R).astype(int)

emb_rows = [emb_id_to_row[tid] for tid in eligible['track_id']]
X_emb    = embeddings[emb_rows]                       # (n, 256)

# Tabular features: median-impute missing values
X_tab_raw = eligible[tab_cols].values.astype(float)
col_med   = np.nanmedian(X_tab_raw, axis=0)
for j in range(X_tab_raw.shape[1]):
    bad = ~np.isfinite(X_tab_raw[:, j])
    X_tab_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
X_tab = X_tab_raw

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 0 — CELL STATISTICS', flush=True)
print('═'*60, flush=True)

used_ids  = set(eligible['track_id'].tolist())
used_mtch = matches[matches['gfp_track_id'].isin(used_ids)].copy()

print(f'\n  Cells with BF embedding + finite b2r: {len(y_reg)}')
print(f'  Early  (b2r <= {CUT_B2R} min): {y_cls.sum()}  ({100*y_cls.mean():.1f}%)')
print(f'  Med+late (b2r > {CUT_B2R} min): {(y_cls==0).sum()}  ({100*(1-y_cls.mean()):.1f}%)')
print(f'  b2r  mean={y_reg.mean():.0f}  median={np.median(y_reg):.0f}  std={y_reg.std():.0f} min')

print(f'\n  Match quality (n={len(used_mtch)}):')
tier_counts = used_mtch['match_tier'].value_counts()
for tier, cnt in tier_counts.items():
    print(f'    {tier}: {cnt}  ({100*cnt/len(used_mtch):.1f}%)')

tc_counts = used_mtch['temporally_consistent'].value_counts()
n_tc  = int(tc_counts.get(True, 0))
n_ntc = int(tc_counts.get(False, 0))
print(f'\n  Temporal consistency (BF↔GFP tracked forward):')
print(f'    Temporally consistent: {n_tc}  ({100*n_tc/len(used_mtch):.1f}%)')
print(f'    Not consistent / data unavailable: {n_ntc}  ({100*n_ntc/len(used_mtch):.1f}%)')

# Save stats CSV
pd.DataFrame([dict(
    n_total=len(y_reg),
    n_early=int(y_cls.sum()), pct_early=round(100*y_cls.mean(), 1),
    n_medlate=int((y_cls==0).sum()),
    b2r_mean=round(y_reg.mean(), 1), b2r_median=round(float(np.median(y_reg)), 1),
    b2r_std=round(y_reg.std(), 1),
    n_confident=int(tier_counts.get('confident', 0)),
    n_plausible=int(tier_counts.get('plausible', 0)),
    n_temporally_consistent=n_tc,
    n_not_consistent=n_ntc,
    pct_temporally_consistent=round(100*n_tc/len(used_mtch), 1),
    cut_b2r=CUT_B2R,
)]).to_csv(RESULTS_DIR / 'b2r_cell_stats.csv', index=False)
print('\n  Saved b2r_cell_stats.csv', flush=True)

# ── Helper functions ───────────────────────────────────────────────────────────
EN_PARAMS = dict(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                 cv=5, max_iter=10000, n_jobs=-1, random_state=SEED)
LR_OUTER  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
LR_INNER  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
LR_PARAMS = dict(penalty='elasticnet', solver='saga',
                 l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                 Cs=np.logspace(-3, 1, 20), cv=LR_INNER,
                 class_weight='balanced', scoring='roc_auc',
                 max_iter=2000, random_state=SEED, n_jobs=-1)


def cv_regression(X, y, label=''):
    kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        en = ElasticNetCV(**EN_PARAMS)
        en.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = en.predict(sc.transform(X[te]))
    r2  = r2_score(y, oof)
    r   = pearsonr(y, oof)[0]
    mae = mean_absolute_error(y, oof)
    if label:
        print(f'    {label}: R²={r2:.3f}  r={r:.3f}  MAE={mae:.1f} min', flush=True)
    return oof, r2, r, mae


def full_regression_coefs(X, y):
    sc = StandardScaler()
    en = ElasticNetCV(**EN_PARAMS)
    en.fit(sc.fit_transform(X), y)
    print(f'    Full-data: alpha={en.alpha_:.5f}  l1={en.l1_ratio_:.2f}  '
          f'nz={( en.coef_!=0).sum()}/{X.shape[1]}', flush=True)
    return en.coef_


def cv_classify(X, y, label=''):
    oof = np.zeros(len(y))
    for tr, te in LR_OUTER.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR_PARAMS)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    auc  = roc_auc_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fn = int(((pred==0)&(y==1)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fp = int(((pred==1)&(y==0)).sum())
    sens = tp/(tp+fn) if tp+fn>0 else 0
    spec = tn/(tn+fp) if tn+fp>0 else 0
    bal  = balanced_accuracy_score(y, pred)
    if label:
        print(f'    {label}: AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}',
              flush=True)
    return oof, auc, sens, spec, bal


def full_classify_coefs(X, y):
    sc = StandardScaler()
    m  = LogisticRegressionCV(**{**LR_PARAMS,
             'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)})
    m.fit(sc.fit_transform(X), y)
    print(f'    Full-data: C={m.C_[0]:.5f}  l1={m.l1_ratio_[0]:.2f}  '
          f'nz={(m.coef_[0]!=0).sum()}/{X.shape[1]}', flush=True)
    return m.coef_[0]


def plot_betas(betas, labels, title, path, top_n=20):
    nz    = np.where(betas != 0)[0]
    vals  = betas[nz]
    ord_  = np.argsort(np.abs(vals))[::-1]
    nz, vals = nz[ord_], vals[ord_]
    show  = min(top_n, len(vals))
    fig, ax = plt.subplots(figsize=(7, max(3, show*0.28)))
    cols  = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals[:show]]
    ax.barh(range(show), vals[:show], color=cols, height=0.7)
    ax.set_yticks(range(show))
    ax.set_yticklabels([labels[i] for i in nz[:show]], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Beta (scaled)')
    ax.set_title(f'{title}\n{len(nz)} non-zero / {len(betas)} features  (top {show} shown)')
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path.name}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW 256-DIM BF EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 1 — RAW 256-DIM BF EMBEDDINGS', flush=True)
print('═'*60, flush=True)

print('\n  Regression (delay_blue_to_red):', flush=True)
_, r2_256_r, r_256_r, mae_256_r = cv_regression(X_emb, y_reg, '5-fold CV')
betas_256_r = full_regression_coefs(X_emb, y_reg)

print('\n  Classification (b2r <= 1094 min):', flush=True)
_, auc_256, sens_256, spec_256, bal_256 = cv_classify(X_emb, y_cls, '5-fold CV')
betas_256_c = full_classify_coefs(X_emb, y_cls)

pd.DataFrame([dict(task='regression_b2r', n_dims=256, n=len(y_reg),
                   cv_r2=round(r2_256_r,4), cv_r=round(r_256_r,4), cv_mae=round(mae_256_r,1),
                   n_nonzero=int((betas_256_r!=0).sum()))
             ]).to_csv(RESULTS_DIR/'b2r_raw256_regression_metrics.csv', index=False)

pd.DataFrame([dict(task='classify_b2r', n_dims=256, n=len(y_cls),
                   n_early=int(y_cls.sum()), auc=round(auc_256,3),
                   sens=round(sens_256,3), spec=round(spec_256,3), bal_acc=round(bal_256,3),
                   n_nonzero=int((betas_256_c!=0).sum()))
             ]).to_csv(RESULTS_DIR/'b2r_raw256_classify_metrics.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOP-20 DIM SELECTION (b2r-targeted)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 2 — TOP-20 DIM SELECTION (b2r-targeted)', flush=True)
print('═'*60, flush=True)

rank_r = pd.Series(np.abs(betas_256_r), index=np.arange(256)).rank(ascending=False)
rank_c = pd.Series(np.abs(betas_256_c), index=np.arange(256)).rank(ascending=False)
avg_rank = (rank_r + rank_c) / 2
top20_dims = avg_rank.sort_values().head(TOP_N_DIMS).index.tolist()
print(f'  Top-20 dims: {top20_dims}', flush=True)

pd.DataFrame({'rank': range(1, TOP_N_DIMS+1),
              'dim': top20_dims,
              'avg_rank': avg_rank[top20_dims].values,
              'abs_beta_reg': np.abs(betas_256_r)[top20_dims],
              'abs_beta_cls': np.abs(betas_256_c)[top20_dims]}
             ).to_csv(RESULTS_DIR/'b2r_top20_dims.csv', index=False)

X_emb_t20 = X_emb[:, top20_dims]
dim_labels = [f'dim_{d}' for d in top20_dims]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TOP-20 BF EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 3 — TOP-20 BF EMBEDDINGS', flush=True)
print('═'*60, flush=True)

print('\n  Regression (delay_blue_to_red):', flush=True)
oof_emb_r, r2_t20_r, r_t20_r, mae_t20_r = cv_regression(X_emb_t20, y_reg, '5-fold CV')
betas_t20_r = full_regression_coefs(X_emb_t20, y_reg)

print('\n  Classification (b2r <= 1094 min):', flush=True)
oof_emb_c, auc_t20, sens_t20, spec_t20, bal_t20 = cv_classify(X_emb_t20, y_cls, '5-fold CV')
betas_t20_c = full_classify_coefs(X_emb_t20, y_cls)

pd.DataFrame([dict(task='regression_b2r_top20', n_dims=20, n=len(y_reg), top20_dims=str(top20_dims),
                   cv_r2=round(r2_t20_r,4), cv_r=round(r_t20_r,4), cv_mae=round(mae_t20_r,1),
                   n_nonzero=int((betas_t20_r!=0).sum()))
             ]).to_csv(RESULTS_DIR/'b2r_top20_regression_metrics.csv', index=False)

pd.DataFrame([dict(task='classify_b2r_top20', n_dims=20, n=len(y_cls), top20_dims=str(top20_dims),
                   n_early=int(y_cls.sum()), auc=round(auc_t20,3),
                   sens=round(sens_t20,3), spec=round(spec_t20,3), bal_acc=round(bal_t20,3),
                   n_nonzero=int((betas_t20_c!=0).sum()))
             ]).to_csv(RESULTS_DIR/'b2r_top20_classify_metrics.csv', index=False)

plot_betas(betas_t20_r, dim_labels,
           'BF embeddings: regression b2r (top-20 dims)',
           FIGURES_DIR/'b2r_top20_emb_regression_betas.png')

plot_betas(betas_t20_c, dim_labels,
           'BF embeddings: classification b2r (top-20 dims)',
           FIGURES_DIR/'b2r_top20_emb_classify_betas.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TABULAR 45 EXTENDED FEATURES (same cells)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 4 — TABULAR 45 EXTENDED FEATURES (exact same cells)', flush=True)
print('═'*60, flush=True)

print('\n  Regression (delay_blue_to_red):', flush=True)
oof_tab_r, r2_tab_r, r_tab_r, mae_tab_r = cv_regression(X_tab, y_reg, '5-fold CV')
betas_tab_r = full_regression_coefs(X_tab, y_reg)

print('\n  Classification (b2r <= 1094 min):', flush=True)
oof_tab_c, auc_tab, sens_tab, spec_tab, bal_tab = cv_classify(X_tab, y_cls, '5-fold CV')
betas_tab_c = full_classify_coefs(X_tab, y_cls)

pd.DataFrame([dict(task='regression_b2r_tabular', n_feat=len(tab_cols), n=len(y_reg),
                   cv_r2=round(r2_tab_r,4), cv_r=round(r_tab_r,4), cv_mae=round(mae_tab_r,1),
                   n_nonzero=int((betas_tab_r!=0).sum()))
             ]).to_csv(RESULTS_DIR/'b2r_tabular_regression_metrics.csv', index=False)

pd.DataFrame([dict(task='classify_b2r_tabular', n_feat=len(tab_cols), n=len(y_cls),
                   n_early=int(y_cls.sum()), auc=round(auc_tab,3),
                   sens=round(sens_tab,3), spec=round(spec_tab,3), bal_acc=round(bal_tab,3),
                   n_nonzero=int((betas_tab_c!=0).sum()))
             ]).to_csv(RESULTS_DIR/'b2r_tabular_classify_metrics.csv', index=False)

plot_betas(betas_tab_r, tab_cols,
           'Tabular 45 features: regression b2r',
           FIGURES_DIR/'b2r_tabular_regression_betas.png')

plot_betas(betas_tab_c, tab_cols,
           'Tabular 45 features: classification b2r',
           FIGURES_DIR/'b2r_tabular_classify_betas.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SUMMARY FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 5 — SUMMARY FIGURES', flush=True)
print('═'*60, flush=True)

# Comparison ROC
fpr_e, tpr_e, _ = roc_curve(y_cls, oof_emb_c)
fpr_t, tpr_t, _ = roc_curve(y_cls, oof_tab_c)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr_e, tpr_e, lw=2, color='#9C27B0',
        label=f'BF emb top-20   AUC={auc_t20:.3f}')
ax.plot(fpr_t, tpr_t, lw=2, color='#FF5722',
        label=f'Tabular 45 feat  AUC={auc_tab:.3f}')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random  AUC=0.500')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title(
    f'B2R: BF Embedding vs Tabular (n={len(y_cls)}, {y_cls.sum()} early)\n'
    f'No half-movie filter  |  cut={CUT_B2R} min',
    fontsize=10
)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR/'b2r_roc_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved b2r_roc_comparison.png', flush=True)

# Comparison regression scatter  (2-panel)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
tertile = np.array(pd.qcut(y_reg, q=3, labels=['early','medium','late']).astype(str))
cols_t  = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}
lims    = [y_reg.min()-100, y_reg.max()+100]

for ax, oof, label, colour in [
        (axes[0], oof_emb_r, f'BF emb top-20\nr={r_t20_r:.3f}  R²={r2_t20_r:.3f}', '#9C27B0'),
        (axes[1], oof_tab_r, f'Tabular 45 feat\nr={r_tab_r:.3f}  R²={r2_tab_r:.3f}', '#FF5722'),
]:
    for t in ['early', 'medium', 'late']:
        m = tertile == t
        ax.scatter(y_reg[m], oof[m], c=cols_t[t], label=t, alpha=0.7, s=20, edgecolors='none')
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual delay_blue_to_red (min)')
    ax.set_ylabel('Predicted (min)')
    ax.set_title(f'{label}\n(n={len(y_reg)})')
    ax.legend(title='Tertile', fontsize=7)

fig.suptitle('BF Cellpose Embedding — delay_blue_to_red regression (5-fold CV OOF)',
             fontsize=11)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR/'b2r_scatter_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved b2r_scatter_comparison.png', flush=True)

# Summary CSV
pd.DataFrame([
    dict(model='BF_emb_raw256', target='b2r', n=len(y_reg),
         cv_r=round(r_256_r,3), cv_r2=round(r2_256_r,3), cv_mae=round(mae_256_r,1),
         auc=round(auc_256,3), sens=round(sens_256,3), spec=round(spec_256,3)),
    dict(model='BF_emb_top20',  target='b2r', n=len(y_reg),
         cv_r=round(r_t20_r,3), cv_r2=round(r2_t20_r,3), cv_mae=round(mae_t20_r,1),
         auc=round(auc_t20,3), sens=round(sens_t20,3), spec=round(spec_t20,3)),
    dict(model='Tabular_45feat',target='b2r', n=len(y_reg),
         cv_r=round(r_tab_r,3), cv_r2=round(r2_tab_r,3), cv_mae=round(mae_tab_r,1),
         auc=round(auc_tab,3), sens=round(sens_tab,3), spec=round(spec_tab,3)),
]).to_csv(RESULTS_DIR/'b2r_summary.csv', index=False)

print('\n' + '═'*60, flush=True)
print('FINAL SUMMARY', flush=True)
print('═'*60, flush=True)
print(f'  n={len(y_reg)}  |  {y_cls.sum()} early / {(y_cls==0).sum()} med+late  '
      f'|  cut={CUT_B2R} min', flush=True)
print(f'  {"Model":<22} {"Regr r":>8} {"Regr R²":>8} {"AUC":>8}')
print(f'  {"BF emb raw 256-dim":<22} {r_256_r:>8.3f} {r2_256_r:>8.3f} {auc_256:>8.3f}')
print(f'  {"BF emb top-20":<22} {r_t20_r:>8.3f} {r2_t20_r:>8.3f} {auc_t20:>8.3f}')
print(f'  {"Tabular 45 feat":<22} {r_tab_r:>8.3f} {r2_tab_r:>8.3f} {auc_tab:>8.3f}')
print('\nDone.', flush=True)

#!/usr/bin/env python3
"""
Regression + classification on delay_blue_to_red using top-20 embedding dims.
Classification labels: early = delay_blue_to_red <= 1094 min (GMM cutoff from BluetoRed tabular analysis).
No half-movie filter (isfinite(delay_blue_to_red) is sufficient).
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

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ     = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'

TOP20_DIMS  = [212, 148, 127, 204, 237, 200, 241, 198, 77, 60, 66, 247, 11, 163, 223, 78, 205, 44, 190, 249]
CUT_B2R     = 1094   # GMM Bayes-optimal cutoff (min) from BluetoRed tabular analysis
SEED        = 42

# ── Load embeddings ────────────────────────────────────────────────────────
print('Loading data...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float64)
X_top = embeddings[:, TOP20_DIMS]
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}

df = pd.read_csv(MODEL_DF)
df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)

# Shared filter: finite delay_blue_to_red, has embedding
df_a2['delay_blue_to_red'] = df_a2['delay_green_to_red'] - df_a2['delay_green_to_blue']
df_b2r = df_a2[
    np.isfinite(df_a2['delay_blue_to_red']) &
    df_a2['track_id'].isin(emb_id_to_row)
].sort_values('track_id').reset_index(drop=True)

rows  = [emb_id_to_row[tid] for tid in df_b2r['track_id']]
X_all = X_top[rows]
y_reg = df_b2r['delay_blue_to_red'].values
y_cls = (y_reg <= CUT_B2R).astype(int)

print(f'  n={len(y_reg)}  b2r mean={y_reg.mean():.0f}  std={y_reg.std():.0f} min')
print(f'  Classification: {y_cls.sum()} early / {(y_cls==0).sum()} med+late  (cut={CUT_B2R} min)')

# ── Beta bar plot ──────────────────────────────────────────────────────────
def plot_betas(betas, dims, title, path):
    order  = np.argsort(np.abs(betas))[::-1]
    vals   = betas[order]
    labels = [f'dim_{dims[i]}' for i in order]
    cols   = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(vals)), vals, color=cols, height=0.7)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Beta (scaled)')
    nz = (betas != 0).sum()
    ax.set_title(f'{title}\n{nz} non-zero / {len(betas)} dims')
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# REGRESSION — delay_blue_to_red
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ REGRESSION (delay_blue_to_red) ═══', flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_reg = np.zeros(len(y_reg))
for tr, te in kf.split(X_all):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_all[tr])
    Xte = sc.transform(X_all[te])
    en  = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                       cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
    en.fit(Xtr, y_reg[tr])
    oof_reg[te] = en.predict(Xte)

cv_r2 = r2_score(y_reg, oof_reg)
cv_r  = pearsonr(y_reg, oof_reg)[0]
cv_mae = mean_absolute_error(y_reg, oof_reg)
print(f'  5-fold CV:  R²={cv_r2:.3f}  r={cv_r:.3f}  MAE={cv_mae:.1f} min')

sc_r = StandardScaler()
en_f = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                    cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
en_f.fit(sc_r.fit_transform(X_all), y_reg)
betas_r = en_f.coef_
print(f'  Full-data: alpha={en_f.alpha_:.4f}  l1={en_f.l1_ratio_:.2f}  '
      f'non-zero: {(betas_r!=0).sum()}/20')

pd.DataFrame([dict(task='regression_b2r', n_dims=20, n=len(y_reg),
                   cv_r2=round(cv_r2, 4), cv_r=round(cv_r, 4), cv_mae=round(cv_mae, 1),
                   n_nonzero=int((betas_r!=0).sum()), cut=CUT_B2R)
             ]).to_csv(str(RESULTS_DIR / 'b2r_regression_metrics.csv'), index=False)

pd.DataFrame({'orig_dim': TOP20_DIMS, 'beta': betas_r, 'abs_beta': np.abs(betas_r)}
             ).sort_values('abs_beta', ascending=False
             ).to_csv(str(RESULTS_DIR / 'b2r_regression_betas.csv'), index=False)

plot_betas(betas_r, TOP20_DIMS,
           'Regression betas: delay_blue_to_red (top-20 dims, b2r labels)',
           FIGURES_DIR / 'b2r_regression_betas.png')

# OOF scatter
tertile = np.array(pd.qcut(y_reg, q=3, labels=['early', 'medium', 'late']).astype(str))
colours = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}
fig, ax = plt.subplots(figsize=(5, 5))
for t in ['early', 'medium', 'late']:
    m = tertile == t
    ax.scatter(y_reg[m], oof_reg[m], c=colours[t], label=t, alpha=0.7, s=25, edgecolors='none')
lims = [y_reg.min() - 100, y_reg.max() + 100]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual delay_blue_to_red (min)')
ax.set_ylabel('Predicted (min)')
ax.set_title(f'delay_blue_to_red — top-20 dims\nCV R²={cv_r2:.3f}  r={cv_r:.3f}  (n={len(y_reg)})')
ax.legend(title='Tertile', fontsize=8)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'b2r_scatter_regression.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved b2r_regression_betas.png + b2r_scatter_regression.png')

# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — early vs med+late (b2r labels, cut=1094 min)
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ CLASSIFICATION (b2r early vs med+late, cut=1094 min) ═══', flush=True)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
LR_CV = dict(penalty='elasticnet', solver='saga',
             l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
             Cs=np.logspace(-3, 1, 20), cv=inner_cv,
             class_weight='balanced', scoring='roc_auc',
             max_iter=2000, random_state=SEED, n_jobs=-1)

oof_cls = np.zeros(len(y_cls))
for tr, te in outer_cv.split(X_all, y_cls):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_all[tr])
    Xte = sc.transform(X_all[te])
    m   = LogisticRegressionCV(**LR_CV)
    m.fit(Xtr, y_cls[tr])
    oof_cls[te] = m.predict_proba(Xte)[:, 1]

auc  = roc_auc_score(y_cls, oof_cls)
pred = (oof_cls >= 0.5).astype(int)
tp = int(((pred == 1) & (y_cls == 1)).sum()); fn = int(((pred == 0) & (y_cls == 1)).sum())
tn = int(((pred == 0) & (y_cls == 0)).sum()); fp = int(((pred == 1) & (y_cls == 0)).sum())
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
bal  = balanced_accuracy_score(y_cls, pred)
print(f'  5-fold CV:  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}')
print(f'  Class balance: {y_cls.sum()} early / {(y_cls==0).sum()} med+late')
print(f'  Tabular (BluetoRed analysis): AUC=0.678 (TabICL A2+A3)')

sc_c = StandardScaler()
lr_f = LogisticRegressionCV(**{**LR_CV,
           'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)})
lr_f.fit(sc_c.fit_transform(X_all), y_cls)
betas_c = lr_f.coef_[0]
print(f'  Full-data: C={lr_f.C_[0]:.4f}  l1={lr_f.l1_ratio_[0]:.2f}  '
      f'non-zero: {(betas_c!=0).sum()}/20')

pd.DataFrame([dict(task='classify_b2r_early_vs_rest', n_dims=20, n=len(y_cls),
                   n_early=int(y_cls.sum()), cut=CUT_B2R,
                   auc=round(auc, 3), sens=round(sens, 3), spec=round(spec, 3),
                   bal_acc=round(bal, 3), n_nonzero=int((betas_c!=0).sum()),
                   ref_tabular_auc=0.678)
             ]).to_csv(str(RESULTS_DIR / 'b2r_classify_metrics.csv'), index=False)

pd.DataFrame({'orig_dim': TOP20_DIMS, 'beta': betas_c, 'abs_beta': np.abs(betas_c)}
             ).sort_values('abs_beta', ascending=False
             ).to_csv(str(RESULTS_DIR / 'b2r_classify_betas.csv'), index=False)

plot_betas(betas_c, TOP20_DIMS,
           'Classification betas: b2r early vs med+late (top-20 dims)',
           FIGURES_DIR / 'b2r_classify_betas.png')

# ROC curve
fpr, tpr, _ = roc_curve(y_cls, oof_cls)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, color='#9C27B0', label=f'Top-20 dims  AUC={auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random AUC=0.500')
ax.axhline(0.678, color='#2196F3', lw=1, ls=':', label='Tabular (A2+A3 TabICL) AUC=0.678')
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'B2R Early vs Med+Late — top-20 embedding dims\n(n={len(y_cls)}, {y_cls.sum()} early, cut={CUT_B2R} min)')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'b2r_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved b2r_classify_betas.png + b2r_roc.png')

print('\nDone.', flush=True)

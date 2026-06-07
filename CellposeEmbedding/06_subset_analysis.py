#!/usr/bin/env python3
"""
Two parts:

PART 1 — Summary figure of all embedding results so far.

PART 2 — Redo regression + classification using only the 116 non-zero dims
          selected by ElasticNet on the full 256-dim raw embeddings.
          Regression target is now delay_blue_to_red
          (= delay_green_to_red - delay_green_to_blue: time from BFP onset to mCherry onset).
          Classification: early vs med+late (threshold on delay_green_to_red ≤ 911 min, unchanged).

Output:
  figures/results_summary_table.png
  results/subset116_regression_metrics.csv
  results/subset116_regression_betas.csv
  results/subset116_classify_metrics.csv
  results/subset116_classify_betas.csv
  figures/subset116_regression_betas.png
  figures/subset116_classify_betas.png
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
                             roc_auc_score, balanced_accuracy_score)
warnings.filterwarnings('ignore')

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ     = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'

CUT_EARLY = 911
SEED      = 42

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — Summary table figure
# ══════════════════════════════════════════════════════════════════════════
print('Generating summary figure...', flush=True)

summary = {
    'Regression (delay_green_to_red)': [
        ('Tabular only — 10-fold CV\n(A2, 29 feat)',       0.359, 0.126, None),
        ('Embedding PCA(50) — 10-fold CV\n(256→50 dims)',  0.128, -0.030, None),
        ('Combined tab+emb — 10-fold CV',                  0.303,  0.083, None),
        ('Raw 256-dim emb — 5-fold CV',                    0.277,  0.073, '116 non-zero'),
    ],
    'Classification (early vs med+late)': [
        ('Tabular only — 5-fold CV\n(A2, 29 feat, script 13)',  None, None, 0.674),
        ('Embedding PCA(50) — 5-fold CV',                       None, None, 0.618),
        ('Combined tab+emb — 5-fold CV',                        None, None, 0.612),
        ('Raw 256-dim emb — 5-fold CV',                         None, None, 0.637),
        ('Reference A2+A3 tabular (script 13)',                  None, None, 0.684),
    ],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colours = ['#2196F3','#FF9800','#4CAF50','#9C27B0','#607D8B']

# Regression panel
ax = axes[0]
rows_r = summary['Regression (delay_green_to_red)']
labels = [r[0] for r in rows_r]
r_vals = [r[1] for r in rows_r]
bars   = ax.barh(labels, r_vals, color=colours[:len(rows_r)], height=0.5)
ax.axvline(0.323, color='red', lw=1.5, ls='--', label='Ref tabular A2+A3 r=0.323')
for bar, v, row in zip(bars, r_vals, rows_r):
    note = f'  r={v:.3f}' + (f'  [{row[3]}]' if row[3] else '')
    ax.text(max(v, 0) + 0.003, bar.get_y() + bar.get_height()/2,
            note, va='center', fontsize=8)
ax.set_xlabel('Pearson r (5/10-fold CV)')
ax.set_title('Regression: delay_green_to_red\n(A2 productive cells, n=291)', fontsize=10)
ax.set_xlim(0, 0.55)
ax.legend(fontsize=8)
ax.invert_yaxis()

# Classification panel
ax = axes[1]
rows_c = summary['Classification (early vs med+late)']
labels = [r[0] for r in rows_c]
auc_vals = [r[3] for r in rows_c]
bars = ax.barh(labels, auc_vals, color=colours[:len(rows_c)], height=0.5)
ax.axvline(0.5, color='gray', lw=1, ls=':', alpha=0.6, label='Random AUC=0.5')
for bar, v in zip(bars, auc_vals):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f'  AUC={v:.3f}', va='center', fontsize=8)
ax.set_xlabel('AUC-ROC (5-fold CV)')
ax.set_title('Classification: early vs med+late\n(A2, n=274, 42 early)', fontsize=10)
ax.set_xlim(0.4, 0.85)
ax.legend(fontsize=8)
ax.invert_yaxis()

fig.suptitle('Cellpose neck embedding — all results summary', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'results_summary_table.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved figures/results_summary_table.png')

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — 116-dim subset analysis
# ══════════════════════════════════════════════════════════════════════════

# ── Load embeddings and select 116 dims ───────────────────────────────────
print('\nLoading data...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float64)   # (291, 256)

betas_reg = pd.read_csv(str(RESULTS_DIR / 'raw_emb_regression_betas.csv'))
selected_dims = betas_reg[betas_reg['abs_beta'] > 0]['dim'].values   # 116 dims
X_sub = embeddings[:, selected_dims]   # (291, 116)
print(f'  Using {len(selected_dims)} selected dims')

# ── Load targets ───────────────────────────────────────────────────────────
df = pd.read_csv(MODEL_DF)
df_full = df.copy()

# Regression: delay_blue_to_red
df_reg = df[df['dataset'] == 'A2'].copy()
df_reg['track_id'] = df_reg['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_reg = df_reg[np.isfinite(df_reg['delay_green_to_red']) &
                np.isfinite(df_reg['delay_green_to_blue'])].copy()
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
df_reg = df_reg[df_reg['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_reg = [emb_id_to_row[tid] for tid in df_reg['track_id']]
X_reg = X_sub[rows_reg]
y_reg = (df_reg['delay_green_to_red'] - df_reg['delay_green_to_blue']).values  # blue→red delay
print(f'  Regression (blue→red): {len(y_reg)} cells  mean={y_reg.mean():.0f} min  std={y_reg.std():.0f} min')

# Classification: early vs med+late (defined on delay_green_to_red)
df_cls = df_full[df_full['abs_gfp_onset_min'] <= df_full['movie_half_min']].reset_index(drop=True)
df_cls = df_cls[df_cls['dataset'] == 'A2'].copy()
df_cls['track_id'] = df_cls['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_cls = df_cls[np.isfinite(df_cls['delay_green_to_red'])].copy()
df_cls = df_cls[df_cls['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_cls = [emb_id_to_row[tid] for tid in df_cls['track_id']]
X_cls = X_sub[rows_cls]
y_cls = (df_cls['delay_green_to_red'].values <= CUT_EARLY).astype(int)
print(f'  Classification: {len(y_cls)} cells  ({y_cls.sum()} early / {(y_cls==0).sum()} med+late)')

# ── Beta plot helper ───────────────────────────────────────────────────────
def plot_betas(betas, dim_labels, title, path, top_n=40):
    order   = np.argsort(np.abs(betas))[::-1]
    show    = min(top_n, (betas != 0).sum())
    idx_s   = order[:show]
    vals_s  = betas[idx_s]
    labs_s  = [f'dim_{dim_labels[i]}' for i in idx_s]
    colours = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals_s]
    fig, ax = plt.subplots(figsize=(7, max(3, show * 0.28)))
    ax.barh(range(show), vals_s, color=colours, height=0.7)
    ax.set_yticks(range(show))
    ax.set_yticklabels(labs_s, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Beta (scaled)')
    total_nz = (betas != 0).sum()
    ax.set_title(f'{title}\n{total_nz} non-zero / {len(betas)} dims  (top {show} shown)')
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# REGRESSION — delay_blue_to_red
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ REGRESSION (delay_blue_to_red, 116 dims) ═══', flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_reg = np.zeros(len(y_reg))
for tr, te in kf.split(X_reg):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_reg[tr])
    Xte = sc.transform(X_reg[te])
    en  = ElasticNetCV(
        l1_ratio=[0.5, 0.9, 1.0],
        alphas=np.logspace(-2, 3, 20),
        cv=5, max_iter=5000, n_jobs=-1, random_state=SEED,
    )
    en.fit(Xtr, y_reg[tr])
    oof_reg[te] = en.predict(Xte)

cv_r2  = r2_score(y_reg, oof_reg)
cv_r   = pearsonr(y_reg, oof_reg)[0]
cv_mae = mean_absolute_error(y_reg, oof_reg)
print(f'  5-fold CV:  R²={cv_r2:.3f}  r={cv_r:.3f}  MAE={cv_mae:.1f} min')

# Full-data model for betas
sc_full = StandardScaler()
en_full = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                       cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
en_full.fit(sc_full.fit_transform(X_reg), y_reg)
betas_r = en_full.coef_
n_nz_r  = (betas_r != 0).sum()
print(f'  Full-data: alpha={en_full.alpha_:.4f}  l1={en_full.l1_ratio_:.2f}  '
      f'non-zero: {n_nz_r}/{len(selected_dims)}')

# Top dims in original 256-dim space
top_order = np.argsort(np.abs(betas_r))[::-1]
top_orig  = selected_dims[top_order[:10]]
print(f'  Top 10 original dims: {top_orig.tolist()}')

pd.DataFrame([dict(task='regression_blue_to_red', n=len(y_reg),
                   cv_r2=round(cv_r2,4), cv_r=round(cv_r,4), cv_mae=round(cv_mae,1),
                   n_nonzero=int(n_nz_r), n_input_dims=len(selected_dims),
                   best_alpha=round(en_full.alpha_,5), best_l1=en_full.l1_ratio_)
             ]).to_csv(str(RESULTS_DIR / 'subset116_regression_metrics.csv'), index=False)

beta_df_r = pd.DataFrame({
    'subset_idx':  np.arange(len(selected_dims)),
    'orig_dim':    selected_dims,
    'beta':        betas_r,
    'abs_beta':    np.abs(betas_r),
}).sort_values('abs_beta', ascending=False)
beta_df_r.to_csv(str(RESULTS_DIR / 'subset116_regression_betas.csv'), index=False)

plot_betas(betas_r, selected_dims,
           'Regression betas: delay_blue_to_red (116-dim subset)',
           FIGURES_DIR / 'subset116_regression_betas.png')
print(f'  Saved figures/subset116_regression_betas.png')

# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — early vs med+late
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ CLASSIFICATION (early vs med+late, 116 dims) ═══', flush=True)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
LR_CV = dict(
    penalty='elasticnet', solver='saga',
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    Cs=np.logspace(-3, 1, 20),
    cv=inner_cv, class_weight='balanced',
    scoring='roc_auc', max_iter=2000,
    random_state=SEED, n_jobs=-1,
)

oof_cls = np.zeros(len(y_cls))
for tr, te in outer_cv.split(X_cls, y_cls):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_cls[tr])
    Xte = sc.transform(X_cls[te])
    m   = LogisticRegressionCV(**LR_CV)
    m.fit(Xtr, y_cls[tr])
    oof_cls[te] = m.predict_proba(Xte)[:, 1]

auc    = roc_auc_score(y_cls, oof_cls)
pred   = (oof_cls >= 0.5).astype(int)
tp = int(((pred==1)&(y_cls==1)).sum()); fn = int(((pred==0)&(y_cls==1)).sum())
tn = int(((pred==0)&(y_cls==0)).sum()); fp = int(((pred==1)&(y_cls==0)).sum())
sens   = tp/(tp+fn) if (tp+fn)>0 else 0
spec   = tn/(tn+fp) if (tn+fp)>0 else 0
bal    = balanced_accuracy_score(y_cls, pred)
print(f'  5-fold CV:  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}')
print(f'  Reference (raw 256-dim): AUC=0.637')

# Full-data model for betas
sc_cls_full = StandardScaler()
lr_full = LogisticRegressionCV(**{**LR_CV,
    'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)})
lr_full.fit(sc_cls_full.fit_transform(X_cls), y_cls)
betas_c = lr_full.coef_[0]
n_nz_c  = (betas_c != 0).sum()
print(f'  Full-data: C={lr_full.C_[0]:.4f}  l1={lr_full.l1_ratio_[0]:.2f}  '
      f'non-zero: {n_nz_c}/{len(selected_dims)}')

top_orig_c = selected_dims[np.argsort(np.abs(betas_c))[::-1][:10]]
print(f'  Top 10 original dims: {top_orig_c.tolist()}')

pd.DataFrame([dict(task='classify_early_vs_rest', n=len(y_cls), n_early=int(y_cls.sum()),
                   auc=round(auc,3), sens=round(sens,3), spec=round(spec,3),
                   bal_acc=round(bal,3), n_nonzero=int(n_nz_c),
                   n_input_dims=len(selected_dims),
                   best_C=round(float(lr_full.C_[0]),5),
                   best_l1=float(lr_full.l1_ratio_[0]),
                   ref_auc_raw256=0.637)
             ]).to_csv(str(RESULTS_DIR / 'subset116_classify_metrics.csv'), index=False)

beta_df_c = pd.DataFrame({
    'subset_idx': np.arange(len(selected_dims)),
    'orig_dim':   selected_dims,
    'beta':       betas_c,
    'abs_beta':   np.abs(betas_c),
}).sort_values('abs_beta', ascending=False)
beta_df_c.to_csv(str(RESULTS_DIR / 'subset116_classify_betas.csv'), index=False)

plot_betas(betas_c, selected_dims,
           'Classification betas: early vs med+late (116-dim subset)',
           FIGURES_DIR / 'subset116_classify_betas.png')
print(f'  Saved figures/subset116_classify_betas.png')

# ── Overlap ───────────────────────────────────────────────────────────────
nz_r = set(selected_dims[betas_r != 0])
nz_c = set(selected_dims[betas_c != 0])
print(f'\n  Non-zero in regression: {len(nz_r)}  |  classification: {len(nz_c)}')
print(f'  Shared: {len(nz_r & nz_c)}  → {sorted(nz_r & nz_c)[:20]}{"..." if len(nz_r & nz_c)>20 else ""}')

print('\nDone.', flush=True)

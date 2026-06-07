#!/usr/bin/env python3
"""
Use raw 256-dim Cellpose neck embeddings (no PCA) for both tasks.
ElasticNet/LogisticRegressionCV selects directly from 256 dims.
Saves ranked beta tables so we can see which dimensions matter.

Tasks:
  A) Regression:      predict delay_green_to_red  (ElasticNetCV, 10-fold)
  B) Classification:  early vs med+late           (LogisticRegressionCV, 5-fold)

Output:
  results/raw_emb_regression_metrics.csv
  results/raw_emb_regression_betas.csv      -- all 256 dims, sorted by |beta|
  results/raw_emb_classify_metrics.csv
  results/raw_emb_classify_betas.csv        -- all 256 dims, sorted by |beta|
  figures/raw_emb_regression_betas.png
  figures/raw_emb_classify_betas.png
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

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ  = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_EARLY  = 911
SEED       = 42

# ── Load embeddings ────────────────────────────────────────────────────────
print('Loading embeddings...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float64)   # (291, 256)
print(f'  {embeddings.shape[0]} cells × {embeddings.shape[1]} dims')

# ── Load model_df ──────────────────────────────────────────────────────────
print('Loading model_df...', flush=True)
df = pd.read_csv(MODEL_DF)
df_full = df.copy()   # keep full for classification filter

# --- Regression dataset (291 cells, no onset filter) ----------------------
df_reg = df[df['dataset'] == 'A2'].copy()
df_reg['track_id'] = df_reg['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_reg = df_reg[np.isfinite(df_reg['delay_green_to_red'])].copy()

emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
df_reg = df_reg[df_reg['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_reg = [emb_id_to_row[tid] for tid in df_reg['track_id']]
X_reg = embeddings[rows_reg]
y_reg = df_reg['delay_green_to_red'].values
print(f'  Regression: {len(y_reg)} cells')

# --- Classification dataset (274 cells, onset filter from script 13) ------
df_cls = df_full.copy()
df_cls = df_cls[df_cls['abs_gfp_onset_min'] <= df_cls['movie_half_min']].reset_index(drop=True)
df_cls = df_cls[df_cls['dataset'] == 'A2'].copy()
df_cls['track_id'] = df_cls['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_cls = df_cls[np.isfinite(df_cls['delay_green_to_red'])].copy()
df_cls = df_cls[df_cls['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_cls = [emb_id_to_row[tid] for tid in df_cls['track_id']]
X_cls = embeddings[rows_cls]
y_cls = (df_cls['delay_green_to_red'].values <= CUT_EARLY).astype(int)
print(f'  Classification: {len(y_cls)} cells  ({y_cls.sum()} early / {(y_cls==0).sum()} med+late)')

# ── Helper: coefficient bar chart (top N non-zero dims) ───────────────────
def plot_betas(betas, title, path, top_n=30):
    nz_mask = betas != 0
    nz_idx  = np.where(nz_mask)[0]
    nz_vals = betas[nz_mask]
    order   = np.argsort(np.abs(nz_vals))[::-1]
    nz_idx  = nz_idx[order]
    nz_vals = nz_vals[order]
    show    = min(top_n, len(nz_vals))

    fig, ax = plt.subplots(figsize=(7, max(3, show * 0.28)))
    colours = ['#e74c3c' if v > 0 else '#2ecc71' for v in nz_vals[:show]]
    ax.barh(range(show), nz_vals[:show], color=colours, height=0.7)
    ax.set_yticks(range(show))
    ax.set_yticklabels([f'dim_{i}' for i in nz_idx[:show]], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Beta (scaled)')
    ax.set_title(f'{title}\n{nz_mask.sum()} non-zero / 256 dims  (top {show} shown)')
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# TASK A — REGRESSION
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ REGRESSION (delay_green_to_red) ═══', flush=True)

# 10-fold CV for honest OOF metrics
kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
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
print(f'  10-fold CV:  R²={cv_r2:.3f}  r={cv_r:.3f}  MAE={cv_mae:.1f} min')
print(f'  Reference tabular (A2+A3): R²=0.104  r=0.323')

# Full-data model for beta extraction
sc_full = StandardScaler()
X_sc    = sc_full.fit_transform(X_reg)
en_full = ElasticNetCV(
    l1_ratio=[0.5, 0.9, 1.0],
    alphas=np.logspace(-2, 3, 20),
    cv=5, max_iter=5000, n_jobs=-1, random_state=SEED,
)
en_full.fit(X_sc, y_reg)
betas_reg = en_full.coef_
n_nz_reg  = (betas_reg != 0).sum()
print(f'  Full-data model: alpha={en_full.alpha_:.4f}  l1={en_full.l1_ratio_:.2f}  '
      f'non-zero dims: {n_nz_reg}/256')

# Save metrics
pd.DataFrame([dict(task='regression', n=len(y_reg), cv_r2=round(cv_r2,4),
                   cv_r=round(cv_r,4), cv_mae=round(cv_mae,1),
                   n_nonzero=int(n_nz_reg),
                   best_alpha=round(en_full.alpha_,5), best_l1=en_full.l1_ratio_,
                   ref_r2=0.104, ref_r=0.323)
             ]).to_csv(str(RESULTS_DIR / 'raw_emb_regression_metrics.csv'), index=False)

# Save beta table
beta_df_reg = pd.DataFrame({
    'dim':      np.arange(256),
    'beta':     betas_reg,
    'abs_beta': np.abs(betas_reg),
}).sort_values('abs_beta', ascending=False)
beta_df_reg.to_csv(str(RESULTS_DIR / 'raw_emb_regression_betas.csv'), index=False)
print(f'  Top 10 dims: {beta_df_reg[beta_df_reg.abs_beta>0].head(10).dim.tolist()}')

plot_betas(betas_reg, 'Regression betas: delay_green_to_red',
           FIGURES_DIR / 'raw_emb_regression_betas.png')
print(f'  Saved figures/raw_emb_regression_betas.png')

# ══════════════════════════════════════════════════════════════════════════
# TASK B — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ CLASSIFICATION (early vs med+late) ═══', flush=True)

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

auc     = roc_auc_score(y_cls, oof_cls)
pred    = (oof_cls >= 0.5).astype(int)
tp = int(((pred==1)&(y_cls==1)).sum()); fn = int(((pred==0)&(y_cls==1)).sum())
tn = int(((pred==0)&(y_cls==0)).sum()); fp = int(((pred==1)&(y_cls==0)).sum())
sens    = tp / (tp + fn) if (tp + fn) > 0 else 0
spec    = tn / (tn + fp) if (tn + fp) > 0 else 0
bal_acc = balanced_accuracy_score(y_cls, pred)
print(f'  5-fold CV:  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal_acc:.3f}')
print(f'  Reference tabular (A2+A3): AUC=0.684')

# Full-data model for beta extraction
sc_cls  = StandardScaler()
X_cls_sc = sc_cls.fit_transform(X_cls)
lr_full  = LogisticRegressionCV(**{**LR_CV, 'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)})
lr_full.fit(X_cls_sc, y_cls)
betas_cls = lr_full.coef_[0]
n_nz_cls  = (betas_cls != 0).sum()
print(f'  Full-data model: C={lr_full.C_[0]:.4f}  l1={lr_full.l1_ratio_[0]:.2f}  '
      f'non-zero dims: {n_nz_cls}/256')

# Save metrics
pd.DataFrame([dict(task='classification', n=len(y_cls), n_early=int(y_cls.sum()),
                   auc=round(auc,3), sens=round(sens,3), spec=round(spec,3),
                   bal_acc=round(bal_acc,3), n_nonzero=int(n_nz_cls),
                   best_C=round(float(lr_full.C_[0]),5),
                   best_l1=float(lr_full.l1_ratio_[0]),
                   ref_auc=0.684)
             ]).to_csv(str(RESULTS_DIR / 'raw_emb_classify_metrics.csv'), index=False)

# Save beta table
beta_df_cls = pd.DataFrame({
    'dim':      np.arange(256),
    'beta':     betas_cls,
    'abs_beta': np.abs(betas_cls),
}).sort_values('abs_beta', ascending=False)
beta_df_cls.to_csv(str(RESULTS_DIR / 'raw_emb_classify_betas.csv'), index=False)
print(f'  Top 10 dims: {beta_df_cls[beta_df_cls.abs_beta>0].head(10).dim.tolist()}')

plot_betas(betas_cls, 'Classification betas: early vs med+late',
           FIGURES_DIR / 'raw_emb_classify_betas.png')
print(f'  Saved figures/raw_emb_classify_betas.png')

# ── Overlap between tasks ──────────────────────────────────────────────────
nz_reg = set(np.where(betas_reg != 0)[0])
nz_cls = set(np.where(betas_cls != 0)[0])
overlap = nz_reg & nz_cls
print(f'\n  Non-zero dims in regression: {len(nz_reg)}')
print(f'  Non-zero dims in classification: {len(nz_cls)}')
print(f'  Shared non-zero dims: {len(overlap)}  → {sorted(overlap)}')

print('\nDone.', flush=True)

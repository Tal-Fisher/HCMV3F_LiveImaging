#!/usr/bin/env python3
"""
Three-way 10-fold CV comparison:
  A) Tabular features only   (from model_df.csv, 29 features)
  B) Embedding only          (PCA of Cellpose neck embeddings)
  C) Combined                (tabular + embedding PCA)

Uses identical 10-fold splits across all three models.
Reference: original R tabular ElasticNet (A2+A3, 45 feat): CV R²=0.104, r=0.323

Output:
  results/combined_cv_metrics.csv   -- CV R², r, MAE for all three models
  figures/combined_cv_scatter.png   -- 3-panel OOF scatter
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore', category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ  = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)

PCA_COMPONENTS = 50
N_FOLDS        = 10
RANDOM_STATE   = 42

EN_PARAMS = dict(
    l1_ratio=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    alphas=np.logspace(-4, 2, 60),
    cv=5,             # inner CV for lambda selection
    max_iter=50000,
    random_state=RANDOM_STATE,
)

# ── Load data ──────────────────────────────────────────────────────────────
print('Loading embeddings...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float32)

print('Loading tabular features...', flush=True)
df = pd.read_csv(MODEL_DF)
df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
a2 = df_a2[np.isfinite(df_a2['delay_green_to_red'])].copy()

DROP = {'Track.ID', 'track_id', 'delay_green_to_red', 'delay_green_to_blue', 'dataset', 'y',
        'green_onset_min', 'track_start_min', 'abs_gfp_onset_min', 'movie_half_min',
        'gfp_snr_mean', 'bf_snr_mean'}
tab_cols = [c for c in a2.columns if c not in DROP]
print(f'  Tabular features: {len(tab_cols)}')

# Align on track_id
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
a2 = a2[a2['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
emb_rows = [emb_id_to_row[tid] for tid in a2['track_id']]

X_emb = embeddings[emb_rows].astype(np.float64)
X_tab = a2[tab_cols].values.astype(np.float64)
y     = a2['delay_green_to_red'].values
print(f'  N cells: {len(y)}', flush=True)

# ── 10-fold CV ─────────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(kf.split(X_tab))

def cv_model(X_raw, label, pca_components=None, impute=False):
    """Run 10-fold CV, return OOF predictions array."""
    print(f'\n── {label} ──', flush=True)
    oof = np.zeros(len(y))
    for i, (tr, te) in enumerate(folds):
        Xtr, Xte = X_raw[tr], X_raw[te]
        if impute:
            imp = SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(Xtr)
            Xte = imp.transform(Xte)
        if pca_components is not None:
            pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        en  = ElasticNetCV(**EN_PARAMS)
        en.fit(Xtr, y[tr])
        oof[te] = en.predict(Xte)
        if i == 0:
            print(f'  fold 1: alpha={en.alpha_:.4f} l1={en.l1_ratio_:.2f} '
                  f'nz={int((en.coef_!=0).sum())}/{len(en.coef_)}', flush=True)
    r2  = r2_score(y, oof)
    r   = pearsonr(y, oof)[0]
    mae = mean_absolute_error(y, oof)
    print(f'  CV R²={r2:.3f}  r={r:.3f}  MAE={mae:.1f} min')
    return oof, dict(model=label, n=len(y), n_features=X_raw.shape[1] if pca_components is None else pca_components,
                     cv_r2=round(r2,4), cv_r=round(r,4), cv_mae=round(mae,1))

oof_A, met_A = cv_model(X_tab,  'Tabular only', pca_components=None, impute=True)
oof_B, met_B = cv_model(X_emb,  'Embedding only', pca_components=PCA_COMPONENTS, impute=False)

# Combined: PCA-reduce embeddings + impute tabular, scale each block independently per fold
print('\n── Combined ──', flush=True)
oof_C = np.zeros(len(y))
for i, (tr, te) in enumerate(folds):
    imp = SimpleImputer(strategy='median')
    Xtab_tr = imp.fit_transform(X_tab[tr])
    Xtab_te = imp.transform(X_tab[te])

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    Xemb_tr = pca.fit_transform(X_emb[tr])
    Xemb_te = pca.transform(X_emb[te])

    sc_tab = StandardScaler()
    Xtab_tr = sc_tab.fit_transform(Xtab_tr)
    Xtab_te = sc_tab.transform(Xtab_te)

    sc_emb = StandardScaler()
    Xemb_tr = sc_emb.fit_transform(Xemb_tr)
    Xemb_te = sc_emb.transform(Xemb_te)

    Xtr = np.hstack([Xtab_tr, Xemb_tr])
    Xte = np.hstack([Xtab_te, Xemb_te])

    en = ElasticNetCV(**EN_PARAMS)
    en.fit(Xtr, y[tr])
    oof_C[te] = en.predict(Xte)
    if i == 0:
        print(f'  fold 1: alpha={en.alpha_:.4f} l1={en.l1_ratio_:.2f} '
              f'nz={int((en.coef_!=0).sum())}/{len(en.coef_)} '
              f'({len(tab_cols)} tab + {PCA_COMPONENTS} emb)', flush=True)

r2_C  = r2_score(y, oof_C)
r_C   = pearsonr(y, oof_C)[0]
mae_C = mean_absolute_error(y, oof_C)
print(f'  CV R²={r2_C:.3f}  r={r_C:.3f}  MAE={mae_C:.1f} min')
met_C = dict(model='Combined', n=len(y), n_features=len(tab_cols)+PCA_COMPONENTS,
             cv_r2=round(r2_C,4), cv_r=round(r_C,4), cv_mae=round(mae_C,1))

# ── Save metrics ───────────────────────────────────────────────────────────
print('\n── Summary ──')
metrics_df = pd.DataFrame([met_A, met_B, met_C])
metrics_df['ref_r2'] = 0.104
metrics_df['ref_r']  = 0.323
print(metrics_df[['model','n','n_features','cv_r2','cv_r','cv_mae']].to_string(index=False))

csv_path = RESULTS_DIR / 'combined_cv_metrics.csv'
metrics_df.to_csv(str(csv_path), index=False)
print(f'Saved: {csv_path}')

# ── Figure: 3-panel OOF scatter ────────────────────────────────────────────
tertile = np.array(pd.qcut(y, q=3, labels=['early', 'medium', 'late']).astype(str))
colours = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (label, oof, m) in zip(axes, [('Tabular only', oof_A, met_A),
                                       ('Embedding only', oof_B, met_B),
                                       ('Combined', oof_C, met_C)]):
    for tert in ['early', 'medium', 'late']:
        mask = tertile == tert
        ax.scatter(y[mask], oof[mask],
                   c=colours[tert], label=tert, alpha=0.6, s=20, edgecolors='none')
    lims = [min(y.min(), oof.min()) - 100, max(y.max(), oof.max()) + 100]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual delay (min)')
    ax.set_ylabel('Predicted delay (min)')
    ax.set_title(f'{label}\nCV R²={m["cv_r2"]:.3f}  r={m["cv_r"]:.3f}')
    if ax == axes[0]:
        ax.legend(title='Tertile', fontsize=8, loc='upper left')

fig.suptitle(f'A2 cells — 10-fold CV: tabular vs embedding vs combined  (n={len(y)})\n'
             f'Reference tabular (A2+A3, R): R²=0.104  r=0.323',
             fontsize=10)
plt.tight_layout()
png_path = FIGURES_DIR / 'combined_cv_scatter.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {png_path}')

print('\nDone.', flush=True)

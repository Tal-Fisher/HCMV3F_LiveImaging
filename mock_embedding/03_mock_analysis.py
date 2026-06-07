#!/usr/bin/env python3
"""
03_mock_analysis.py

Null control analysis: run the same CV pipelines on mock (random-crop) embeddings
with the same labels and compare to real results.

Expected:
  - Mock r ≈ 0, AUC ≈ 0.5  → signal is cell-specific ✓
  - Mock r or AUC clearly above chance → frame-level confound or pipeline artefact ✗
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
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
warnings.filterwarnings('ignore')

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging')
BF_DIR  = BASE / 'BrightFieldEmbedding'
OUT_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/mock_embedding')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

EXT_DF_CSV = BASE / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'
REAL_GFP_NPZ = BASE / 'CellposeEmbedding' / 'embeddings' / 'A2_cell_embeddings.npz'
REAL_BF_NPZ  = BF_DIR / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
BF_TOP20_CSV = BF_DIR / 'results' / 'b2r_top20_dims.csv'

CUT_B2R = 1094
SEED    = 42

GFP_TOP20_DIMS = [212, 148, 127, 204, 237, 200, 241, 198, 77, 60, 66, 247,
                  11, 163, 223, 78, 205, 44, 190, 249]

META_COLS = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
             'abs_gfp_onset_min', 'movie_half_min'}
EXTRAS_18 = {'cell_aspect_start', 'cell_aspect_mean', 'bfp_nuc_frac_start',
             'nuc_ratio_start', 'nuc_ratio_end',
             'bf_ctrst_start', 'bf_ctrst_end', 'bf_ctrst_slope'}

EN_PARAMS = dict(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                 cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
LR_OUTER  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
LR_INNER  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
LR_PARAMS = dict(penalty='elasticnet', solver='saga',
                 l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                 Cs=np.logspace(-3, 1, 20), cv=LR_INNER,
                 class_weight='balanced', scoring='roc_auc',
                 max_iter=2000, random_state=SEED, n_jobs=-1)

# ── Helpers ────────────────────────────────────────────────────────────────────
def cv_regression(X, y, label=''):
    kf  = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        en = ElasticNetCV(**EN_PARAMS)
        en.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = en.predict(sc.transform(X[te]))
    r   = pearsonr(y, oof)[0]
    r2  = r2_score(y, oof)
    mae = mean_absolute_error(y, oof)
    print(f'  {label}: r={r:.3f}  R²={r2:.3f}  MAE={mae:.1f} min', flush=True)
    return r, r2, mae

def cv_classify(X, y, label=''):
    oof = np.zeros(len(y))
    for tr, te in LR_OUTER.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR_PARAMS)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    auc = roc_auc_score(y, oof)
    print(f'  {label}: AUC={auc:.3f}', flush=True)
    return auc

# ── Load labels ────────────────────────────────────────────────────────────────
ext = pd.read_csv(EXT_DF_CSV)
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id'] = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['delay_blue_to_red'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']
label_idx = ext.set_index('track_id')

def get_labels(track_ids):
    tids = [t for t in track_ids if t in label_idx.index]
    y_reg = np.array([label_idx.loc[t, 'delay_blue_to_red'] for t in tids])
    mask = np.isfinite(y_reg)
    tids  = [t for t, m in zip(tids, mask) if m]
    y_reg = y_reg[mask]
    y_cls = (y_reg <= CUT_B2R).astype(int)
    return np.array(tids), y_reg, y_cls

# ── BF top-20 dims ─────────────────────────────────────────────────────────────
bf_top20_dims = pd.read_csv(BF_TOP20_CSV)['dim'].values.tolist()
print(f'BF top-20 dims: {bf_top20_dims}', flush=True)

# Known real results (from completed analyses)
REAL = {
    'GFP top-20  reg':  0.297,
    'GFP top-20  cls':  0.708,
    'BF raw-256  reg': -0.010,
    'BF raw-256  cls':  0.715,
    'BF top-20   reg':  0.243,
    'BF top-20   cls':  0.876,
}

results = []

# ══════════════════════════════════════════════════════════════════════════════
# GFP MOCK
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60, flush=True)
print('GFP MOCK ANALYSIS', flush=True)
print('='*60, flush=True)

mock_gfp = np.load(str(OUT_DIR / 'gfp_mock_embeddings.npz'))
mock_gfp_ids  = mock_gfp['gfp_track_ids'].astype(int)
mock_gfp_embs = mock_gfp['embeddings']

tids, y_reg, y_cls = get_labels(mock_gfp_ids)
id_to_row = {t: i for i, t in enumerate(mock_gfp_ids.tolist())}
rows = [id_to_row[t] for t in tids]
X_gfp = mock_gfp_embs[rows][:, GFP_TOP20_DIMS]

print(f'n={len(tids)}  early={y_cls.sum()}  med+late={(y_cls==0).sum()}', flush=True)

print('\nRegression (top-20 dims):')
r, r2, mae = cv_regression(X_gfp, y_reg, 'GFP mock top-20 reg')
results.append({'model': 'GFP top-20', 'type': 'reg', 'kind': 'mock', 'value': r})
results.append({'model': 'GFP top-20', 'type': 'reg', 'kind': 'real', 'value': REAL['GFP top-20  reg']})

print('\nClassification (top-20 dims):')
auc = cv_classify(X_gfp, y_cls, 'GFP mock top-20 cls')
results.append({'model': 'GFP top-20', 'type': 'cls', 'kind': 'mock', 'value': auc})
results.append({'model': 'GFP top-20', 'type': 'cls', 'kind': 'real', 'value': REAL['GFP top-20  cls']})

# ══════════════════════════════════════════════════════════════════════════════
# BF MOCK
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60, flush=True)
print('BF MOCK ANALYSIS', flush=True)
print('='*60, flush=True)

mock_bf = np.load(str(OUT_DIR / 'bf_mock_embeddings.npz'))
mock_bf_ids  = mock_bf['gfp_track_ids'].astype(int)
mock_bf_embs = mock_bf['embeddings']

# Filter to cells with labels (same as real analysis)
real_bf = np.load(str(REAL_BF_NPZ))
real_bf_id_to_row = {int(t): i for i, t in enumerate(real_bf['gfp_track_ids'])}
# Use only cells that are in both mock and real eligible set
eligible_ids = [t for t in mock_bf_ids if t in real_bf_id_to_row]
tids_bf, y_reg_bf, y_cls_bf = get_labels(eligible_ids)
mock_id_to_row = {int(t): i for i, t in enumerate(mock_bf_ids.tolist())}
rows_bf = [mock_id_to_row[t] for t in tids_bf]
X_bf_256 = mock_bf_embs[rows_bf]          # raw 256-dim
X_bf_top20 = X_bf_256[:, bf_top20_dims]   # locked top-20

print(f'n={len(tids_bf)}  early={y_cls_bf.sum()}  med+late={(y_cls_bf==0).sum()}', flush=True)

print('\nRaw 256-dim:')
r, r2, mae = cv_regression(X_bf_256, y_reg_bf, 'BF mock raw-256 reg')
results.append({'model': 'BF raw-256', 'type': 'reg', 'kind': 'mock', 'value': r})
results.append({'model': 'BF raw-256', 'type': 'reg', 'kind': 'real', 'value': REAL['BF raw-256  reg']})
auc = cv_classify(X_bf_256, y_cls_bf, 'BF mock raw-256 cls')
results.append({'model': 'BF raw-256', 'type': 'cls', 'kind': 'mock', 'value': auc})
results.append({'model': 'BF raw-256', 'type': 'cls', 'kind': 'real', 'value': REAL['BF raw-256  cls']})

print('\nTop-20 dims (locked from real analysis):')
r, r2, mae = cv_regression(X_bf_top20, y_reg_bf, 'BF mock top-20 reg')
results.append({'model': 'BF top-20', 'type': 'reg', 'kind': 'mock', 'value': r})
results.append({'model': 'BF top-20', 'type': 'reg', 'kind': 'real', 'value': REAL['BF top-20   reg']})
auc = cv_classify(X_bf_top20, y_cls_bf, 'BF mock top-20 cls')
results.append({'model': 'BF top-20', 'type': 'cls', 'kind': 'mock', 'value': auc})
results.append({'model': 'BF top-20', 'type': 'cls', 'kind': 'real', 'value': REAL['BF top-20   cls']})

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / 'mock_summary.csv', index=False)

print('\n\nSUMMARY TABLE', flush=True)
print('='*70, flush=True)
print(f'{"Model":<14} {"Task":<6} {"Real":>8} {"Mock":>8} {"Δ":>8}', flush=True)
print('-'*70, flush=True)
for model in ['GFP top-20', 'BF raw-256', 'BF top-20']:
    for task, label in [('reg', 'r'), ('cls', 'AUC')]:
        sub = df[(df['model'] == model) & (df['type'] == task)]
        real_v = sub[sub['kind'] == 'real']['value'].values[0]
        mock_v = sub[sub['kind'] == 'mock']['value'].values[0]
        print(f'{model:<14} {label:<6} {real_v:>8.3f} {mock_v:>8.3f} {mock_v-real_v:>+8.3f}',
              flush=True)
print('='*70, flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
models_order = ['GFP top-20', 'BF raw-256', 'BF top-20']
x = np.arange(len(models_order))
width = 0.35
colors = {'real': '#2196F3', 'mock': '#FF9800'}

for ax, (task, ylabel, chance) in zip(axes, [
    ('reg', 'Pearson r', 0.0),
    ('cls', 'AUC',       0.5),
]):
    for ki, kind in enumerate(['real', 'mock']):
        vals = [df[(df['model'] == m) & (df['type'] == task) &
                   (df['kind'] == kind)]['value'].values[0]
                for m in models_order]
        bars = ax.bar(x + ki * width - width/2, vals, width,
                      label=kind.capitalize(), color=colors[kind], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(chance, color='red', linestyle='--', linewidth=1, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{"Regression" if task == "reg" else "Classification"}: Real vs Mock',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(min(0, chance - 0.05) - 0.05, None)

fig.suptitle('Mock (random-crop) vs Real Embedding Performance\nSame labels, same frames, different crop location',
             fontsize=11)
plt.tight_layout()
fig.savefig(str(FIG_DIR / 'mock_vs_real_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nFigure saved → {FIG_DIR}/mock_vs_real_comparison.png', flush=True)
print('Done.', flush=True)

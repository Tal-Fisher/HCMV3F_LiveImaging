#!/usr/bin/env python3
"""
10_tabular_cls_wider_C.py

Re-run tabular classification with wider C grid (up to 1000) to check
if C=0.048 / nz=1 was due to grid truncation or genuine preference for sparsity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ    = BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
EXT_DF_CSV = LIVEIMG / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'

CUT_B2R = 1094
SEED    = 42

META_COLS = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
             'abs_gfp_onset_min', 'movie_half_min'}
EXTRAS_18 = {'cell_aspect_start', 'cell_aspect_mean', 'bfp_nuc_frac_start',
             'nuc_ratio_start', 'nuc_ratio_end',
             'bf_ctrst_start', 'bf_ctrst_end', 'bf_ctrst_slope'}

# ── Load same cells as 07_ ─────────────────────────────────────────────────────
d = np.load(str(EMB_NPZ))
emb_id_to_row = {int(tid): i for i, tid in enumerate(d['gfp_track_ids'])}

ext = pd.read_csv(EXT_DF_CSV)
tab_cols = [c for c in ext.columns if c not in META_COLS and c not in EXTRAS_18]
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id'] = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['delay_blue_to_red'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']

eligible = ext[
    ext['track_id'].isin(emb_id_to_row) &
    np.isfinite(ext['delay_blue_to_red'])
].sort_values('track_id').reset_index(drop=True)

y = (eligible['delay_blue_to_red'].values <= CUT_B2R).astype(int)
X_raw = eligible[tab_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
X = X_raw

print(f'n={len(y)}  early={y.sum()}  med+late={(y==0).sum()}')
print(f'Tabular features: {len(tab_cols)}\n')

LR_OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
LR_INNER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

for max_C, label in [(10, 'original  C up to 1e1 '),
                     (1000, 'wider     C up to 1e3 ')]:
    Cs = np.logspace(-3, np.log10(max_C), 30)
    LR_PARAMS = dict(
        penalty='elasticnet', solver='saga',
        l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
        Cs=Cs, cv=LR_INNER,
        class_weight='balanced', scoring='roc_auc',
        max_iter=5000, random_state=SEED, n_jobs=-1,
    )

    oof = np.zeros(len(y))
    for tr, te in LR_OUTER.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR_PARAMS)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]

    auc  = roc_auc_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    bal  = balanced_accuracy_score(y, pred)

    # full-data model to report selected C, l1, nz
    sc_full = StandardScaler()
    m_full  = LogisticRegressionCV(**{**LR_PARAMS,
                  'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)})
    m_full.fit(sc_full.fit_transform(X), y)
    nz = (m_full.coef_[0] != 0).sum()

    print(f'{label}: AUC={auc:.3f}  BalAcc={bal:.3f}  '
          f'C={m_full.C_[0]:.4f}  l1={m_full.l1_ratio_[0]:.2f}  nz={nz}/{X.shape[1]}')

    if max_C == 1000 and nz > 0:
        feat_names = np.array(tab_cols)
        order = np.argsort(np.abs(m_full.coef_[0]))[::-1]
        top = order[:min(10, nz)]
        print('  Top features (full-data model):')
        for rank, idx in enumerate(top, 1):
            print(f'    {rank}. {feat_names[idx]:40s}  beta={m_full.coef_[0][idx]:+.4f}')

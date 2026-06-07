#!/usr/bin/env python3
"""
09_dim148_only.py

5-fold CV regression using dim_148 alone from BF embeddings as predictor of
delay_blue_to_red. Tests whether the single feature kept by the raw 256-dim
ElasticNet has any genuine signal on its own.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ    = BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
EXT_DF_CSV = LIVEIMG / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'

CUT_B2R = 1094
SEED    = 42
DIM     = 148

META_COLS = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
             'abs_gfp_onset_min', 'movie_half_min'}
EXTRAS_18 = {'cell_aspect_start', 'cell_aspect_mean', 'bfp_nuc_frac_start',
             'nuc_ratio_start', 'nuc_ratio_end',
             'bf_ctrst_start', 'bf_ctrst_end', 'bf_ctrst_slope'}

# ── Load ───────────────────────────────────────────────────────────────────────
d = np.load(str(EMB_NPZ))
emb_track_ids = d['gfp_track_ids']
embeddings    = d['embeddings'].astype(np.float64)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}

ext = pd.read_csv(EXT_DF_CSV)
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id'] = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['delay_blue_to_red'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']

eligible = ext[
    ext['track_id'].isin(emb_id_to_row) &
    np.isfinite(ext['delay_blue_to_red'])
].sort_values('track_id').reset_index(drop=True)

y = eligible['delay_blue_to_red'].values
emb_rows = [emb_id_to_row[tid] for tid in eligible['track_id']]
X_full   = embeddings[emb_rows]               # (n, 256)
X        = X_full[:, DIM].reshape(-1, 1)      # single feature

print(f'n={len(y)}  early={( y <= CUT_B2R).sum()}  med+late={(y > CUT_B2R).sum()}')
print(f'Using dim_{DIM} only as predictor\n')

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_pred = np.full(len(y), np.nan)

for fold, (tr, te) in enumerate(kf.split(X)):
    model = LinearRegression()
    model.fit(X[tr], y[tr])
    oof_pred[te] = model.predict(X[te])

r   = pearsonr(y, oof_pred)[0]
r2  = r2_score(y, oof_pred)
mae = mean_absolute_error(y, oof_pred)

print('5-fold CV (dim_148 only, LinearRegression):')
print(f'  r   = {r:.4f}')
print(f'  R²  = {r2:.4f}')
print(f'  MAE = {mae:.1f} min')

# ── Baseline comparison ───────────────────────────────────────────────────────
print('\nBaseline (predict mean for every cell):')
mean_pred = np.full(len(y), y.mean())
print(f'  r   = {pearsonr(y, mean_pred)[0]:.4f}')
print(f'  R²  = {r2_score(y, mean_pred):.4f}')
print(f'  MAE = {mean_absolute_error(y, mean_pred):.1f} min')

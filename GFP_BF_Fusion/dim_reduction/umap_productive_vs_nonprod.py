#!/usr/bin/env python3
"""
umap_productive_vs_nonprod.py

UMAP on all GFP-expressing cells (productive + non-productive), using
*_all.npz embeddings. Colors: productive (turns red) vs non-productive.

Feature sets: BF (256), GFP (256), BF+GFP (512).
Output: dim_reduction/umap_prod_vs_nonprod_all.png  (1x3 panel)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap

BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')
CPE      = LIVEIMG / 'CellposeEmbedding' / 'embeddings'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
OUT_DIR  = BASE / 'dim_reduction'
OUT_DIR.mkdir(exist_ok=True)

DATASETS    = ['A2', 'A3']
COLOR_PROD    = '#e06c75'
COLOR_NONPROD = '#98c379'
ALPHA       = 0.55
S           = 18
UMAP_PARAMS = dict(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)

# ── Load embeddings ────────────────────────────────────────────────────────────
print('Loading embeddings...')
gfp_ids_all, gfp_embs_all = [], []
bf_ids_all,  bf_embs_all  = [], []
datasets_used = []

for ds in DATASETS:
    gfp_f = CPE  / f'{ds}_cell_embeddings_all.npz'
    bf_f  = BASE / 'embeddings' / f'{ds}_bf_at_gfp_onset_all.npz'
    if not gfp_f.exists() or not bf_f.exists():
        print(f'  {ds}: _all.npz not yet available — skipping')
        continue
    d_gfp = np.load(str(gfp_f))
    d_bf  = np.load(str(bf_f))
    gfp_ids_all.append(pd.DataFrame({'track_id': d_gfp['track_ids'].astype(int), 'dataset': ds}))
    gfp_embs_all.append(d_gfp['embeddings'].astype(np.float32))
    bf_ids_all.append(pd.DataFrame({'track_id': d_bf['track_ids'].astype(int), 'dataset': ds}))
    bf_embs_all.append(d_bf['embeddings'].astype(np.float32))
    datasets_used.append(ds)
    print(f'  {ds}: GFP {d_gfp["embeddings"].shape}  BF {d_bf["embeddings"].shape}')

if not datasets_used:
    raise SystemExit('No _all.npz embedding files found. Submit cluster jobs first.')

gfp_id_df = pd.concat(gfp_ids_all).reset_index(drop=True)
bf_id_df  = pd.concat(bf_ids_all).reset_index(drop=True)
GFP_EMB   = np.vstack(gfp_embs_all)
BF_EMB    = np.vstack(bf_embs_all)

# ── Load labels ────────────────────────────────────────────────────────────────
print('Loading labels...')
mdf  = pd.read_csv(MODEL_DF)
rows = []
for ds in datasets_used:
    sub = mdf[mdf['dataset'] == ds].copy()
    sub['track_id']   = sub['Track.ID'].str.replace(f'{ds}_', '', regex=False).astype(int)
    sub['productive'] = np.isfinite(sub['delay_green_to_red']).astype(int)
    rows.append(sub)
meta = pd.concat(rows).reset_index(drop=True)

gfp_key  = gfp_id_df.set_index(['dataset', 'track_id']).index
bf_key   = bf_id_df.set_index(['dataset', 'track_id']).index
meta_key = pd.MultiIndex.from_arrays([meta['dataset'], meta['track_id']])

eligible = meta[
    meta_key.isin(gfp_key) &
    meta_key.isin(bf_key)
].sort_values(['dataset', 'track_id']).reset_index(drop=True)

gfp_index = {(r.dataset, r.track_id): i for i, r in gfp_id_df.iterrows()}
bf_index  = {(r.dataset, r.track_id): i for i, r in bf_id_df.iterrows()}
gfp_rows  = [gfp_index[(r.dataset, r.track_id)] for _, r in eligible.iterrows()]
bf_rows   = [bf_index[(r.dataset, r.track_id)]  for _, r in eligible.iterrows()]

X_GFP = GFP_EMB[gfp_rows]
X_BF  = BF_EMB[bf_rows]
y     = eligible['productive'].values

n_prod    = y.sum()
n_nonprod = (1 - y).sum()
print(f'Cells: {len(y)} total  productive={n_prod}  non-productive={n_nonprod}')

feature_sets = [
    ('BF at GFP Onset', X_BF),
    ('GFP at Onset',    X_GFP),
    ('BF + GFP',        np.hstack([X_BF, X_GFP])),
]


def run_umap(X):
    X_sc = StandardScaler().fit_transform(X)
    return umap.UMAP(**UMAP_PARAMS).fit_transform(X_sc)


def make_ax(ax, Z, y, title):
    mask_prod    = y == 1
    mask_nonprod = y == 0
    ax.scatter(Z[mask_nonprod, 0], Z[mask_nonprod, 1],
               c=COLOR_NONPROD, alpha=ALPHA, s=S, linewidths=0,
               label=f'Non-productive (n={mask_nonprod.sum()})')
    ax.scatter(Z[mask_prod, 0], Z[mask_prod, 1],
               c=COLOR_PROD, alpha=ALPHA, s=S, linewidths=0,
               label=f'Productive (n={mask_prod.sum()})')
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=1.4, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)


fig_all, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (title, X) in enumerate(feature_sets):
    print(f'Running UMAP: {title}...', flush=True)
    Z = run_umap(X)
    make_ax(axes[idx], Z, y, title)

fig_all.suptitle('UMAP — Productive vs Non-Productive Cells at GFP Onset',
                 fontsize=13, fontweight='bold', y=1.02)
fig_all.tight_layout()
fig_all.savefig(OUT_DIR / 'umap_prod_vs_nonprod_all.png', dpi=300, bbox_inches='tight')
plt.close(fig_all)
print('saved umap_prod_vs_nonprod_all.png')

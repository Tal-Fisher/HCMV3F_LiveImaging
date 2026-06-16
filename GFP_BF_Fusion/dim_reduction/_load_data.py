"""Shared data loading for dim_reduction scripts."""

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')
CPE      = LIVEIMG / 'CellposeEmbedding' / 'embeddings'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

DATASETS = ['A2', 'A3']
CUT_B2R  = 1094


def load_embeddings_and_labels():
    """Return X_GFP, X_BF (n x 256), y (n,), eligible DataFrame."""
    print('Loading embeddings...')
    gfp_ids_all, gfp_embs_all = [], []
    bf_ids_all,  bf_embs_all  = [], []
    datasets_used = []

    for ds in DATASETS:
        gfp_f = CPE / f'{ds}_cell_embeddings.npz'
        bf_f  = BASE / 'embeddings' / f'{ds}_bf_at_gfp_onset.npz'
        if not gfp_f.exists() or not bf_f.exists():
            print(f'  {ds}: missing embeddings — skipping')
            continue
        d_gfp = np.load(str(gfp_f))
        d_bf  = np.load(str(bf_f))
        gfp_ids_all.append(pd.DataFrame({'track_id': d_gfp['track_ids'].astype(int), 'dataset': ds}))
        gfp_embs_all.append(d_gfp['embeddings'].astype(np.float32))
        bf_ids_all.append(pd.DataFrame({'track_id': d_bf['track_ids'].astype(int), 'dataset': ds}))
        bf_embs_all.append(d_bf['embeddings'].astype(np.float32))
        datasets_used.append(ds)
        print(f'  {ds}: GFP {d_gfp["embeddings"].shape}  BF {d_bf["embeddings"].shape}')

    gfp_id_df = pd.concat(gfp_ids_all).reset_index(drop=True)
    bf_id_df  = pd.concat(bf_ids_all).reset_index(drop=True)
    GFP_EMB   = np.vstack(gfp_embs_all)
    BF_EMB    = np.vstack(bf_embs_all)

    print('Loading labels...')
    mdf  = pd.read_csv(MODEL_DF)
    rows = []
    for ds in datasets_used:
        sub = mdf[mdf['dataset'] == ds].copy()
        sub['track_id'] = sub['Track.ID'].str.replace(f'{ds}_', '', regex=False).astype(int)
        sub['b2r']      = sub['delay_green_to_red'] - sub['delay_green_to_blue']
        rows.append(sub)
    meta = pd.concat(rows).reset_index(drop=True)

    gfp_key  = gfp_id_df.set_index(['dataset', 'track_id']).index
    bf_key   = bf_id_df.set_index(['dataset', 'track_id']).index
    meta_key = pd.MultiIndex.from_arrays([meta['dataset'], meta['track_id']])

    eligible = meta[
        meta_key.isin(gfp_key) &
        meta_key.isin(bf_key)  &
        meta['b2r'].notna()    &
        (meta['abs_gfp_onset_min'] <= meta['movie_half_min'])
    ].sort_values(['dataset', 'track_id']).reset_index(drop=True)

    gfp_index = {(r.dataset, r.track_id): i for i, r in gfp_id_df.iterrows()}
    bf_index  = {(r.dataset, r.track_id): i for i, r in bf_id_df.iterrows()}
    gfp_rows  = [gfp_index[(r.dataset, r.track_id)] for _, r in eligible.iterrows()]
    bf_rows   = [bf_index[(r.dataset, r.track_id)]  for _, r in eligible.iterrows()]

    X_GFP = GFP_EMB[gfp_rows]
    X_BF  = BF_EMB[bf_rows]
    y     = (eligible['b2r'].values <= CUT_B2R).astype(int)

    print(f'Cells after filtering: {len(y)}  fast={y.sum()}  slow={(1-y).sum()}')
    return X_GFP, X_BF, y

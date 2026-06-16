#!/usr/bin/env python3
"""
00_build_gfp_onset_all.py

Like 00_build_gfp_onset.py but includes ALL GFP-expressing A3 cells —
both productive (finite g2r) and non-productive (g2r = Inf).

Output: CompleteImage/A3_gfp_onset_all.csv
  track_id | gfp_onset_frame | x_at_onset | y_at_onset

Run on head node (pure pandas, no GPU):
  python GFP_BF_Fusion/00_build_gfp_onset_all.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging')
DATASET = 'A3'

SPOTS_CSV = BASE / 'CompleteImage' / f'{DATASET}_Merged_spots.csv'
MODEL_DF  = BASE / 'cache' / 'python_export' / 'model_df.csv'
OUT_CSV   = BASE / 'CompleteImage' / f'{DATASET}_gfp_onset_all.csv'

print('Loading model_df...', flush=True)
mdf = pd.read_csv(MODEL_DF)
ds  = mdf[mdf['dataset'] == DATASET].copy()
ds['track_id'] = ds['Track.ID'].str.replace(f'{DATASET}_', '', regex=False).astype(int)

# Include all cells with a GFP onset time (productive and non-productive)
all_cells = ds[ds['abs_gfp_onset_min'].notna()].copy()
n_prod    = np.isfinite(ds['delay_green_to_red']).sum()
n_nonprod = (~np.isfinite(ds['delay_green_to_red'])).sum()
print(f'  All {DATASET} cells with onset: {len(all_cells)}  '
      f'(productive={n_prod}, non-productive={n_nonprod})')

print(f'Loading {SPOTS_CSV.name}...', flush=True)
spots = pd.read_csv(SPOTS_CSV)
spots['_tid'] = spots['Track ID'].astype(str).str.extract(r'(\d+)$')[0].astype(int)
print(f'  {len(spots):,} spots, {spots["_tid"].nunique():,} unique tracks')

print('Matching onset times to frames...', flush=True)
records, skipped = [], []

for _, row in all_cells.iterrows():
    tid   = int(row['track_id'])
    t_sec = row['abs_gfp_onset_min'] * 60.0
    cell  = spots[spots['_tid'] == tid]
    if len(cell) == 0:
        skipped.append(tid)
        continue
    best = cell.loc[(cell['T (sec)'] - t_sec).abs().idxmin()]
    records.append(dict(
        track_id        = tid,
        gfp_onset_frame = int(best['Frame']),
        x_at_onset      = float(best['X']),
        y_at_onset      = float(best['Y']),
    ))

if skipped:
    print(f'  WARNING: {len(skipped)} tracks not found in spots CSV: {skipped}')

out_df = pd.DataFrame(records)

tid_to_tsec = dict(zip(all_cells['track_id'], all_cells['abs_gfp_onset_min'] * 60.0))
residuals = []
for rec in records:
    cell  = spots[spots['_tid'] == rec['track_id']]
    best_t = cell.loc[(cell['T (sec)'] - tid_to_tsec[rec['track_id']]).abs().idxmin(), 'T (sec)']
    residuals.append(abs(best_t - tid_to_tsec[rec['track_id']]))

print(f'  Time residuals:  median={np.median(residuals):.0f} s  max={np.max(residuals):.0f} s')
print(f'  Frame range: {out_df["gfp_onset_frame"].min()} – {out_df["gfp_onset_frame"].max()}')

out_df.to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}  ({len(out_df)} rows)')
print('Done.', flush=True)

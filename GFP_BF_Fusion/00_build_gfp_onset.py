#!/usr/bin/env python3
"""
00_build_gfp_onset.py

Build CompleteImage/A3_gfp_onset.csv — equivalent of A2_gfp_onset.csv.

For each productive A3 cell (finite delay_green_to_red AND delay_green_to_blue),
finds the spot in A3_Merged_spots.csv whose T (sec) is closest to
abs_gfp_onset_min * 60, and records that spot's Frame, X, Y.

Output columns: track_id | gfp_onset_frame | x_at_onset | y_at_onset

Run on head node (pure pandas, no GPU):
  python GFP_BF_Fusion/00_build_gfp_onset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging')
DATASET = 'A3'

SPOTS_CSV = BASE / 'CompleteImage' / f'{DATASET}_Merged_spots.csv'
MODEL_DF  = BASE / 'cache' / 'python_export' / 'model_df.csv'
OUT_CSV   = BASE / 'CompleteImage' / f'{DATASET}_gfp_onset.csv'

# ── Productive cells: finite b2r (both g2r and g2b finite) ────────────────────
print('Loading model_df...', flush=True)
mdf = pd.read_csv(MODEL_DF)
ds  = mdf[mdf['dataset'] == DATASET].copy()
ds['track_id'] = ds['Track.ID'].str.replace(f'{DATASET}_', '', regex=False).astype(int)
prod = ds[np.isfinite(ds['delay_green_to_red']) & np.isfinite(ds['delay_green_to_blue'])].copy()
print(f'  Productive A3 cells (finite b2r): {len(prod)}')

# ── Load spots, parse Track ID → integer ──────────────────────────────────────
print(f'Loading {SPOTS_CSV.name}...', flush=True)
spots = pd.read_csv(SPOTS_CSV)
spots['_tid'] = spots['Track ID'].astype(str).str.extract(r'(\d+)$')[0].astype(int)
print(f'  {len(spots):,} spots, {spots["_tid"].nunique():,} unique tracks')

# ── Match: for each productive cell, find spot closest to onset time ──────────
print('Matching onset times to frames...', flush=True)
records, skipped = [], []

for _, row in prod.iterrows():
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

# ── Sanity check: print time residuals ────────────────────────────────────────
out_df = pd.DataFrame(records)
tid_to_tsec = dict(zip(prod['track_id'], prod['abs_gfp_onset_min'] * 60.0))
residuals = []
for rec in records:
    cell  = spots[spots['_tid'] == rec['track_id']]
    best_t = cell.loc[(cell['T (sec)'] - tid_to_tsec[rec['track_id']]).abs().idxmin(), 'T (sec)']
    residuals.append(abs(best_t - tid_to_tsec[rec['track_id']]))

print(f'  Time residuals (|matched T − onset T|):  '
      f'median={np.median(residuals):.0f} s  max={np.max(residuals):.0f} s  '
      f'(~half a frame interval is expected)')
print(f'  Frame range: {out_df["gfp_onset_frame"].min()} – {out_df["gfp_onset_frame"].max()}')

# ── Save ──────────────────────────────────────────────────────────────────────
out_df.to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}  ({len(out_df)} rows)')
print('Done.', flush=True)

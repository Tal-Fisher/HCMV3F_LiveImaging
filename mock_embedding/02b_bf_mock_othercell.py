#!/usr/bin/env python3
"""
02b_bf_mock_othercell.py

Stronger null control for BF embedding analysis.
For each of the eligible A2 cells, extract a Cellpose neck embedding from a
DIFFERENT BF cell in the same frame (onset-10). The other cell is chosen
randomly from all BF allspot centroids at that frame, excluding the real
matched cell and any cell whose centroid falls within MIN_DIST_PX pixels
(so the crop doesn't substantially overlap the real crop).

Labels are unchanged. Compare results to:
  - Real BF embeddings (A2_bf_embeddings_m10_relaxed.npz)  → r=0.297, AUC=0.742
  - Random-crop mock   (bf_mock_embeddings.npz)            → r≈0, AUC≈0.5
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from cellpose import models

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging')
BF_DIR  = BASE / 'BrightFieldEmbedding'
OUT_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/mock_embedding')

MODEL_PATH   = str(BF_DIR / 'models' / 'cpsam_BrightField')
BF_TIFF      = BASE / 'CompleteImage' / 'A2_BrightField_raw.tif'
REAL_EMB_NPZ = BF_DIR / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
EXT_DF_CSV   = BASE / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'
MATCHES_CSV  = BF_DIR / 'bf_gfp_matches.csv'
BF_SPOTS_CSV = BASE / 'CompleteImage' / 'A2_BrightField_allspots.csv'

CROP_SIZE    = 256
HALF         = 128
DIAMETER     = 40
PIXEL_SCALE  = 0.2871   # µm per pixel
SEED         = 42
LOOKBACK     = 10
MIN_DIST_PX  = 150      # exclude other cells within this pixel distance of the real cell
                         # (~43 µm, just over 2 cell radii — ensures crops don't overlap)

# ── Eligible cells (same filter as real analysis) ────────────────────────────
print('Identifying eligible cells...', flush=True)
d = np.load(str(REAL_EMB_NPZ))
emb_track_ids = d['gfp_track_ids'].astype(int)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}

ext = pd.read_csv(EXT_DF_CSV)
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id'] = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['delay_blue_to_red'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']

eligible = ext[
    ext['track_id'].isin(emb_id_to_row) &
    np.isfinite(ext['delay_blue_to_red'])
].sort_values('track_id').reset_index(drop=True)

gfp_tids = eligible['track_id'].values
print(f'  {len(gfp_tids)} eligible cells', flush=True)

# ── Load BF match table ───────────────────────────────────────────────────────
print('Loading match table...', flush=True)
matches = pd.read_csv(MATCHES_CSV)
matches = matches[matches['gfp_track_id'].isin(set(gfp_tids.tolist()))].copy()
match_idx = matches.set_index('gfp_track_id')

# ── Load BF allspot positions (all cells, all frames) ────────────────────────
print('Loading BF allspot positions...', flush=True)
bf_all = pd.read_csv(BF_SPOTS_CSV,
                     usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y'],
                     low_memory=False)
for col in ['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y']:
    bf_all[col] = pd.to_numeric(bf_all[col], errors='coerce')
bf_all = bf_all.dropna(subset=['FRAME', 'POSITION_X', 'POSITION_Y'])
bf_all['FRAME'] = bf_all['FRAME'].astype(int)

# Per-frame list of (x_px, y_px, track_id) for all BF cells
# Convert µm → pixels once
bf_all['cx_px'] = (bf_all['POSITION_X'] / PIXEL_SCALE).round().astype(int)
bf_all['cy_px'] = (bf_all['POSITION_Y'] / PIXEL_SCALE).round().astype(int)
bf_all['TRACK_ID'] = bf_all['TRACK_ID'].fillna(-1).astype(int)

frame_cells = {}   # frame → array (N, 3): cx_px, cy_px, track_id
for frame, grp in bf_all.groupby('FRAME'):
    frame_cells[frame] = grp[['cx_px', 'cy_px', 'TRACK_ID']].values

# Real matched BF cell position at target frame
bf_pos_lookup = {
    (int(r['TRACK_ID']), int(r['FRAME'])): (int(r['cx_px']), int(r['cy_px']))
    for _, r in bf_all[bf_all['TRACK_ID'] >= 0].iterrows()
}

def real_crop_center(gfp_tid):
    if gfp_tid not in match_idx.index:
        return None, None, None, None
    row = match_idx.loc[gfp_tid]
    if pd.isna(row.get('bf_track_id', np.nan)):
        return None, None, None, None
    bf_tid = int(row['bf_track_id']); onset_f = int(row['onset_frame'])
    target_f = onset_f - LOOKBACK
    key = (bf_tid, target_f)
    if key not in bf_pos_lookup:
        return None, None, None, target_f
    cx, cy = bf_pos_lookup[key]
    return cx, cy, bf_tid, target_f

def other_cell_center(target_f, real_cx, real_cy, real_bf_tid, rng):
    """Return (cx_px, cy_px) of a randomly chosen OTHER BF cell in target_f."""
    if target_f not in frame_cells:
        return None, None
    cells = frame_cells[target_f]   # (N, 3): cx, cy, track_id
    candidates = []
    for cx, cy, tid in cells:
        if int(tid) == real_bf_tid:
            continue  # skip the real matched cell
        if real_cx is not None:
            dist_sq = (cx - real_cx)**2 + (cy - real_cy)**2
            if dist_sq < MIN_DIST_PX**2:
                continue  # too close — crops would overlap
        # keep cell within image bounds (crop must fit)
        if HALF <= cx < W - HALF and HALF <= cy < H - HALF:
            candidates.append((cx, cy))
    if not candidates:
        return None, None
    idx = rng.integers(0, len(candidates))
    return candidates[idx]

# ── Load Cellpose model ───────────────────────────────────────────────────────
print('Loading Cellpose model...', flush=True)
model = models.CellposeModel(gpu=True, pretrained_model=MODEL_PATH)

neck_outputs = []
def _hook_fn(module, inp, output):
    neck_outputs.append(output.detach().cpu().float())
hook_handle = model.net.encoder.neck.register_forward_hook(_hook_fn)

# ── Memmap BF TIFF ────────────────────────────────────────────────────────────
print(f'Memmapping {BF_TIFF.name}...', flush=True)
bf_img = tifffile.memmap(str(BF_TIFF))
T, H, W = bf_img.shape
print(f'  Shape: {bf_img.shape}', flush=True)

def get_crop(frame, cx, cy):
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = bf_img[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0-y0:iy0-y0+(iy1-iy0), ix0-x0:ix0-x0+(ix1-ix0)] = patch
    return crop

# ── Extract mock embeddings ───────────────────────────────────────────────────
print(f'\nExtracting other-cell mock embeddings...', flush=True)
out_ids  = []
out_embs = []
n_skipped = 0

for i, gfp_tid in enumerate(gfp_tids):
    real_cx, real_cy, real_bf_tid, target_f = real_crop_center(gfp_tid)

    if target_f is None:
        print(f'  [{i+1}/{len(gfp_tids)}] gfp_track {gfp_tid}: no match — skipping', flush=True)
        n_skipped += 1
        continue

    rng = np.random.default_rng(SEED + int(gfp_tid))
    result = other_cell_center(target_f, real_cx, real_cy, real_bf_tid or -1, rng)

    if result is None or result[0] is None:
        print(f'  [{i+1}/{len(gfp_tids)}] gfp_track {gfp_tid}: no other cell found — skipping', flush=True)
        n_skipped += 1
        continue

    cx, cy = result
    crop = get_crop(target_f, cx, cy)

    neck_outputs.clear()
    model.eval([crop], diameter=DIAMETER, channels=[0, 0],
               flow_threshold=0.8, cellprob_threshold=-1,
               min_size=100, do_3D=False)

    if not neck_outputs:
        print(f'  [{i+1}/{len(gfp_tids)}] gfp_track {gfp_tid}: no neck output — skipping', flush=True)
        n_skipped += 1
        continue

    emb = neck_outputs[0].mean(dim=[2, 3])[0].numpy().astype(np.float32)
    out_ids.append(gfp_tid)
    out_embs.append(emb)

    if (i + 1) % 10 == 0:
        print(f'  [{i+1}/{len(gfp_tids)}] done', flush=True)

hook_handle.remove()
print(f'\nExtracted {len(out_ids)} embeddings, skipped {n_skipped}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / 'bf_mock_othercell_embeddings.npz'
np.savez(str(out_path),
         gfp_track_ids=np.array(out_ids, dtype=np.int64),
         embeddings=np.array(out_embs, dtype=np.float32))
print(f'Saved → {out_path}', flush=True)

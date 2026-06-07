#!/usr/bin/env python3
"""
02_bf_mock_extract.py

Null control for BF embedding analysis.
For each of the 151 eligible A2 cells, extract a Cellpose neck embedding from a
RANDOM 256×256 crop in the same BF frame (onset-10), at least 500 px from the
real BF cell centre. Labels are unchanged — used in 03_mock_analysis.py.
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

CROP_SIZE        = 256
HALF             = 128
DIAMETER         = 40
PIXEL_SCALE      = 0.2871
SEED             = 42
LOOKBACK         = 10
EXCLUSION_RADIUS = 500

META_COLS = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
             'abs_gfp_onset_min', 'movie_half_min'}
EXTRAS_18 = {'cell_aspect_start', 'cell_aspect_mean', 'bfp_nuc_frac_start',
             'nuc_ratio_start', 'nuc_ratio_end',
             'bf_ctrst_start', 'bf_ctrst_end', 'bf_ctrst_slope'}

# ── Determine the same 151 eligible cells as in 07_bf_b2r_analysis.py ─────────
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
print(f'  {len(gfp_tids)} eligible cells')

# ── Build real BF crop-centre lookup ──────────────────────────────────────────
print('Building BF position lookup...', flush=True)
matches = pd.read_csv(MATCHES_CSV)
matches = matches[matches['gfp_track_id'].isin(set(gfp_tids.tolist()))].copy()
match_idx = matches.set_index('gfp_track_id')

# Load only needed columns from the large BF allspots CSV
bf_all = pd.read_csv(BF_SPOTS_CSV, usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y'],
                     low_memory=False)
bf_all = bf_all.drop_duplicates(subset=['TRACK_ID', 'FRAME'])
bf_pos = bf_all.set_index(['TRACK_ID', 'FRAME'])[['POSITION_X', 'POSITION_Y']].to_dict('index')
print(f'  BF allspots loaded: {len(bf_pos)} (track,frame) entries', flush=True)

def real_crop_center(gfp_tid):
    if gfp_tid not in match_idx.index:
        return None, None
    row = match_idx.loc[gfp_tid]
    if pd.isna(row.get('bf_track_id', np.nan)):
        return None, None
    bf_tid      = int(row['bf_track_id'])
    onset_frame = int(row['onset_frame'])
    target_f    = onset_frame - LOOKBACK
    key = (bf_tid, target_f)
    if key not in bf_pos:
        return None, None
    x_um = bf_pos[key]['POSITION_X']
    y_um = bf_pos[key]['POSITION_Y']
    cx   = int(round(x_um / PIXEL_SCALE))
    cy   = int(round(y_um / PIXEL_SCALE))
    return cx, cy

# ── Load Cellpose BF model ─────────────────────────────────────────────────────
print('Loading BF Cellpose model...', flush=True)
model = models.CellposeModel(gpu=True, pretrained_model=MODEL_PATH)

neck_outputs = []
def _hook_fn(module, inp, output):
    neck_outputs.append(output.detach().cpu().float())
hook_handle = model.net.encoder.neck.register_forward_hook(_hook_fn)

# ── Memmap BF TIFF ─────────────────────────────────────────────────────────────
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

def random_center(real_cx, real_cy, rng):
    if real_cx is None:
        # No exclusion zone available — fully random
        return int(rng.integers(HALF, W - HALF)), int(rng.integers(HALF, H - HALF))
    for _ in range(10000):
        rx = int(rng.integers(HALF, W - HALF))
        ry = int(rng.integers(HALF, H - HALF))
        if (rx - real_cx)**2 + (ry - real_cy)**2 >= EXCLUSION_RADIUS**2:
            return rx, ry
    return HALF + 2000, HALF + 2000

# ── Extract mock embeddings ────────────────────────────────────────────────────
print(f'\nExtracting mock embeddings...', flush=True)
out_ids  = []
out_embs = []
n = len(gfp_tids)

for i, gfp_tid in enumerate(gfp_tids):
    # Get the onset-10 frame from match table
    if gfp_tid not in match_idx.index:
        print(f'  WARNING: gfp_track {gfp_tid} not in match table — skipping', flush=True)
        continue

    mrow        = match_idx.loc[gfp_tid]
    onset_frame = int(mrow['onset_frame'])
    target_f    = onset_frame - LOOKBACK

    real_cx, real_cy = real_crop_center(gfp_tid)

    rng    = np.random.default_rng(SEED + int(gfp_tid))
    rx, ry = random_center(real_cx, real_cy, rng)

    crop = get_crop(target_f, rx, ry)

    neck_outputs.clear()
    model.eval([crop], diameter=DIAMETER, channels=[0, 0],
               flow_threshold=0.8, cellprob_threshold=-1,
               min_size=100, do_3D=False)

    if not neck_outputs:
        print(f'  WARNING: no neck output for gfp_track {gfp_tid}', flush=True)
        continue

    emb = neck_outputs[0].mean(dim=[2, 3])[0].numpy().astype(np.float32)
    out_ids.append(gfp_tid)
    out_embs.append(emb)

    if (i + 1) % 30 == 0:
        print(f'  {i+1}/{n} cells done', flush=True)

hook_handle.remove()

# ── Save ───────────────────────────────────────────────────────────────────────
out_ids  = np.array(out_ids,  dtype=np.int64)
out_embs = np.array(out_embs, dtype=np.float32)
out_path = OUT_DIR / 'bf_mock_embeddings.npz'
np.savez(str(out_path), gfp_track_ids=out_ids, embeddings=out_embs)
print(f'\nSaved {len(out_ids)} mock BF embeddings → {out_path}')
print(f'Embedding stats: mean={out_embs.mean():.4f}  std={out_embs.std():.4f}')
print('Done.', flush=True)

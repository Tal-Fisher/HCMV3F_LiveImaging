#!/usr/bin/env python3
"""
01_gfp_mock_extract.py

Null control for GFP embedding analysis.
For each of the 291 A2 cells, extract a Cellpose neck embedding from a RANDOM
256×256 crop in the SAME GFP frame, but at least 500 px from the real cell
centre. Labels are unchanged — used in 03_mock_analysis.py.
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from cellpose import models

BASE   = Path('/home/labs/ginossar/talfis/LiveImaging')
CE_DIR = BASE / 'CellposeEmbedding'
OUT_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/mock_embedding')

MODEL_PATH   = str(CE_DIR / 'Cellpose_Cells_Model' / 'cpsam_20260328_104454')
GFP_TIFF     = CE_DIR / 'B2_GFP_raw.tif'
REAL_EMB_NPZ = CE_DIR / 'embeddings' / 'A2_cell_embeddings.npz'
ONSET_CSV    = BASE / 'CompleteImage' / 'A2_gfp_onset.csv'

CROP_SIZE        = 256
HALF             = 128
DIAMETER         = 40
PIXEL_SCALE      = 0.2871
SEED             = 42
EXCLUSION_RADIUS = 500

# ── Load real track IDs ────────────────────────────────────────────────────────
print('Loading real embedding track IDs...', flush=True)
d = np.load(str(REAL_EMB_NPZ))
track_ids = d['track_ids'].astype(int)
print(f'  {len(track_ids)} cells')

# ── Load onset metadata ────────────────────────────────────────────────────────
onset = pd.read_csv(ONSET_CSV).set_index('track_id')

# ── Load Cellpose model ────────────────────────────────────────────────────────
print('Loading Cellpose model...', flush=True)
model = models.CellposeModel(gpu=True, pretrained_model=MODEL_PATH)

neck_outputs = []
def _hook_fn(module, inp, output):
    neck_outputs.append(output.detach().cpu().float())
hook_handle = model.net.encoder.neck.register_forward_hook(_hook_fn)

# ── Memmap GFP TIFF ────────────────────────────────────────────────────────────
print(f'Memmapping {GFP_TIFF.name}...', flush=True)
gfp = tifffile.memmap(str(GFP_TIFF))
T, H, W = gfp.shape
print(f'  Shape: {gfp.shape}', flush=True)

def get_crop(frame, cx, cy):
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = gfp[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0-y0:iy0-y0+(iy1-iy0), ix0-x0:ix0-x0+(ix1-ix0)] = patch
    return crop

def random_center(real_cx, real_cy, rng):
    for _ in range(10000):
        rx = int(rng.integers(HALF, W - HALF))
        ry = int(rng.integers(HALF, H - HALF))
        if (rx - real_cx)**2 + (ry - real_cy)**2 >= EXCLUSION_RADIUS**2:
            return rx, ry
    # Fallback (rejection very unlikely to fail given image is 9051×9357)
    return HALF + 2000, HALF + 2000

# ── Extract mock embeddings ────────────────────────────────────────────────────
print(f'\nExtracting mock embeddings...', flush=True)
out_ids  = []
out_embs = []
n = len(track_ids)

for i, tid in enumerate(track_ids):
    if tid not in onset.index:
        print(f'  WARNING: track {tid} not in onset table — skipping', flush=True)
        continue

    orow    = onset.loc[tid]
    frame   = int(orow['gfp_onset_frame'])
    real_cx = int(round(float(orow['x_at_onset']) / PIXEL_SCALE))
    real_cy = int(round(float(orow['y_at_onset']) / PIXEL_SCALE))

    rng     = np.random.default_rng(SEED + int(tid))
    rx, ry  = random_center(real_cx, real_cy, rng)

    crop = get_crop(frame, rx, ry)

    neck_outputs.clear()
    model.eval([crop], diameter=DIAMETER, channels=[0, 0],
               flow_threshold=0.8, cellprob_threshold=-1,
               min_size=100, do_3D=False)

    if not neck_outputs:
        print(f'  WARNING: no neck output for track {tid}', flush=True)
        continue

    emb = neck_outputs[0].mean(dim=[2, 3])[0].numpy().astype(np.float32)
    out_ids.append(tid)
    out_embs.append(emb)

    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{n} cells done', flush=True)

hook_handle.remove()

# ── Save ───────────────────────────────────────────────────────────────────────
out_ids  = np.array(out_ids,  dtype=np.int64)
out_embs = np.array(out_embs, dtype=np.float32)
out_path = OUT_DIR / 'gfp_mock_embeddings.npz'
np.savez(str(out_path), gfp_track_ids=out_ids, embeddings=out_embs)
print(f'\nSaved {len(out_ids)} mock GFP embeddings → {out_path}')
print(f'Embedding stats: mean={out_embs.mean():.4f}  std={out_embs.std():.4f}')
print('Done.', flush=True)

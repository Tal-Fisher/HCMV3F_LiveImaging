#!/usr/bin/env python3
"""
Extract 256-dim Cellpose SAM neck embeddings for 291 productive A2 cells.
Embedding taken at the GFP onset frame (first GFP-positive frame per cell).

Output:
  embeddings/A2_cell_embeddings.npz  -- keys: track_ids (int64), embeddings (float32, N×256)
  embeddings/A2_cell_embeddings.csv  -- track_id + emb_0 … emb_255
  figures/sample_crops_5.png         -- 256×256 GFP crops for 5 random cells
"""

import sys
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')

MODEL_PATH = str(BASE / 'Cellpose_Cells_Model' / 'cpsam_20260328_104454')
GFP_TIFF   = BASE / 'A2_GFP_raw.tif'
ONSET_CSV  = LIVEIMG / 'CompleteImage' / 'A2_gfp_onset.csv'
MODEL_DF   = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

OUT_DIR     = BASE / 'embeddings'
FIGURES_DIR = BASE / 'figures'
OUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

PIXEL_SCALE    = 0.2871   # µm/px
CROP_SIZE      = 256
HALF           = CROP_SIZE // 2
DIAMETER       = 40
N_SAMPLE_CROPS = 5
RANDOM_SEED    = 42

# ── Load model ─────────────────────────────────────────────────────────────
print('Loading Cellpose model...', flush=True)
from cellpose import models
model = models.CellposeModel(pretrained_model=MODEL_PATH, gpu=True)
device = next(model.net.parameters()).device
print(f'  device: {device}', flush=True)

# ── Register neck hook ─────────────────────────────────────────────────────
# model.eval() style vector (result[2]) is always zeros in Cellpose 4 SAM —
# hardcoded stub for CP3 compatibility (vit_sam.py line 82).
# Real embedding is captured from model.net.encoder.neck via forward hook.
neck_outputs = []

def _hook_fn(module, inp, output):
    neck_outputs.append(output.detach().cpu().float())

hook_handle = model.net.encoder.neck.register_forward_hook(_hook_fn)

# ── Load metadata ──────────────────────────────────────────────────────────
print('Loading metadata...', flush=True)
df    = pd.read_csv(MODEL_DF)
onset = pd.read_csv(ONSET_CSV)

a2 = df[(df['dataset'] == 'A2') & np.isfinite(df['delay_green_to_red'])].copy()
a2['track_id'] = a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
onset_idx = onset.set_index('track_id')

print(f'  Productive A2 cells: {len(a2)}', flush=True)

rng        = np.random.default_rng(RANDOM_SEED)
sample_ids = set(rng.choice(a2['track_id'].values, size=N_SAMPLE_CROPS, replace=False).tolist())

# ── Memmap GFP TIFF ────────────────────────────────────────────────────────
print(f'Memmapping {GFP_TIFF.name}...', flush=True)
gfp = tifffile.memmap(str(GFP_TIFF))
T, H, W = gfp.shape
print(f'  Shape: {gfp.shape}, dtype: {gfp.dtype}', flush=True)

def get_crop(frame, cx, cy):
    """256×256 uint8 crop centred on (cx, cy), zero-padded at image boundaries."""
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = gfp[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

# ── Extract embeddings ─────────────────────────────────────────────────────
track_ids_out  = []
embeddings_out = []
sample_crops   = {}   # track_id → crop

print(f'Extracting embeddings...', flush=True)
n = len(a2)
for i, (_, row) in enumerate(a2.iterrows()):
    tid = int(row['track_id'])

    if tid not in onset_idx.index:
        print(f'  WARNING: track {tid} not in onset table — skipping', flush=True)
        continue

    orow  = onset_idx.loc[tid]
    frame = int(orow['gfp_onset_frame'])
    cx    = int(round(float(orow['x_at_onset']) / PIXEL_SCALE))
    cy    = int(round(float(orow['y_at_onset']) / PIXEL_SCALE))

    crop = get_crop(frame, cx, cy)
    if tid in sample_ids:
        sample_crops[tid] = (crop.copy(), frame)

    neck_outputs.clear()
    model.eval(
        [crop],
        diameter=DIAMETER,
        channels=[0, 0],
        flow_threshold=0.8,
        cellprob_threshold=-1,
        min_size=100,
        do_3D=False,
    )

    if not neck_outputs:
        print(f'  WARNING: no neck output for track {tid}', flush=True)
        continue

    # shape (batch, 256, H', W') → global avg pool → (256,)
    emb = neck_outputs[0].mean(dim=[2, 3])[0].numpy().astype(np.float32)

    track_ids_out.append(tid)
    embeddings_out.append(emb)

    if i == 0:
        print(f'  First cell — emb shape: {emb.shape}, std: {emb.std():.4f}  [sanity check]',
              flush=True)
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{n} cells done', flush=True)

hook_handle.remove()

# ── Save embeddings ────────────────────────────────────────────────────────
track_ids_arr  = np.array(track_ids_out,  dtype=np.int64)
embeddings_arr = np.array(embeddings_out, dtype=np.float32)   # (N, 256)

print(f'\nEmbeddings shape : {embeddings_arr.shape}')
print(f'Embedding stats  : mean={embeddings_arr.mean():.4f}  std={embeddings_arr.std():.4f}')

npz_path = OUT_DIR / 'A2_cell_embeddings.npz'
np.savez(str(npz_path), track_ids=track_ids_arr, embeddings=embeddings_arr)
print(f'Saved: {npz_path}')

csv_path = OUT_DIR / 'A2_cell_embeddings.csv'
dim = embeddings_arr.shape[1]
emb_df = pd.DataFrame(
    np.column_stack([track_ids_arr[:, None], embeddings_arr]),
    columns=['track_id'] + [f'emb_{k}' for k in range(dim)],
)
emb_df['track_id'] = emb_df['track_id'].astype(int)
emb_df.to_csv(str(csv_path), index=False)
print(f'Saved: {csv_path}')

# ── Save sample crop PNG ───────────────────────────────────────────────────
print(f'\nSaving {N_SAMPLE_CROPS} sample crops...')
fig, axes = plt.subplots(1, N_SAMPLE_CROPS, figsize=(4 * N_SAMPLE_CROPS, 4))
for ax, (tid, (crop, frame)) in zip(axes, sample_crops.items()):
    vmax = np.percentile(crop, 99.5) if crop.max() > 0 else 1
    ax.imshow(crop, cmap='gray', vmin=0, vmax=vmax)
    ax.set_title(f'Track {tid}\nframe {frame}', fontsize=9)
    ax.axis('off')
fig.suptitle('GFP onset crops (256×256 px)', fontsize=11)
plt.tight_layout()
png_path = FIGURES_DIR / 'sample_crops_5.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {png_path}')

print('\nDone.', flush=True)

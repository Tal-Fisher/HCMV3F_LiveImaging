#!/usr/bin/env python3
"""
Crop raw GFP+BF pixel patches (256x256, no Cellpose) at the GFP onset
frame/coords, for ALL GFP-expressing cells (productive + non-productive).

This is the raw-pixel analogue of GFP_BF_Fusion/01_extract_bf_at_gfp_coords_all.py
(same get_crop() logic, same onset CSVs, same PIXEL_SCALE) but skips the
Cellpose forward pass entirely -- it just crops and stacks GFP+BF as a
2-channel uint8 array per cell. No GPU needed.

Usage:
  python 01_extract_raw_patches.py --dataset A2
  python 01_extract_raw_patches.py --dataset A3

Output:
  CNN/patches/{DATASET}_patches_all.npz
    track_ids   int64,   (N,)
    patches     uint8,   (N, 2, 256, 256)   channel 0 = GFP, channel 1 = BF
    norm_p1     float32, (2,)   1st percentile per channel (sampled)
    norm_p995   float32, (2,)   99.5th percentile per channel (sampled)
  CNN/patches/{DATASET}_patches_all.csv   -- track_id, gfp_onset_frame
  CNN/figures/sample_patches_{DATASET}.png
"""

import argparse
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='A2', choices=['A2', 'A3'])
args = parser.parse_args()
DATASET = args.dataset

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CNN')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

ONSET_CSV_PATHS = {
    'A2': LIVEIMG / 'CompleteImage' / 'A2_gfp_onset.csv',
    'A3': LIVEIMG / 'CompleteImage' / 'A3_gfp_onset_all.csv',
}
ONSET_CSV = ONSET_CSV_PATHS[DATASET]

GFP_TIFF_PATHS = {
    'A2': LIVEIMG / 'CellposeEmbedding' / 'A2_GFP_raw.tif',
    'A3': LIVEIMG / 'CompleteImage' / 'A3_GFP.tif',
}
BF_TIFF_PATHS = {
    'A2': LIVEIMG / 'CompleteImage' / 'A2_BrightField_raw.tif',
    'A3': LIVEIMG / 'CompleteImage' / 'A3_BrightField.tif',
}
GFP_TIFF = GFP_TIFF_PATHS[DATASET]
BF_TIFF  = BF_TIFF_PATHS[DATASET]

OUT_DIR = BASE / 'patches'
FIG_DIR = BASE / 'figures'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

PIXEL_SCALE     = 0.2871   # micrometers per pixel
CROP_SIZE       = 256
HALF            = CROP_SIZE // 2
N_SAMPLE_CROPS  = 5
N_PCTL_SAMPLE   = 2000
RANDOM_SEED     = 42

print(f'Dataset: {DATASET}', flush=True)

for p in (GFP_TIFF, BF_TIFF):
    if not p.exists():
        raise FileNotFoundError(f"tif not found: {p}")

print('Loading metadata...', flush=True)
df    = pd.read_csv(MODEL_DF)
onset = pd.read_csv(ONSET_CSV)

cells = df[df['dataset'] == DATASET].copy()
cells['track_id'] = cells['Track.ID'].str.replace(f'{DATASET}_', '', regex=False).astype(int)
onset_idx = onset.set_index('track_id')

n_prod    = np.isfinite(cells['delay_green_to_red']).sum()
n_nonprod = (~np.isfinite(cells['delay_green_to_red'])).sum()
print(f'  {DATASET} cells: {len(cells)} total  productive={n_prod}  non-productive={n_nonprod}')
print(f'  Onset CSV rows: {len(onset)}')

rng        = np.random.default_rng(RANDOM_SEED)
sample_ids = set(rng.choice(cells['track_id'].values, size=N_SAMPLE_CROPS, replace=False).tolist())

print(f'Memmapping {GFP_TIFF.name} and {BF_TIFF.name}...', flush=True)
gfp = tifffile.memmap(str(GFP_TIFF))
bf  = tifffile.memmap(str(BF_TIFF))
_, H, W = bf.shape
print(f'  GFP shape: {gfp.shape}, dtype: {gfp.dtype}', flush=True)
print(f'  BF  shape: {bf.shape}, dtype: {bf.dtype}', flush=True)


def get_crop(img, frame, cx, cy):
    """256x256 crop centred on (cx, cy), zero-padded at boundaries."""
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = img[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop


track_ids_out = []
patches_out   = []
frames_out    = []
sample_crops  = {}

print('Extracting patches...', flush=True)
n = len(cells)
for i, (_, row) in enumerate(cells.iterrows()):
    tid = int(row['track_id'])

    if tid not in onset_idx.index:
        print(f'  WARNING: track {tid} not in onset table -- skipping', flush=True)
        continue

    orow  = onset_idx.loc[tid]
    frame = int(orow['gfp_onset_frame'])
    cx    = int(round(float(orow['x_at_onset']) / PIXEL_SCALE))
    cy    = int(round(float(orow['y_at_onset']) / PIXEL_SCALE))

    gfp_crop = get_crop(gfp, frame, cx, cy)
    bf_crop  = get_crop(bf,  frame, cx, cy)
    patch    = np.stack([gfp_crop, bf_crop], axis=0)  # (2, 256, 256)

    if tid in sample_ids:
        sample_crops[tid] = (gfp_crop.copy(), bf_crop.copy(), frame)

    track_ids_out.append(tid)
    patches_out.append(patch)
    frames_out.append(frame)

    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{n} cells done', flush=True)

track_ids_arr = np.array(track_ids_out, dtype=np.int64)
patches_arr   = np.array(patches_out,   dtype=np.uint8)  # (N, 2, 256, 256)

print(f'\nPatches shape: {patches_arr.shape}', flush=True)

# ── Fixed per-channel percentile normalisation constants (movie-level) ─────
pctl_sample_idx = rng.choice(len(patches_arr), size=min(N_PCTL_SAMPLE, len(patches_arr)),
                              replace=False)
norm_p1   = np.array([np.percentile(patches_arr[pctl_sample_idx, c], 1.0)   for c in (0, 1)],
                      dtype=np.float32)
norm_p995 = np.array([np.percentile(patches_arr[pctl_sample_idx, c], 99.5) for c in (0, 1)],
                      dtype=np.float32)
print(f'  norm_p1   (GFP, BF) = {norm_p1}', flush=True)
print(f'  norm_p995 (GFP, BF) = {norm_p995}', flush=True)

# ── Save ─────────────────────────────────────────────────────────────────
npz_path = OUT_DIR / f'{DATASET}_patches_all.npz'
np.savez_compressed(str(npz_path), track_ids=track_ids_arr, patches=patches_arr,
                     norm_p1=norm_p1, norm_p995=norm_p995)
print(f'Saved: {npz_path}')

csv_path = OUT_DIR / f'{DATASET}_patches_all.csv'
pd.DataFrame({'track_id': track_ids_arr, 'gfp_onset_frame': frames_out}).to_csv(
    str(csv_path), index=False)
print(f'Saved: {csv_path}')

# ── Sample crops PNG (GFP + BF side by side, color-coded by fate) ─────────
cells_idx = cells.set_index('track_id')
b2r = cells_idx['delay_green_to_red'] - cells_idx['delay_green_to_blue']
CUT_B2R = 1094

fig, axes = plt.subplots(2, N_SAMPLE_CROPS, figsize=(4 * N_SAMPLE_CROPS, 8))
for col, (tid, (g_crop, b_crop, frame)) in enumerate(sample_crops.items()):
    val = b2r.loc[tid] if tid in b2r.index else np.nan
    if not np.isfinite(val):
        status, colour = 'non-productive', 'gray'
    elif val <= CUT_B2R:
        status, colour = 'fast', 'tab:red'
    else:
        status, colour = 'slow', 'tab:blue'

    for row, (crop, label) in enumerate([(g_crop, 'GFP'), (b_crop, 'BF')]):
        ax = axes[row, col]
        vmax = np.percentile(crop, 99.5) if crop.max() > 0 else 1
        ax.imshow(crop, cmap='gray', vmin=0, vmax=vmax)
        ax.axis('off')
        if row == 0:
            ax.set_title(f'Track {tid} ({status})\nframe {frame}', fontsize=9, color=colour)
        else:
            ax.set_title(label, fontsize=8)

fig.suptitle(f'Raw GFP+BF patches at GFP onset -- {DATASET} (256x256 px)', fontsize=12)
plt.tight_layout()
png_path = FIG_DIR / f'sample_patches_{DATASET}.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {png_path}')

print('\nDone.', flush=True)

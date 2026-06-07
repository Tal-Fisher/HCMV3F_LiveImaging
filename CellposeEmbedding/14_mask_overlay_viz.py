#!/usr/bin/env python3
"""
14_mask_overlay_viz.py

Visualise 10 random cells from each embedding analysis with their cell mask
overlaid in red (alpha=0.5) to confirm crops are properly centred.

GFP  : crops at GFP onset frame from B2_GFP_raw.tif
       masks from Mosaic001_Merged-cellpose.tif (global segmentation)
BF   : crops at onset-10 from A2_BrightField_raw.tif
       masks from running the BF Cellpose model on each crop

Outputs:
  CellposeEmbedding/figures/gfp_mask_overlay_10cells.png
  BrightFieldEmbedding/figures/bf_mask_overlay_10cells.png
"""

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_GFP  = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
BASE_BF   = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG   = Path('/home/labs/ginossar/talfis/LiveImaging')

GFP_TIFF    = BASE_GFP / 'B2_GFP_raw.tif'
GFP_MASK    = BASE_GFP / 'Mosaic001_Merged-cellpose.tif'
BF_TIFF     = LIVEIMG / 'CompleteImage' / 'A2_BrightField_raw.tif'
BF_MODEL    = str(BASE_BF / 'models' / 'cpsam_BrightField')
ONSET_CSV   = LIVEIMG / 'CompleteImage' / 'A2_gfp_onset.csv'
MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
MATCHES_CSV = BASE_BF / 'bf_gfp_matches.csv'
ALLSPOTS    = LIVEIMG / 'CompleteImage' / 'A2_BrightField_allspots.csv'

PIXEL_SCALE = 0.2871    # µm/px
CROP_SIZE   = 256
HALF        = CROP_SIZE // 2
N_CELLS     = 10
SEED        = 123
LOOKBACK    = 10        # frames before GFP onset for BF crops

# ── Load BF Cellpose model (GPU) ────────────────────────────────────────────
print('Loading BF Cellpose model...', flush=True)
from cellpose import models
bf_model = models.CellposeModel(pretrained_model=BF_MODEL, gpu=True)

# ── Memmap raw images ───────────────────────────────────────────────────────
print('Memmapping GFP image...', flush=True)
gfp_img = tifffile.memmap(str(GFP_TIFF))
T, H, W = gfp_img.shape
print(f'  GFP shape: {gfp_img.shape}', flush=True)

print('Memmapping GFP mask...', flush=True)
gfp_mask_vol = tifffile.memmap(str(GFP_MASK))
print(f'  GFP mask shape: {gfp_mask_vol.shape}  dtype: {gfp_mask_vol.dtype}', flush=True)

print('Memmapping BF image...', flush=True)
bf_img = tifffile.memmap(str(BF_TIFF))
print(f'  BF shape: {bf_img.shape}', flush=True)


# ── Crop helpers ────────────────────────────────────────────────────────────
def get_img_crop(vol, frame, cx, cy):
    """256×256 crop, zero-padded at boundaries."""
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = np.asarray(vol[frame, iy0:iy1, ix0:ix1])
    crop = np.zeros((CROP_SIZE, CROP_SIZE), dtype=patch.dtype)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop


def get_mask_crop(mask_vol, frame, cx, cy):
    """256×256 float32 label crop, zero-padded (label 0 = background)."""
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    # np.asarray handles big-endian float32 byte swap transparently
    patch = np.asarray(mask_vol[frame, iy0:iy1, ix0:ix1]).astype(np.float32)
    crop = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop


def overlay_and_draw(ax, img_crop, binary_mask, title):
    """Draw grayscale image with red (alpha=0.5) mask overlay and a crosshair."""
    vmin = float(np.percentile(img_crop[img_crop > 0], 0.5)) if img_crop.max() > 0 else 0
    vmax = float(np.percentile(img_crop, 99.5)) if img_crop.max() > 0 else 1
    ax.imshow(img_crop, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    if binary_mask.any():
        rgba = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
        rgba[binary_mask, 0] = 1.0   # R
        rgba[binary_mask, 3] = 0.5   # alpha
        ax.imshow(rgba, interpolation='nearest')
    # Small crosshair at crop centre
    ax.axhline(HALF, color='cyan', linewidth=0.5, alpha=0.6)
    ax.axvline(HALF, color='cyan', linewidth=0.5, alpha=0.6)
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis('off')


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — GFP crops
# ═══════════════════════════════════════════════════════════════════════════
print('\n=== GFP crops ===', flush=True)

df    = pd.read_csv(MODEL_DF)
onset = pd.read_csv(ONSET_CSV)

a2 = df[(df['dataset'] == 'A2') & np.isfinite(df['delay_green_to_red'])].copy()
a2['track_id'] = a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
onset_idx = onset.set_index('track_id')

rng = np.random.default_rng(SEED)
sample_gfp = rng.choice(a2['track_id'].values, size=N_CELLS, replace=False)
print(f'  Sampled track IDs: {sample_gfp}', flush=True)

fig_gfp, axes_gfp = plt.subplots(2, 5, figsize=(22, 9))
axes_gfp = axes_gfp.flatten()

n_found = 0
n_zero_label = 0

for i, tid in enumerate(sample_gfp):
    if tid not in onset_idx.index:
        axes_gfp[i].set_title(f'Track {tid}\n(no onset row)', fontsize=7)
        axes_gfp[i].axis('off')
        print(f'  [{i+1}] track {tid}: NOT in onset table', flush=True)
        continue

    orow  = onset_idx.loc[tid]
    frame = int(orow['gfp_onset_frame'])
    cx    = int(round(float(orow['x_at_onset']) / PIXEL_SCALE))
    cy    = int(round(float(orow['y_at_onset']) / PIXEL_SCALE))

    img_crop  = get_img_crop(gfp_img, frame, cx, cy)
    mask_crop = get_mask_crop(gfp_mask_vol, frame, cx, cy)

    center_label = mask_crop[HALF, HALF]
    if center_label != 0:
        binary_mask = (mask_crop == center_label)
    else:
        binary_mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)
        n_zero_label += 1

    title = (f'Track {tid}  frame {frame}\n'
             f'cx={cx} cy={cy}  label={int(center_label)}')
    overlay_and_draw(axes_gfp[i], img_crop, binary_mask, title)
    n_found += 1
    print(f'  [{i+1}] track={tid}  frame={frame}  cx={cx}  cy={cy}  '
          f'label={int(center_label)}  mask_px={binary_mask.sum()}', flush=True)

fig_gfp.suptitle(
    f'GFP onset crops (256×256 px) — cell mask in red (α=0.5)\n'
    f'{n_zero_label}/{N_CELLS} cells had label=0 at crop centre',
    fontsize=11
)
plt.tight_layout()
gfp_out = BASE_GFP / 'figures' / 'gfp_mask_overlay_10cells_b.png'
fig_gfp.savefig(str(gfp_out), dpi=150, bbox_inches='tight')
plt.close(fig_gfp)
print(f'\nSaved GFP figure: {gfp_out}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — BF crops
# ═══════════════════════════════════════════════════════════════════════════
print('\n=== BF crops ===', flush=True)

matches = pd.read_csv(MATCHES_CSV)
matches["bf_track_id"]       = pd.to_numeric(matches["bf_track_id"],       errors="coerce")
matches["bf_earliest_frame"] = pd.to_numeric(matches["bf_earliest_frame"], errors="coerce")

cells = matches[
    (matches["match_tier"].isin({'confident', 'plausible'})) &
    (~matches["is_ambiguous"]) &
    matches["bf_track_id"].notna() &
    (matches["bf_earliest_frame"] <= matches["onset_frame"] - LOOKBACK)
].copy()
cells["bf_track_id"]  = cells["bf_track_id"].astype(int)
cells["gfp_track_id"] = cells["gfp_track_id"].astype(int)
cells["target_frame"] = cells["onset_frame"] - LOOKBACK
print(f'  Eligible BF cells: {len(cells)}', flush=True)

# Load BF allspot positions
print('  Loading BF allspot positions...', flush=True)
chunks = []
for chunk in pd.read_csv(
        ALLSPOTS,
        usecols=["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"],
        low_memory=False, chunksize=200_000):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunks.append(chunk.dropna(subset=["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"]))
bf_all = pd.concat(chunks, ignore_index=True)
bf_all["TRACK_ID"] = bf_all["TRACK_ID"].astype(int)
bf_all["FRAME"]    = bf_all["FRAME"].astype(int)
bf_pos = bf_all.set_index(["TRACK_ID", "FRAME"])[["POSITION_X", "POSITION_Y"]].to_dict("index")
print(f'  BF allspot lookup: {len(bf_pos)} entries', flush=True)

rng2 = np.random.default_rng(SEED)
sample_idx = rng2.choice(len(cells), size=min(N_CELLS, len(cells)), replace=False)
selected   = cells.iloc[sample_idx].reset_index(drop=True)

fig_bf, axes_bf = plt.subplots(2, 5, figsize=(22, 9))
axes_bf = axes_bf.flatten()

n_zero_bf = 0
for i, row in selected.iterrows():
    gfp_tid  = int(row["gfp_track_id"])
    bf_tid   = int(row["bf_track_id"])
    target_f = int(row["target_frame"])
    onset_f  = int(row["onset_frame"])

    key = (bf_tid, target_f)
    if key not in bf_pos:
        axes_bf[i].set_title(f'GFP {gfp_tid}\n(no BF pos)', fontsize=7)
        axes_bf[i].axis('off')
        print(f'  [{i+1}] GFP {gfp_tid}: BF position missing at frame {target_f}', flush=True)
        continue

    x_um = bf_pos[key]["POSITION_X"]
    y_um = bf_pos[key]["POSITION_Y"]
    cx   = int(round(x_um / PIXEL_SCALE))
    cy   = int(round(y_um / PIXEL_SCALE))

    img_crop = get_img_crop(bf_img, target_f, cx, cy)

    # Run BF model on this crop to obtain the segmentation mask
    masks_out, _, _ = bf_model.eval(
        [img_crop],
        diameter=40,
        channels=[0, 0],
        flow_threshold=0.8,
        cellprob_threshold=-1,
        min_size=100,
        do_3D=False,
    )
    cell_mask_2d  = masks_out[0]    # 2D int label map for this crop
    center_label  = int(cell_mask_2d[HALF, HALF])

    if center_label != 0:
        binary_mask = (cell_mask_2d == center_label)
    else:
        binary_mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)
        n_zero_bf += 1

    title = (f'GFP {gfp_tid} / BF {bf_tid}\n'
             f'frame {target_f} (onset-{LOOKBACK})  label={center_label}')
    overlay_and_draw(axes_bf[i], img_crop, binary_mask, title)
    print(f'  [{i+1}] gfp={gfp_tid}  bf={bf_tid}  frame={target_f}  '
          f'cx={cx}  cy={cy}  label={center_label}  mask_px={binary_mask.sum()}', flush=True)

fig_bf.suptitle(
    f'BF crops onset−{LOOKBACK} (256×256 px) — cell mask in red (α=0.5)\n'
    f'{n_zero_bf}/{N_CELLS} cells had label=0 at crop centre',
    fontsize=11
)
plt.tight_layout()
bf_out = BASE_BF / 'figures' / 'bf_mask_overlay_10cells_b.png'
fig_bf.savefig(str(bf_out), dpi=150, bbox_inches='tight')
plt.close(fig_bf)
print(f'\nSaved BF figure: {bf_out}', flush=True)

print('\nAll done.', flush=True)

#!/usr/bin/env python3
"""
07_gfp_bf_mask_comparison.py

For 10 randomly selected GFP-positive cells, show crops of the GFP and BF
channels with segmentation mask circles overlaid:
  Green  (alpha=0.5) = GFP segmentation (radius from A2_Merged_spots.csv col R)
  Magenta (alpha=0.5) = BF segmentation  (radius from A2_BrightField_allspots.csv)

Layout: 10 rows × 2 columns
  Left : GFP fluorescence + GFP mask
  Right : BF image + GFP mask (green) + BF mask (magenta)

Cells are sampled from quality GFP tracks (>=30 frames).
Frame used: GFP onset frame (= first frame of each GFP track).
BF mask: nearest BF allspot to the GFP centroid at the same frame.
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
GFP_TIFF = ROOT / 'CellposeEmbedding' / 'A2_GFP_raw.tif'
BF_TIFF  = ROOT / 'CompleteImage'     / 'A2_BrightField_raw.tif'
GFP_SPOTS = ROOT / 'CompleteImage'    / 'A2_Merged_spots.csv'
BF_ALLSPOTS = ROOT / 'CompleteImage'  / 'A2_BrightField_allspots.csv'
OUT_DIR  = Path(__file__).resolve().parent

# ── Parameters ─────────────────────────────────────────────────────────────────
PIXEL_SCALE  = 0.2871   # µm / px
CROP_SIZE    = 300      # px — crop width and height
HALF         = CROP_SIZE // 2
N_CELLS      = 10
MIN_FRAMES   = 30       # minimum track length to sample from
MAX_BF_DIST  = 15.0     # µm — max distance to match a BF spot to a GFP cell
MIN_BF_RADIUS = 5.0     # µm — ignore tiny spurious BF detections
RANDOM_SEED  = 42

# ── Load GFP spots — quality tracks, onset frame + radius ─────────────────────
print('Loading GFP spots …')
gfp = pd.read_csv(GFP_SPOTS, low_memory=False,
                  usecols=['Track ID', 'Frame', 'X', 'Y', 'R'])
gfp = gfp.dropna(subset=['Track ID'])
gfp['Track ID'] = gfp['Track ID'].astype(int)
gfp['Track.ID'] = 'A2_' + gfp['Track ID'].astype(str)
gfp['R'] = pd.to_numeric(gfp['R'], errors='coerce')

# Quality filter: keep tracks with >= MIN_FRAMES detections
track_len = gfp.groupby('Track.ID')['Frame'].count()
quality_ids = set(track_len.index[track_len >= MIN_FRAMES])
gfp = gfp[gfp['Track.ID'].isin(quality_ids)].copy()

# Onset = first frame per track
onset = (gfp.sort_values('Frame')
            .groupby('Track.ID')
            .first()
            .reset_index()[['Track.ID', 'Frame', 'X', 'Y', 'R']]
            .rename(columns={'Frame': 'onset_frame',
                             'X': 'x_um', 'Y': 'y_um', 'R': 'gfp_r_um'}))
onset = onset.dropna(subset=['gfp_r_um'])
onset = onset[onset['gfp_r_um'] > 0]
print(f'  {len(onset)} quality GFP cells with valid radius')

# Sample N_CELLS
rng = np.random.default_rng(RANDOM_SEED)
sample = onset.sample(n=N_CELLS, random_state=RANDOM_SEED).reset_index(drop=True)
print(f'  Sampled {N_CELLS} cells')

# ── Load BF allspots — build per-frame KD-tree ────────────────────────────────
print('Loading BF allspot detections …')
chunks = []
for chunk in pd.read_csv(
        BF_ALLSPOTS, low_memory=False, chunksize=200_000,
        usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y', 'RADIUS']):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk = chunk.dropna(subset=['FRAME', 'POSITION_X', 'POSITION_Y', 'RADIUS'])
    chunk = chunk[chunk['RADIUS'] >= MIN_BF_RADIUS]
    chunks.append(chunk)
bf_all = pd.concat(chunks, ignore_index=True)
bf_all['FRAME'] = bf_all['FRAME'].astype(int)
print(f'  {len(bf_all):,} BF spots with radius ≥ {MIN_BF_RADIUS} µm')

# Build per-frame KD-tree over BF positions (µm)
bf_trees  = {}
bf_coords = {}
bf_radii_by_frame = {}
for f, grp in bf_all.groupby('FRAME'):
    xy = grp[['POSITION_X', 'POSITION_Y']].values
    bf_trees[f]          = cKDTree(xy)
    bf_coords[f]         = xy
    bf_radii_by_frame[f] = grp['RADIUS'].values

# ── Memmap both image stacks ───────────────────────────────────────────────────
print('Memmapping image stacks …')
gfp_img = tifffile.memmap(str(GFP_TIFF))   # (T, H, W)
bf_img  = tifffile.memmap(str(BF_TIFF))    # (T, H, W)
T, H, W = bf_img.shape
print(f'  BF: {bf_img.shape}  GFP: {gfp_img.shape}')

def get_crop(stack, frame, cx_px, cy_px):
    """Return CROP_SIZE × CROP_SIZE crop centred on (cx_px, cy_px)."""
    y0, y1 = cy_px - HALF, cy_px + HALF
    x0, x1 = cx_px - HALF, cx_px + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = stack[frame, iy0:iy1, ix0:ix1].astype(np.float32)
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

def norm(crop):
    """Stretch contrast to [0, 1] for display."""
    lo, hi = np.percentile(crop[crop > 0], [1, 99]) if crop.max() > 0 \
             else (0, 1)
    return np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1)

def circle_mask(center_y, center_x, radius_px, h=CROP_SIZE, w=CROP_SIZE):
    """Filled boolean circle mask."""
    yy, xx = np.ogrid[:h, :w]
    return (yy - center_y)**2 + (xx - center_x)**2 <= radius_px**2

def rgba_overlay(mask, color_rgb, alpha=0.5):
    """Convert boolean mask to RGBA array with given colour and alpha."""
    rgba = np.zeros((CROP_SIZE, CROP_SIZE, 4), dtype=np.float32)
    rgba[mask, :3] = color_rgb
    rgba[mask,  3] = alpha
    return rgba

# ── Build figure ───────────────────────────────────────────────────────────────
print('Building figure …')
fig, axes = plt.subplots(N_CELLS, 2, figsize=(7, 3.5 * N_CELLS))

for i, row in sample.iterrows():
    frame  = int(row['onset_frame'])
    x_um   = float(row['x_um'])
    y_um   = float(row['y_um'])
    gfp_r  = float(row['gfp_r_um'])

    cx_px = int(round(x_um / PIXEL_SCALE))
    cy_px = int(round(y_um / PIXEL_SCALE))
    gfp_r_px = gfp_r / PIXEL_SCALE

    # Crop centre offset within the crop array
    # (the GFP cell is always at the centre of the crop)
    cy_in_crop = HALF
    cx_in_crop = HALF

    # ── Find nearest BF spot ──────────────────────────────────────────────────
    bf_r_px = None
    bf_dy = bf_dx = 0    # offset of BF centre relative to crop centre (px)
    if frame in bf_trees:
        d, idx = bf_trees[frame].query([x_um, y_um], k=1)
        if d <= MAX_BF_DIST:
            bf_x_um, bf_y_um = bf_coords[frame][idx]
            bf_r_um = bf_radii_by_frame[frame][idx]
            bf_r_px = bf_r_um / PIXEL_SCALE
            bf_dx = int(round((bf_x_um - x_um) / PIXEL_SCALE))
            bf_dy = int(round((bf_y_um - y_um) / PIXEL_SCALE))

    # ── Crops ─────────────────────────────────────────────────────────────────
    crop_gfp = norm(get_crop(gfp_img, frame, cx_px, cy_px))
    crop_bf  = norm(get_crop(bf_img,  frame, cx_px, cy_px))

    # ── Masks ─────────────────────────────────────────────────────────────────
    gfp_mask = circle_mask(cy_in_crop, cx_in_crop, gfp_r_px)
    bf_mask  = circle_mask(cy_in_crop + bf_dy, cx_in_crop + bf_dx,
                           bf_r_px) if bf_r_px is not None else \
               np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)

    gfp_overlay = rgba_overlay(gfp_mask, [0, 1, 0], alpha=0.5)   # green
    bf_overlay  = rgba_overlay(bf_mask,  [1, 0, 1], alpha=0.5)   # magenta

    ax_gfp = axes[i, 0]
    ax_bf  = axes[i, 1]

    # ── Left panel: GFP channel + GFP mask ────────────────────────────────────
    ax_gfp.imshow(crop_gfp, cmap='gray', vmin=0, vmax=1)
    ax_gfp.imshow(gfp_overlay)
    ax_gfp.set_title(f'Cell {i+1}  frame {frame}\n'
                     f'GFP r={gfp_r:.1f} µm  ({gfp_r_px:.0f} px)',
                     fontsize=7)
    ax_gfp.axis('off')

    # ── Right panel: BF channel + both masks ──────────────────────────────────
    ax_bf.imshow(crop_bf, cmap='gray', vmin=0, vmax=1)
    ax_bf.imshow(gfp_overlay)
    ax_bf.imshow(bf_overlay)
    bf_label = f'BF r={bf_r_um:.1f} µm  ({bf_r_px:.0f} px)' \
               if bf_r_px is not None else 'BF: no match'
    ax_bf.set_title(f'{bf_label}\n'
                    f'GFP=green  BF=magenta',
                    fontsize=7)
    ax_bf.axis('off')

# Column headers
axes[0, 0].set_title('GFP channel  +  GFP mask (green)\n' +
                      axes[0, 0].get_title(), fontsize=7)
axes[0, 1].set_title('BF channel  +  GFP mask (green)  +  BF mask (magenta)\n' +
                      axes[0, 1].get_title(), fontsize=7)

fig.suptitle('GFP vs BF segmentation size comparison — 10 random cells at GFP onset',
             fontsize=11, y=1.002)
fig.tight_layout()
out = OUT_DIR / 'gfp_bf_mask_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out.name}')
print('Done.')

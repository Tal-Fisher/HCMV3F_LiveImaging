#!/usr/bin/env python3
"""
fig6_cell_crops.py

20 random GFP-positive cells at their GFP onset frame.
Each cell shown as a 256×256 px crop in BF and GFP channels separately.

Layout: 4 rows × 10 columns
  Row 0 : BF  for cells 1-10
  Row 1 : GFP for cells 1-10
  Row 2 : BF  for cells 11-20
  Row 3 : GFP for cells 11-20

Contrast: per-crop percentile stretch [1 %, 99.5 %].
"""

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
ONSET_CSV = ROOT / 'CompleteImage'      / 'A2_gfp_onset.csv'
BF_TIFF   = ROOT / 'CompleteImage'      / 'A2_BrightField_raw.tif'
GFP_TIFF  = ROOT / 'CellposeEmbedding'  / 'A2_GFP_raw.tif'
OUT_DIR   = Path(__file__).resolve().parent

# ── Parameters ─────────────────────────────────────────────────────────────────
PIXEL_SCALE = 0.2871   # µm / px
CROP_SIZE   = 256
HALF        = CROP_SIZE // 2
N_CELLS     = 20
RANDOM_SEED = 42

# ── Sample cells ───────────────────────────────────────────────────────────────
onset = pd.read_csv(ONSET_CSV)
# exclude cells whose onset is so early there may not be enough pre-frames
onset = onset[onset['gfp_onset_frame'] >= 0].copy()
sample = onset.sample(n=N_CELLS, random_state=RANDOM_SEED).reset_index(drop=True)
print(f'Sampled {N_CELLS} cells from {len(onset)} total')

# ── Memmap image stacks ────────────────────────────────────────────────────────
print('Memmapping images …')
bf_img  = tifffile.memmap(str(BF_TIFF))
gfp_img = tifffile.memmap(str(GFP_TIFF))
T, H, W = bf_img.shape

def get_crop(stack, frame, cx_px, cy_px):
    y0, y1 = cy_px - HALF, cy_px + HALF
    x0, x1 = cx_px - HALF, cx_px + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = stack[frame, iy0:iy1, ix0:ix1].astype(np.float32)
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

def stretch(crop):
    """Percentile contrast stretch to [0, 1]."""
    flat = crop.ravel()
    nonzero = flat[flat > 0]
    if len(nonzero) == 0:
        return crop
    lo = np.percentile(nonzero, 1)
    hi = np.percentile(nonzero, 99.5)
    return np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1)

# ── Pre-extract all crops ──────────────────────────────────────────────────────
print('Extracting crops …')
bf_crops  = []
gfp_crops = []

for _, row in sample.iterrows():
    frame = int(row['gfp_onset_frame'])
    cx    = int(round(float(row['x_at_onset']) / PIXEL_SCALE))
    cy    = int(round(float(row['y_at_onset']) / PIXEL_SCALE))
    bf_crops.append(stretch(get_crop(bf_img,  frame, cx, cy)))
    gfp_crops.append(stretch(get_crop(gfp_img, frame, cx, cy)))

# ── Build figure ───────────────────────────────────────────────────────────────
# 4 rows × 10 columns; rows 0-1 = cells 1-10, rows 2-3 = cells 11-20
print('Building figure …')

NCOLS     = 10
GAP_FRAC  = 0.04   # gap between the two blocks of rows (as fraction of row height)

fig = plt.figure(figsize=(NCOLS * 1.4, 4 * 1.5))

# Use gridspec so we can add a small extra vertical gap between blocks
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, NCOLS,
              figure=fig,
              left=0.04, right=0.99,
              top=0.93,  bottom=0.01,
              hspace=0.04,
              wspace=0.02,
              height_ratios=[1, 1, 1, 1])

# Add a visual separator between the two 10-cell blocks by adjusting top/bottom
# via a nested gridspec approach — simplest: adjust hspace manually for pairs
gs2 = GridSpec(4, NCOLS,
               figure=fig,
               left=0.04, right=0.99,
               top=0.93,  bottom=0.01,
               hspace=0.08,
               wspace=0.02)

# clear first gs
fig.clf()

# Recreate with unequal row spacing: pack rows 0-1 tight, 1-2 gap, rows 2-3 tight
gs = GridSpec(4, NCOLS,
              figure=fig,
              left=0.04, right=0.99,
              top=0.93,  bottom=0.01,
              hspace=0.0,
              wspace=0.02)

def show_crop(ax, crop, cmap='gray'):
    ax.imshow(crop, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')

row_channel = ['BF', 'GFP', 'BF', 'GFP']
row_color   = ['#aaaaaa', '#6fbc6f', '#aaaaaa', '#6fbc6f']

for row in range(4):
    block  = 0 if row < 2 else 1          # first or second block of 10
    is_gfp = (row % 2 == 1)
    crops  = gfp_crops if is_gfp else bf_crops
    offset = block * 10                    # cell index offset

    for col in range(NCOLS):
        cell_idx = offset + col
        ax = fig.add_subplot(gs[row, col])
        show_crop(ax, crops[cell_idx])

        # Label first column of each row with channel name
        if col == 0:
            ax.set_ylabel(row_channel[row], fontsize=8,
                          color=row_color[row], rotation=0,
                          labelpad=28, va='center', fontweight='bold')

        # Label top of first row of each block with cell number
        if row in (0, 2):
            ax.set_title(str(cell_idx + 1), fontsize=7, pad=2, color='#555555')

# ── Block separators and column header ────────────────────────────────────────
# Horizontal line between the two blocks (drawn in figure coordinates)
line_y = 0.49   # approximate mid-point between row 1 and row 2
fig.add_artist(plt.Line2D([0.04, 0.99], [line_y, line_y],
                           transform=fig.transFigure,
                           color='#cccccc', linewidth=0.8, linestyle='--'))

fig.suptitle(
    f'20 random GFP-positive cells at GFP onset — BF (top) and GFP (bottom) channels\n'
    f'256 × 256 px crops  ({CROP_SIZE * PIXEL_SCALE:.0f} × {CROP_SIZE * PIXEL_SCALE:.0f} µm)  ·  A2 dataset',
    fontsize=9, y=0.98, color='#222222'
)

# ── Save ───────────────────────────────────────────────────────────────────────
for ext in ('png', 'pdf'):
    out = OUT_DIR / f'fig6_cell_crops.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {out.name}')

plt.close(fig)
print('Done.')

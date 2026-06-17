#!/usr/bin/env python3
"""
bf_crops_frame1_frame55.py

Plot brightfield crops of 10 random productive and 10 random nonproductive
cells at their 1st and 55th tracked BF frame.

Output: bf_crops_frame1_frame55.png in this folder
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIVEIMG   = Path('/home/labs/ginossar/talfis/LiveImaging')
BARRIER   = Path('/home/labs/ginossar/talfis/LiveImaging/barrier_analysis')
BF_EMBED  = LIVEIMG / 'BrightFieldEmbedding'
CACHE     = LIVEIMG / 'cache' / 'python_export'

BF_TIFF   = LIVEIMG / 'CompleteImage' / 'A2_BrightField_raw.tif'
ALLSPOTS  = LIVEIMG / 'CompleteImage' / 'A2_BrightField_allspots.csv'
MATCHES   = BF_EMBED / 'bf_gfp_matches.csv'
MODEL_DF  = CACHE / 'model_df.csv'

PIXEL_SCALE = 0.2871   # µm / px
CROP_SIZE   = 256
HALF        = CROP_SIZE // 2
N_CELLS     = 10
FRAME_A     = 1    # 1st frame of cell track (1-indexed)
FRAME_B     = 55   # 55th frame of cell track
RANDOM_SEED = 42

# ── Load cell classifications ─────────────────────────────────────────────────
print('Loading model_df...', flush=True)
model_df = pd.read_csv(MODEL_DF)
model_df = model_df[model_df['dataset'] == 'A2'].copy()
model_df['gfp_track_id'] = model_df['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
model_df['productive'] = model_df['delay_green_to_red'].apply(
    lambda x: False if str(x).strip() in ('Inf', 'NA', '') else np.isfinite(float(x))
)
print(f'  A2 cells: {len(model_df)}  productive: {model_df["productive"].sum()}  '
      f'nonproductive: {(~model_df["productive"]).sum()}', flush=True)

# ── Load BF↔GFP matches ───────────────────────────────────────────────────────
print('Loading BF↔GFP matches...', flush=True)
matches = pd.read_csv(MATCHES)
matches['bf_track_id'] = pd.to_numeric(matches['bf_track_id'], errors='coerce')
matches = matches[
    matches['match_tier'].isin({'confident', 'plausible'}) &
    ~matches['is_ambiguous'].astype(bool) &
    matches['bf_track_id'].notna()
].copy()
matches['bf_track_id'] = matches['bf_track_id'].astype(int)
matches['gfp_track_id'] = matches['gfp_track_id'].astype(int)

# Merge classification into matches
merged = matches.merge(model_df[['gfp_track_id', 'productive']], on='gfp_track_id', how='inner')
print(f'  Cells with BF match + classification: {len(merged)}  '
      f'productive: {merged["productive"].sum()}  '
      f'nonproductive: {(~merged["productive"]).sum()}', flush=True)

# ── Load BF allspot positions ─────────────────────────────────────────────────
print('Loading BF allspot positions...', flush=True)
chunks = []
for chunk in pd.read_csv(
        ALLSPOTS,
        usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y'],
        low_memory=False, chunksize=200_000):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunks.append(chunk.dropna(subset=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y']))
bf_all = pd.concat(chunks, ignore_index=True)
bf_all['TRACK_ID'] = bf_all['TRACK_ID'].astype(int)
bf_all['FRAME'] = bf_all['FRAME'].astype(int)

# Build dict: bf_track_id → sorted list of (frame, x_um, y_um)
print('Building per-track frame lists...', flush=True)
bf_tracks = {}
for tid, grp in bf_all.sort_values('FRAME').groupby('TRACK_ID'):
    bf_tracks[int(tid)] = list(zip(grp['FRAME'], grp['POSITION_X'], grp['POSITION_Y']))
print(f'  BF tracks loaded: {len(bf_tracks)}', flush=True)

def get_frame_pos(bf_tid, rel_frame_1indexed):
    """Return (abs_frame, cx_px, cy_px) for the nth track frame (1-indexed), or None."""
    frames = bf_tracks.get(bf_tid)
    if frames is None or len(frames) < rel_frame_1indexed:
        return None
    f, x_um, y_um = frames[rel_frame_1indexed - 1]
    cx = int(round(x_um / PIXEL_SCALE))
    cy = int(round(y_um / PIXEL_SCALE))
    return int(f), cx, cy

# ── Select random qualifying cells ────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

def select_cells(group_df, n, label):
    candidates = []
    for _, row in group_df.iterrows():
        bf_tid = int(row['bf_track_id'])
        gfp_tid = int(row['gfp_track_id'])
        pos_a = get_frame_pos(bf_tid, FRAME_A)
        pos_b = get_frame_pos(bf_tid, FRAME_B)
        if pos_a is not None and pos_b is not None:
            candidates.append({'gfp_track_id': gfp_tid, 'bf_track_id': bf_tid,
                                'pos_a': pos_a, 'pos_b': pos_b})
    print(f'  {label}: {len(candidates)} cells have ≥{FRAME_B} tracked BF frames', flush=True)
    n = min(n, len(candidates))
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in sorted(idx)]

prod_cells    = select_cells(merged[merged['productive']],  N_CELLS, 'productive')
nonprod_cells = select_cells(merged[~merged['productive']], N_CELLS, 'nonproductive')

# ── Memmap BF TIFF ────────────────────────────────────────────────────────────
print(f'Memmapping {BF_TIFF.name}...', flush=True)
bf_img = tifffile.memmap(str(BF_TIFF))
T, H, W = bf_img.shape
print(f'  BF shape: {bf_img.shape}, dtype: {bf_img.dtype}', flush=True)

def get_crop(frame, cx, cy):
    """256×256 crop centred on (cx, cy), zero-padded at boundaries."""
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = bf_img[frame, iy0:iy1, ix0:ix1].astype(np.float32)
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

def draw_crop(ax, frame, cx, cy, title=''):
    crop = get_crop(frame, cx, cy)
    if crop.max() > crop.min():
        vmin = np.percentile(crop, 1)
        vmax = np.percentile(crop, 99)
    else:
        vmin, vmax = 0, 1
    ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.axis('off')

# ── Build figure ──────────────────────────────────────────────────────────────
ncols = max(len(prod_cells), len(nonprod_cells))
fig, axes = plt.subplots(4, ncols, figsize=(2.0 * ncols, 9),
                          gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

# Row labels
row_meta = [
    (prod_cells,    0, 'frame 1',  'Productive',     '#2ecc71'),
    (prod_cells,    1, 'frame 55', 'Productive',     '#2ecc71'),
    (nonprod_cells, 2, 'frame 1',  'Non-productive', '#e74c3c'),
    (nonprod_cells, 3, 'frame 55', 'Non-productive', '#e74c3c'),
]

for cells, row_idx, frame_label, group_label, color in row_meta:
    pos_key = 'pos_a' if frame_label == 'frame 1' else 'pos_b'
    frame_num = FRAME_A if frame_label == 'frame 1' else FRAME_B
    for col, cell in enumerate(cells):
        abs_f, cx, cy = cell[pos_key]
        draw_crop(axes[row_idx, col], abs_f, cx, cy)
    # Hide any unused columns
    for col in range(len(cells), ncols):
        axes[row_idx, col].axis('off')
    # Row label on the left using fig.text
    y_pos = axes[row_idx, 0].get_position().y0 + axes[row_idx, 0].get_position().height / 2
    fig.text(0.01, y_pos,
             f'{group_label}\n{frame_label}',
             ha='left', va='center', fontsize=9, color=color, fontweight='bold',
             rotation=90, transform=fig.transFigure)

# Column numbers
for col, cell in enumerate(prod_cells):
    axes[0, col].set_title(f'cell {col+1}\nGFP {cell["gfp_track_id"]}', fontsize=6.5, pad=2)

for col, cell in enumerate(nonprod_cells):
    axes[2, col].set_title(f'cell {col+1}\nGFP {cell["gfp_track_id"]}', fontsize=6.5, pad=2)

# Horizontal divider between productive / nonproductive groups
# (add a thin line between row 1 and row 2)
y_mid = (axes[1, 0].get_position().y0 + axes[2, 0].get_position().y1) / 2
fig.add_artist(plt.Line2D([0.06, 0.99], [y_mid, y_mid],
                           transform=fig.transFigure, color='#aaaaaa', linewidth=0.8))

fig.suptitle(
    f'Brightfield crops — frame {FRAME_A} vs frame {FRAME_B} of cell BF track\n'
    f'10 random productive  |  10 random non-productive  (A2 dataset)',
    fontsize=11, y=0.995
)

out_path = BARRIER / 'bf_crops_frame1_frame55.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out_path}')

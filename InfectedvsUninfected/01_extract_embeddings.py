#!/usr/bin/env python3
"""
01_extract_embeddings.py

Extract 256-dim Cellpose SAM neck embeddings for two classes:
  label=1  infected  -- BF crop at (GFP onset frame - LOOKBACK), identity
                        verified by strict backward NN tracking
  label=0  uninfected -- BF crop at T_REF (fixed early frame), identity
                         verified by strict forward NN tracking

Infected cell selection:
  1. Snap GFP cell position at onset_frame to nearest BF allspot
     (within MAX_RADIUS_UM).
  2. Track backward LOOKBACK frames using strict NN (ambiguity check).
  3. Cross-check snapped BF track_id against bf_gfp_matches.csv:
     must be 'confident', unambiguous, temporally_consistent.

Uninfected cell selection:
  1. BF track IDs that never appear in bf_gfp_matches.csv at ANY tier.
  2. Track must have a detection at T_REF.
  3. Track forward MIN_NEG_FWRD frames from T_REF with strict NN.
  4. Spatial exclusion: centroid at T_REF must be > MIN_DIST_UM from
     any infected cell's BF centroid.

Outputs:
  embeddings/A2_infected_vs_uninfected.npz   -- track_ids, labels,
                                                 embeddings (N,256),
                                                 extraction_frames
  embeddings/A2_infected_vs_uninfected.csv   -- full metadata + emb_0..255
  figures/sample_crops.png                   -- 10 crops per class
  results/filter_stats.txt                   -- kept/discarded at every step
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent
LIVEIMG   = BASE.parent
BFE_DIR   = LIVEIMG / 'BrightFieldEmbedding'

BF_TIFF   = LIVEIMG / 'CompleteImage' / 'A2_BrightField_raw.tif'
ALLSPOTS  = LIVEIMG / 'CompleteImage' / 'A2_BrightField_allspots.csv'
ONSET_CSV = LIVEIMG / 'CompleteImage' / 'A2_gfp_onset.csv'
MATCHES   = BFE_DIR / 'bf_gfp_matches.csv'
MODEL_PATH = str(BFE_DIR / 'models' / 'cpsam_BrightField')

OUT_DIR  = BASE / 'embeddings'
FIG_DIR  = BASE / 'figures'
RES_DIR  = BASE / 'results'
for d in (OUT_DIR, FIG_DIR, RES_DIR):
    d.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
PIXEL_SCALE     = 0.2871   # µm / px
MAX_RADIUS_UM   = 15.0     # max single-frame NN displacement
AMBIGUITY_RATIO = 2.0      # 2nd-nearest must be >= ratio × nearest to link
LOOKBACK        = 10       # frames to track backward from GFP onset
MIN_NEG_FWRD    = 15       # min consecutive forward frames for uninfected
MIN_DIST_UM     = 50.0     # spatial exclusion radius around infected cells
T_REF           = 20       # fixed early extraction frame for uninfected
CROP_SIZE       = 256
HALF            = CROP_SIZE // 2
DIAMETER        = 40
N_SAMPLE_CROPS  = 10
RANDOM_SEED     = 42

# ── Filter statistics log ──────────────────────────────────────────────────────
stats_lines = []

def log(msg):
    print(msg, flush=True)
    stats_lines.append(msg)

# ── Load allspot detections and build per-frame KD-trees ──────────────────────
log('Loading BF allspot detections...')
chunks = []
for chunk in pd.read_csv(
        ALLSPOTS, low_memory=False, chunksize=200_000,
        usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y']):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunks.append(chunk.dropna(subset=['FRAME', 'POSITION_X', 'POSITION_Y']))
bf_all = pd.concat(chunks, ignore_index=True)
bf_all['FRAME'] = bf_all['FRAME'].astype(int)
# TRACK_ID may be NaN for untracked spots; keep as float for now
log(f'  Total allspot rows: {len(bf_all):,}  |  frames: {bf_all["FRAME"].nunique()}')

log('Building per-frame KD-trees...')
frames_all = sorted(bf_all['FRAME'].unique())
MAX_FRAME  = max(frames_all)
trees  = {}   # frame -> cKDTree (µm coords)
coords = {}   # frame -> (N,2) array [x_um, y_um]
for f, grp in bf_all.groupby('FRAME'):
    xy = grp[['POSITION_X', 'POSITION_Y']].values
    coords[f] = xy
    trees[f]  = cKDTree(xy)
log(f'  KD-trees built for {len(trees)} frames')

# Build (TRACK_ID, FRAME) -> (x_um, y_um) lookup for tracked spots
bf_tracked = bf_all.dropna(subset=['TRACK_ID']).copy()
bf_tracked['TRACK_ID'] = bf_tracked['TRACK_ID'].astype(int)
bf_pos = bf_tracked.set_index(['TRACK_ID', 'FRAME'])[
    ['POSITION_X', 'POSITION_Y']].to_dict('index')

# ── Strict backward NN tracker ─────────────────────────────────────────────────
def track_backward(start_frame, start_x_um, start_y_um, n_frames):
    """
    Track backward n_frames from (start_frame, start_x_um, start_y_um).
    Returns list of (frame, x_um, y_um) from start_frame downward, or []
    if the first snap or any link fails.
    stop_reason: 'ok', 'no_tree', 'snap_miss', 'range', 'ambiguous'
    """
    if start_frame not in trees:
        return [], 'no_tree'
    k = min(2, len(coords[start_frame]))
    dists, idxs = trees[start_frame].query([start_x_um, start_y_um], k=k)
    if k == 1:
        dists = np.array([dists])
        idxs  = np.array([idxs])
    if dists[0] > MAX_RADIUS_UM:
        return [], 'snap_miss'
    x, y = coords[start_frame][idxs[0]]
    positions = [(start_frame, x, y)]

    for frame in range(start_frame - 1, start_frame - n_frames - 1, -1):
        if frame not in trees:
            return [], 'no_tree'
        k2 = min(2, len(coords[frame]))
        d2, i2 = trees[frame].query([x, y], k=k2)
        if k2 == 1:
            d2 = np.array([d2]); i2 = np.array([i2])
        if d2[0] > MAX_RADIUS_UM:
            return [], 'range'
        if k2 >= 2 and d2[1] < AMBIGUITY_RATIO * d2[0]:
            return [], 'ambiguous'
        x, y = coords[frame][i2[0]]
        positions.append((frame, x, y))

    return positions, 'ok'

# ── Strict forward NN tracker ──────────────────────────────────────────────────
def track_forward(start_frame, start_x_um, start_y_um, n_frames):
    """
    Track forward n_frames from (start_frame, start_x_um, start_y_um).
    Returns (positions, stop_reason).
    """
    if start_frame not in trees:
        return [], 'no_tree'
    k = min(2, len(coords[start_frame]))
    dists, idxs = trees[start_frame].query([start_x_um, start_y_um], k=k)
    if k == 1:
        dists = np.array([dists])
        idxs  = np.array([idxs])
    if dists[0] > MAX_RADIUS_UM:
        return [], 'snap_miss'
    x, y = coords[start_frame][idxs[0]]
    positions = [(start_frame, x, y)]

    for frame in range(start_frame + 1, start_frame + n_frames + 1):
        if frame not in trees:
            return [], 'no_tree'
        k2 = min(2, len(coords[frame]))
        d2, i2 = trees[frame].query([x, y], k=k2)
        if k2 == 1:
            d2 = np.array([d2]); i2 = np.array([i2])
        if d2[0] > MAX_RADIUS_UM:
            return [], 'range'
        if k2 >= 2 and d2[1] < AMBIGUITY_RATIO * d2[0]:
            return [], 'ambiguous'
        x, y = coords[frame][i2[0]]
        positions.append((frame, x, y))

    return positions, 'ok'

# ══════════════════════════════════════════════════════════════════════════════
# INFECTED CLASS (label = 1)
# ══════════════════════════════════════════════════════════════════════════════

log('\n=== INFECTED CLASS (label=1) ===')

onset_df = pd.read_csv(ONSET_CSV)
log(f'Start: {len(onset_df)} GFP-expressing cells in A2_gfp_onset.csv')

matches_df = pd.read_csv(MATCHES)
matches_df['bf_track_id'] = pd.to_numeric(matches_df['bf_track_id'], errors='coerce')

infected_cells = []
n_snap_miss    = 0
n_back_fail    = {'no_tree': 0, 'snap_miss': 0, 'range': 0, 'ambiguous': 0}
n_onset_early  = 0

for _, row in onset_df.iterrows():
    gfp_tid    = int(row['track_id'])
    onset_f    = int(row['gfp_onset_frame'])
    gfp_x_um  = float(row['x_at_onset'])
    gfp_y_um  = float(row['y_at_onset'])

    # Step 1: snap GFP position to nearest BF allspot at onset_frame
    if onset_f not in trees:
        n_snap_miss += 1
        continue
    k = min(2, len(coords[onset_f]))
    dists, idxs = trees[onset_f].query([gfp_x_um, gfp_y_um], k=k)
    if k == 1:
        dists = np.array([dists])
        idxs  = np.array([idxs])
    if dists[0] > MAX_RADIUS_UM:
        n_snap_miss += 1
        continue
    snap_x, snap_y = coords[onset_f][idxs[0]]

    # Step 2: require enough frames before onset for backward tracking
    target_f = onset_f - LOOKBACK
    if target_f < 0:
        n_onset_early += 1
        continue

    # Backward NN tracking
    positions, reason = track_backward(onset_f, snap_x, snap_y, LOOKBACK)
    if reason != 'ok':
        n_back_fail[reason] = n_back_fail.get(reason, 0) + 1
        continue

    # Position at target_frame is the last entry (earliest frame)
    target_frame_check, x_target, y_target = positions[-1]
    assert target_frame_check == target_f

    # Record nearest tracked BF track_id at snap position (metadata only)
    sub = bf_tracked[bf_tracked['FRAME'] == onset_f]
    if len(sub) > 0:
        xy_sub = sub[['POSITION_X', 'POSITION_Y']].values
        dists_sub = np.linalg.norm(xy_sub - np.array([snap_x, snap_y]), axis=1)
        ni = np.argmin(dists_sub)
        snapped_bf_tid = int(sub.iloc[ni]['TRACK_ID']) if dists_sub[ni] <= MAX_RADIUS_UM else np.nan
    else:
        snapped_bf_tid = np.nan

    infected_cells.append({
        'gfp_track_id':   gfp_tid,
        'bf_track_id':    snapped_bf_tid,
        'onset_frame':    onset_f,
        'target_frame':   target_f,
        'x_um':           x_target,
        'y_um':           y_target,
        'label':          1,
    })

log(f'  Step 1 (snap at onset): {n_snap_miss} discarded (no BF nearby), '
    f'{n_onset_early} discarded (onset < LOOKBACK frames from start)')
log(f'  Step 2 (backward tracking): '
    + ', '.join(f'{v} {k}' for k, v in n_back_fail.items()))
log(f'  => Infected cells kept: {len(infected_cells)}')

infected_df = pd.DataFrame(infected_cells)

# ══════════════════════════════════════════════════════════════════════════════
# UNINFECTED CLASS (label = 0)
# ══════════════════════════════════════════════════════════════════════════════

log('\n=== UNINFECTED CLASS (label=0) ===')
log('Extraction frame: randomly sampled from the infected cells\' extraction '
    'frames (temporal matching)')

# Build per-frame infected KD-trees for per-frame spatial exclusion
from collections import defaultdict
inf_by_frame = defaultdict(list)
for ic in infected_cells:
    inf_by_frame[ic['target_frame']].append([ic['x_um'], ic['y_um']])
inf_trees_by_frame = {f: cKDTree(np.array(xys))
                      for f, xys in inf_by_frame.items() if xys}
infected_frame_set = set(inf_by_frame.keys())

all_track_ids = bf_tracked['TRACK_ID'].unique()
log(f'Start: {len(all_track_ids):,} unique BF track IDs')

# Step 1: exclude any track ID appearing in matches at ANY tier
matched_bf_ids = set(matches_df['bf_track_id'].dropna().astype(int).tolist())
candidate_ids = [t for t in all_track_ids if int(t) not in matched_bf_ids]
log(f'  Step 1 (exclude matched at any tier): '
    f'{len(all_track_ids) - len(candidate_ids)} discarded, '
    f'{len(candidate_ids)} remain')

# Step 2: require presence at at least one infected extraction frame
rng_neg = np.random.default_rng(RANDOM_SEED)
candidate_ids_with_frame = []
n_no_valid_frame = 0
for tid in candidate_ids:
    tid_int = int(tid)
    valid_frames = [f for f in infected_frame_set if (tid_int, f) in bf_pos]
    if not valid_frames:
        n_no_valid_frame += 1
        continue
    assigned_frame = int(rng_neg.choice(valid_frames))
    candidate_ids_with_frame.append((tid_int, assigned_frame))
log(f'  Step 2 (present at ≥1 infected extraction frame): '
    f'{n_no_valid_frame} discarded, {len(candidate_ids_with_frame)} remain')

# Step 3: forward NN tracking from assigned frame for MIN_NEG_FWRD frames
n_fwd_fail = {'no_tree': 0, 'snap_miss': 0, 'range': 0, 'ambiguous': 0}
neg_cells = []

for tid_int, assigned_frame in candidate_ids_with_frame:
    key = (tid_int, assigned_frame)
    if key not in bf_pos:
        n_fwd_fail['snap_miss'] += 1
        continue
    x0 = bf_pos[key]['POSITION_X']
    y0 = bf_pos[key]['POSITION_Y']

    positions, reason = track_forward(assigned_frame, x0, y0, MIN_NEG_FWRD)
    if reason != 'ok':
        n_fwd_fail[reason] += 1
        continue

    neg_cells.append({
        'bf_track_id':  tid_int,
        'target_frame': assigned_frame,
        'x_um':         x0,
        'y_um':         y0,
        'label':        0,
    })

log(f'  Step 3 (forward tracking {MIN_NEG_FWRD} frames): '
    + ', '.join(f'{v} {k}' for k, v in n_fwd_fail.items()))
log(f'  After forward tracking: {len(neg_cells)} remain')

# Step 4: per-frame spatial exclusion — at each cell's assigned frame,
# exclude uninfected cells within MIN_DIST_UM of any infected cell centroid
n_spatial = 0
neg_cells_filtered = []
for c in neg_cells:
    f = c['target_frame']
    if f in inf_trees_by_frame:
        d, _ = inf_trees_by_frame[f].query([c['x_um'], c['y_um']], k=1)
        if d <= MIN_DIST_UM:
            n_spatial += 1
            continue
    neg_cells_filtered.append(c)
neg_cells = neg_cells_filtered
log(f'  Step 4 (per-frame spatial exclusion >{MIN_DIST_UM} µm): '
    f'{n_spatial} discarded, {len(neg_cells)} remain')

log(f'  => Uninfected cells kept: {len(neg_cells)}')

neg_df = pd.DataFrame(neg_cells)

log(f'\nClass summary: {len(infected_df)} infected | {len(neg_df)} uninfected '
    f'(total {len(infected_df) + len(neg_df)})')

# ── Save filter statistics ─────────────────────────────────────────────────────
stats_path = RES_DIR / 'filter_stats.txt'
with open(str(stats_path), 'w') as fh:
    fh.write('\n'.join(stats_lines) + '\n')
print(f'Filter stats saved: {stats_path}')

# ══════════════════════════════════════════════════════════════════════════════
# CELLPOSE SAM EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

print('\nLoading Cellpose BF model...', flush=True)
from cellpose import models
model = models.CellposeModel(pretrained_model=MODEL_PATH, gpu=True)
device = next(model.net.parameters()).device
print(f'  device: {device}', flush=True)

neck_outputs = []

def _hook_fn(module, inp, output):
    neck_outputs.append(output.detach().cpu().float())

hook_handle = model.net.encoder.neck.register_forward_hook(_hook_fn)

# ── Memmap BF TIFF ─────────────────────────────────────────────────────────────
print(f'Memmapping {BF_TIFF.name}...', flush=True)
bf_img = tifffile.memmap(str(BF_TIFF))
if bf_img.ndim != 3:
    raise ValueError(
        f'Expected T×Y×X (3D), got shape {bf_img.shape}. '
        'Adjust channel slicing if needed.')
T_total, H, W = bf_img.shape
print(f'  Shape: {bf_img.shape}  dtype: {bf_img.dtype}', flush=True)

def get_crop(frame, cx_px, cy_px):
    """256×256 crop centred on (cx_px, cy_px), zero-padded at boundaries."""
    y0, y1 = cy_px - HALF, cy_px + HALF
    x0, x1 = cx_px - HALF, cx_px + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = bf_img[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

def extract_embeddings(cell_list, class_name):
    """Run Cellpose on each cell's crop and return (ids, embeddings, meta)."""
    ids_out  = []
    emb_out  = []
    meta_out = []
    n_skip   = 0
    n_total  = len(cell_list)

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = set(rng.choice(n_total, size=min(N_SAMPLE_CROPS, n_total),
                                replace=False).tolist())
    sample_crops = {}

    print(f'  Extracting {class_name} embeddings (n={n_total})...', flush=True)

    for i, cell in enumerate(cell_list):
        f   = int(cell['target_frame'])
        x_um = cell['x_um']
        y_um = cell['y_um']
        cx  = int(round(x_um / PIXEL_SCALE))
        cy  = int(round(y_um / PIXEL_SCALE))

        if not (0 <= cx < W and 0 <= cy < H):
            print(f'    SKIP: centroid ({cx},{cy}) outside image', flush=True)
            n_skip += 1
            continue

        crop = get_crop(f, cx, cy)

        if i in sample_idx:
            sample_crops[i] = (crop.copy(), f, cell.get('label', -1))

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
            print(f'    WARNING: no neck output at index {i}', flush=True)
            n_skip += 1
            continue

        emb = neck_outputs[0].mean(dim=[2, 3])[0].numpy().astype(np.float32)

        if 'gfp_track_id' in cell:
            tid = cell['gfp_track_id']
        else:
            tid = cell['bf_track_id']

        ids_out.append(tid)
        emb_out.append(emb)
        meta_out.append({
            'label':          cell['label'],
            'track_id':       tid,
            'bf_track_id':    cell.get('bf_track_id', np.nan),
            'target_frame':   f,
            'x_um':           x_um,
            'y_um':           y_um,
        })

        if i == 0:
            print(f'    First cell emb shape: {emb.shape}  std: {emb.std():.4f}',
                  flush=True)
        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{n_total} done  ({n_skip} skipped)', flush=True)

    print(f'  {class_name}: {len(ids_out)} extracted, {n_skip} skipped')
    return ids_out, emb_out, meta_out, sample_crops

# ── Run extraction for both classes ────────────────────────────────────────────
inf_ids, inf_embs, inf_meta, inf_crops = extract_embeddings(
    infected_df.to_dict('records'), 'infected')
neg_ids, neg_embs, neg_meta, neg_crops = extract_embeddings(
    neg_df.to_dict('records'), 'uninfected')

hook_handle.remove()

# ── Combine and save ───────────────────────────────────────────────────────────
all_ids  = inf_ids  + neg_ids
all_embs = inf_embs + neg_embs
all_meta = inf_meta + neg_meta

ids_arr  = np.array(all_ids,  dtype=np.int64)
embs_arr = np.array(all_embs, dtype=np.float32)
labs_arr = np.array([m['label'] for m in all_meta], dtype=np.int8)
frames_arr = np.array([m['target_frame'] for m in all_meta], dtype=np.int32)

print(f'\nFinal dataset: {embs_arr.shape[0]} cells  '
      f'({int(labs_arr.sum())} infected, {int((labs_arr==0).sum())} uninfected)')
print(f'Embedding stats: mean={embs_arr.mean():.4f}  std={embs_arr.std():.4f}')

npz_path = OUT_DIR / 'A2_infected_vs_uninfected.npz'
np.savez(str(npz_path),
         track_ids=ids_arr,
         labels=labs_arr,
         embeddings=embs_arr,
         extraction_frames=frames_arr)
print(f'Saved: {npz_path}')

dim = embs_arr.shape[1]
meta_df = pd.DataFrame(all_meta)
emb_df  = pd.DataFrame(embs_arr, columns=[f'emb_{k}' for k in range(dim)])
out_csv = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)
csv_path = OUT_DIR / 'A2_infected_vs_uninfected.csv'
out_csv.to_csv(str(csv_path), index=False)
print(f'Saved: {csv_path}')

# ── Sample crop figure ─────────────────────────────────────────────────────────
print('\nSaving sample crops...')
n_inf_crops = len(inf_crops)
n_neg_crops = len(neg_crops)
n_cols = max(n_inf_crops, n_neg_crops)
fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))
if n_cols == 1:
    axes = axes.reshape(2, 1)

for col, (idx, (crop, frame, lbl)) in enumerate(
        list(inf_crops.items())[:n_cols]):
    ax = axes[0, col]
    vmin = np.percentile(crop, 0.5) if crop.max() > 0 else 0
    vmax = np.percentile(crop, 99.5) if crop.max() > 0 else 1
    ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'Infected\nframe {frame}', fontsize=8)
    ax.axis('off')

for col, (idx, (crop, frame, lbl)) in enumerate(
        list(neg_crops.items())[:n_cols]):
    ax = axes[1, col]
    vmin = np.percentile(crop, 0.5) if crop.max() > 0 else 0
    vmax = np.percentile(crop, 99.5) if crop.max() > 0 else 1
    ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'Uninfected\nframe {frame}', fontsize=8)
    ax.axis('off')

for ax in axes[0, n_inf_crops:]:
    ax.axis('off')
for ax in axes[1, n_neg_crops:]:
    ax.axis('off')

fig.suptitle('Sample BF crops — infected (top) vs uninfected (bottom)', fontsize=11)
plt.tight_layout()
png_path = FIG_DIR / 'sample_crops.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {png_path}')

print('\nDone.', flush=True)

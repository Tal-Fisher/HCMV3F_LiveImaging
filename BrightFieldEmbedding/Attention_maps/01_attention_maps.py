#!/usr/bin/env python3
"""
01_attention_maps.py

Two attention-map approaches for the BF Cellpose embedding model,
applied to 10 cells (5 early / 5 med+late by b2r cutoff):

  Approach 1 — Gradient saliency
    Forward pass with input.requires_grad=True.
    Score = weighted sum of top-20 b2r dims (weights = abs classification beta).
    Saliency = |gradient of score w.r.t. input| smoothed with a Gaussian.

  Approach 2 — Occlusion map
    Mask each 16×16 patch (stride 8) in turn.
    Score drop = baseline_score - occluded_score.
    Higher values = that region was more important.

Output
------
  figures/attention_maps_10cells.png   (10 rows × 3 cols)
  figures/attention_maps_10cells_hires.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')

BF_TIFF      = LIVEIMG / 'CompleteImage' / 'A2_BrightField_raw.tif'
MODEL_PATH   = str(BASE / 'models' / 'cpsam_BrightField')
ALLSPOTS     = LIVEIMG / 'CompleteImage' / 'A2_BrightField_allspots.csv'
MATCHES_CSV  = BASE / 'bf_gfp_matches.csv'
EMB_NPZ      = BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
EXT_DF_CSV   = LIVEIMG / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'
TOP20_CSV    = BASE / 'results' / 'b2r_top20_dims.csv'

OUT_DIR = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

PIXEL_SCALE = 0.2871   # µm / px
CROP_SIZE   = 256
HALF        = CROP_SIZE // 2
LOOKBACK    = 10
CUT_B2R     = 1094     # minutes — early/med+late cutoff
N_PER_CLASS = 5        # early + med+late
SEED        = 42

# Occlusion params
OCC_PATCH   = 16
OCC_STRIDE  = 8

# ── Load top-20 dims and weights ─────────────────────────────────────────────────
top20_df  = pd.read_csv(TOP20_CSV)
TOP20_DIMS   = top20_df['dim'].tolist()
TOP20_BETAS  = top20_df['abs_beta_cls'].values.astype(np.float32)
TOP20_BETAS  = TOP20_BETAS / TOP20_BETAS.sum()   # normalise to sum=1
WEIGHT_VEC   = torch.zeros(256)
for d, w in zip(TOP20_DIMS, TOP20_BETAS):
    WEIGHT_VEC[d] = float(w)
print('Top-20 b2r dims:', TOP20_DIMS)
print('Weight sum:', WEIGHT_VEC.sum().item())

# ── Select 10 cells ──────────────────────────────────────────────────────────────
print('\nSelecting cells...', flush=True)
emb_data      = np.load(str(EMB_NPZ))
emb_track_ids = set(emb_data['gfp_track_ids'].tolist())

ext = pd.read_csv(EXT_DF_CSV)
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id']        = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['delay_blue_to_red'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']

labeled = ext[
    ext['track_id'].isin(emb_track_ids) &
    np.isfinite(ext['delay_blue_to_red'])
].sort_values('track_id').reset_index(drop=True)

early   = labeled[labeled['delay_blue_to_red'] <= CUT_B2R].reset_index(drop=True)
medlate = labeled[labeled['delay_blue_to_red'] >  CUT_B2R].reset_index(drop=True)

rng = np.random.default_rng(SEED)
sel_early   = early.iloc[rng.choice(len(early),   N_PER_CLASS, replace=False)].copy()
sel_medlate = medlate.iloc[rng.choice(len(medlate), N_PER_CLASS, replace=False)].copy()
sel_cells   = pd.concat([sel_early, sel_medlate], ignore_index=True)
sel_cells['class_label'] = (['early'] * N_PER_CLASS + ['med+late'] * N_PER_CLASS)

print(f'  Selected {len(sel_cells)} cells: {N_PER_CLASS} early, {N_PER_CLASS} med+late')
for _, row in sel_cells.iterrows():
    print(f'    track {int(row.track_id):6d}  b2r={row.delay_blue_to_red:.0f} min  [{row.class_label}]')

# ── Build BF position lookup ─────────────────────────────────────────────────────
print('\nBuilding BF position lookup...', flush=True)
matches = pd.read_csv(MATCHES_CSV)
matches['bf_track_id']      = pd.to_numeric(matches['bf_track_id'], errors='coerce')
matches['bf_earliest_frame'] = pd.to_numeric(matches['bf_earliest_frame'], errors='coerce')
MATCH_TIERS = {'confident', 'plausible'}

match_lookup = {}
for _, row in matches.iterrows():
    if (row.get('match_tier') in MATCH_TIERS
            and not row.get('is_ambiguous', True)
            and pd.notna(row.get('bf_track_id'))):
        match_lookup[int(row['gfp_track_id'])] = {
            'bf_track_id':      int(row['bf_track_id']),
            'onset_frame':      int(row['onset_frame']),
            'target_frame':     int(row['onset_frame']) - LOOKBACK,
        }

chunks = []
for chunk in pd.read_csv(
        ALLSPOTS, usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y'],
        low_memory=False, chunksize=200_000):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunks.append(chunk.dropna())
bf_all = pd.concat(chunks, ignore_index=True)
bf_all['TRACK_ID'] = bf_all['TRACK_ID'].astype(int)
bf_all['FRAME']    = bf_all['FRAME'].astype(int)
bf_pos = bf_all.set_index(['TRACK_ID', 'FRAME'])[['POSITION_X', 'POSITION_Y']].to_dict('index')
print(f'  BF positions loaded: {len(bf_pos)} entries')

# ── Memmap BF TIFF ───────────────────────────────────────────────────────────────
print(f'Memmapping {BF_TIFF.name}...', flush=True)
bf_img = tifffile.memmap(str(BF_TIFF))
T, H, W = bf_img.shape
print(f'  Shape: {bf_img.shape}')

def get_crop(frame, cx, cy):
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = bf_img[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

# ── Load model ───────────────────────────────────────────────────────────────────
print('\nLoading Cellpose BF model...', flush=True)
from cellpose import models, transforms
model = models.CellposeModel(pretrained_model=MODEL_PATH, gpu=True)
device = next(model.net.parameters()).device
print(f'  Device: {device}')

# Cast to float32 for gradient computation
model.net.float()
model.net.eval()
for p in model.net.parameters():
    p.requires_grad_(False)   # only the input will carry gradients

# ── Hook: capture neck output before avg-pool ────────────────────────────────────
neck_out_storage = []

def _neck_hook(module, inp, output):
    neck_out_storage.append(output)   # (B, 256, 32, 32)

hook_h = model.net.encoder.neck.register_forward_hook(_neck_hook)

# ── Preprocessing helper ─────────────────────────────────────────────────────────
def crop_to_tensor(crop_uint8, device, requires_grad=False):
    """Convert 256×256 uint8 crop to (1,3,256,256) float32 tensor."""
    img = transforms.normalize99(crop_uint8.astype(np.float32))
    img = np.stack([img] * 3, axis=0)[None]     # (1, 3, H, W)
    # .detach() ensures leaf tensor so .grad is populated after backward()
    t   = torch.from_numpy(img.copy()).float().to(device).detach()
    t.requires_grad_(requires_grad)
    return t

def get_embedding_and_neck(tensor):
    """Forward pass; returns (embedding 256-dim, neck_map 256×32×32)."""
    neck_out_storage.clear()
    _ = model.net(tensor)
    neck = neck_out_storage[0]                  # (1, 256, 32, 32)
    emb  = neck.mean(dim=[2, 3])[0]            # (256,)
    return emb, neck[0]

def score_fn(emb):
    """Weighted sum of top-20 classification dims."""
    return (WEIGHT_VEC.to(emb.device) * emb).sum()

# ═══════════════════════════════════════════════════════════════════════════════════
# APPROACH 1: GRADIENT SALIENCY
# ═══════════════════════════════════════════════════════════════════════════════════
def gradient_saliency(crop_uint8):
    """
    Return a 256×256 saliency map: |grad of score w.r.t. input|
    smoothed with sigma=4 Gaussian.
    """
    t = crop_to_tensor(crop_uint8, device, requires_grad=True)
    emb, _ = get_embedding_and_neck(t)
    s = score_fn(emb)
    s.backward()
    grad = t.grad.squeeze().cpu().numpy()       # (3, 256, 256) → use channel mean
    saliency = np.abs(grad).mean(axis=0)        # (256, 256)
    saliency = gaussian_filter(saliency, sigma=4)
    return saliency

# ═══════════════════════════════════════════════════════════════════════════════════
# APPROACH 2: OCCLUSION MAP
# ═══════════════════════════════════════════════════════════════════════════════════
def occlusion_map(crop_uint8, patch=OCC_PATCH, stride=OCC_STRIDE):
    """
    Batch-occlude all (patch × patch) tiles (stride=stride).
    Return a 256×256 map of score-drop values.
    """
    H = W = CROP_SIZE
    positions = [(r, c)
                 for r in range(0, H - patch + 1, stride)
                 for c in range(0, W - patch + 1, stride)]

    # Baseline score
    with torch.no_grad():
        t0 = crop_to_tensor(crop_uint8, device, requires_grad=False)
        emb0, _ = get_embedding_and_neck(t0)
        baseline = score_fn(emb0).item()

    # Build all occluded crops as one big float32 numpy array
    img_norm = transforms.normalize99(crop_uint8.astype(np.float32))
    base3    = np.stack([img_norm] * 3, axis=0)  # (3, H, W)

    batch_size = 64
    drop_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for start in range(0, len(positions), batch_size):
        batch_pos  = positions[start : start + batch_size]
        batch_imgs = np.stack([base3.copy() for _ in batch_pos], axis=0).astype(np.float32)

        for k, (r, c) in enumerate(batch_pos):
            batch_imgs[k, :, r:r+patch, c:c+patch] = 0.0

        with torch.no_grad():
            tb = torch.from_numpy(batch_imgs).to(device)
            neck_out_storage.clear()
            _ = model.net(tb)
            neck_batch = neck_out_storage[0]        # (B, 256, 32, 32)
            emb_batch  = neck_batch.mean(dim=[2, 3])  # (B, 256)
            scores = (WEIGHT_VEC.to(device) * emb_batch).sum(dim=1).cpu().numpy()

        for k, (r, c) in enumerate(batch_pos):
            drop = baseline - float(scores[k])
            drop_map[r:r+patch, c:c+patch] += drop
            count_map[r:r+patch, c:c+patch] += 1.0

    count_map = np.maximum(count_map, 1)
    drop_map /= count_map
    drop_map = gaussian_filter(drop_map, sigma=2)
    return drop_map

# ═══════════════════════════════════════════════════════════════════════════════════
# COMPUTE MAPS FOR ALL 10 CELLS
# ═══════════════════════════════════════════════════════════════════════════════════
crops_list    = []
saliency_list = []
occlusion_list = []
meta_list      = []

print('\nComputing attention maps...', flush=True)

for i, row in sel_cells.iterrows():
    gfp_tid = int(row['track_id'])
    b2r_val = float(row['delay_blue_to_red'])
    cls_lbl = str(row['class_label'])

    if gfp_tid not in match_lookup:
        print(f'  [WARN] GFP track {gfp_tid} not in match lookup — skipping')
        continue

    m          = match_lookup[gfp_tid]
    bf_tid     = m['bf_track_id']
    target_f   = m['target_frame']

    key = (bf_tid, target_f)
    if key not in bf_pos:
        print(f'  [WARN] BF track {bf_tid} frame {target_f} not in allspots — skipping')
        continue

    x_um = bf_pos[key]['POSITION_X']
    y_um = bf_pos[key]['POSITION_Y']
    cx   = int(round(x_um / PIXEL_SCALE))
    cy   = int(round(y_um / PIXEL_SCALE))
    crop = get_crop(target_f, cx, cy)

    print(f'  Cell {i+1}/10  GFP track {gfp_tid}  b2r={b2r_val:.0f} min  [{cls_lbl}]', flush=True)

    sal  = gradient_saliency(crop)
    occ  = occlusion_map(crop)

    crops_list.append(crop)
    saliency_list.append(sal)
    occlusion_list.append(occ)
    meta_list.append(dict(track_id=gfp_tid, b2r=b2r_val, label=cls_lbl))

hook_h.remove()
n_cells = len(crops_list)
print(f'\nComputed maps for {n_cells} cells.', flush=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# FIGURE: 10 rows × 3 cols
# ═══════════════════════════════════════════════════════════════════════════════════
print('\nGenerating figure...', flush=True)

CMAP_SAL = 'inferno'
CMAP_OCC = 'RdBu_r'

fig, axes = plt.subplots(n_cells, 3, figsize=(12, 3.5 * n_cells),
                          constrained_layout=True)
if n_cells == 1:
    axes = axes[None]

col_titles = ['BF crop  (t = onset − 10)', 'Gradient saliency', 'Occlusion map']
for j, ct in enumerate(col_titles):
    axes[0, j].set_title(ct, fontsize=11, fontweight='bold', pad=8)

for row_i in range(n_cells):
    crop = crops_list[row_i]
    sal  = saliency_list[row_i]
    occ  = occlusion_list[row_i]
    meta = meta_list[row_i]

    b2r_disp  = f'{meta["b2r"]:.0f} min'
    row_label = (f'GFP track {meta["track_id"]}\n'
                 f'b2r = {b2r_disp}  [{meta["label"]}]')

    vmin_bf = np.percentile(crop, 0.5)
    vmax_bf = np.percentile(crop, 99.5)

    # col 0 — BF crop
    ax0 = axes[row_i, 0]
    ax0.imshow(crop, cmap='gray', vmin=vmin_bf, vmax=vmax_bf,
               interpolation='nearest')
    ax0.set_ylabel(row_label, fontsize=8, labelpad=4)
    ax0.set_xticks([]); ax0.set_yticks([])

    # col 1 — gradient saliency overlay
    ax1 = axes[row_i, 1]
    ax1.imshow(crop, cmap='gray', vmin=vmin_bf, vmax=vmax_bf,
               interpolation='nearest')
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
    ax1.imshow(sal_n, cmap=CMAP_SAL, alpha=0.55, interpolation='bilinear',
               vmin=0, vmax=1)
    ax1.set_xticks([]); ax1.set_yticks([])

    # col 2 — occlusion overlay
    ax2 = axes[row_i, 2]
    ax2.imshow(crop, cmap='gray', vmin=vmin_bf, vmax=vmax_bf,
               interpolation='nearest')
    # symmetric colormap around zero
    occ_abs = np.abs(occ).max()
    ax2.imshow(occ, cmap=CMAP_OCC, alpha=0.55, interpolation='bilinear',
               vmin=-occ_abs, vmax=occ_abs)
    ax2.set_xticks([]); ax2.set_yticks([])

# shared colourbars
sm_sal = ScalarMappable(cmap=CMAP_SAL, norm=Normalize(0, 1))
sm_sal.set_array([])
cbar_sal = fig.colorbar(sm_sal, ax=axes[:, 1], shrink=0.6, pad=0.02)
cbar_sal.set_label('normalised |∇ score|', fontsize=9)

sm_occ = ScalarMappable(cmap=CMAP_OCC, norm=Normalize(-1, 1))
sm_occ.set_array([])
cbar_occ = fig.colorbar(sm_occ, ax=axes[:, 2], shrink=0.6, pad=0.02)
cbar_occ.set_label('score drop (red = important)', fontsize=9)

fig.suptitle(
    'BF Cellpose embedding — attention maps\n'
    f'Score = weighted sum of top-20 b2r dims  |  '
    f'Patch {OCC_PATCH}×{OCC_PATCH} stride {OCC_STRIDE}',
    fontsize=12, y=1.002,
)

png_path = OUT_DIR / 'attention_maps_10cells.png'
pdf_path = OUT_DIR / 'attention_maps_10cells_hires.pdf'

fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
fig.savefig(str(pdf_path), bbox_inches='tight')
plt.close(fig)

print(f'Saved: {png_path}')
print(f'Saved: {pdf_path}')
print('\nDone.', flush=True)

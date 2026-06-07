#!/usr/bin/env python3
"""
Attention rollout visualisation for Cellpose SAM ViT encoder.

ViT processes 256x256 crops → 32x32 spatial tokens (8x8 patches, no CLS token).
Rollout: A_hat = 0.5*A + 0.5*I per layer (residual), multiply across 24 layers.
Saliency = rollout.mean(axis=0) reshaped to (32,32), upsampled to (256,256).

Shows 6 cells: 3 earliest + 3 latest by delay_blue_to_red.
Each cell: GFP crop | attention heatmap | overlay.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.ndimage import zoom
from cellpose import models

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')
FIGURES = BASE / 'figures'

MODEL_PATH = str(BASE / 'Cellpose_Cells_Model' / 'cpsam_20260328_104454')
GFP_TIFF   = LIVEIMG / 'CellposeEmbedding' / 'B2_GFP_raw.tif'
SPOTS_CSV  = LIVEIMG / 'CompleteImage' / 'A2_Merged_spots.csv'
MODEL_DF   = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
ONSET_CSV  = LIVEIMG / 'CompleteImage' / 'A2_gfp_onset.csv'

PIXEL_SCALE = 0.2871   # µm/px
CROP        = 128      # half-crop (full crop = 256x256)
SEED        = 42

# ── Load metadata ──────────────────────────────────────────────────────────
print('Loading metadata...', flush=True)
df = pd.read_csv(MODEL_DF)
df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_a2['delay_blue_to_red'] = df_a2['delay_green_to_red'] - df_a2['delay_green_to_blue']
df_a2 = df_a2[np.isfinite(df_a2['delay_blue_to_red'])].reset_index(drop=True)

onset = pd.read_csv(ONSET_CSV)
onset['track_id_int'] = onset['track_id'].astype(str).str.replace('A2_', '', regex=False).astype(int)
onset_map = onset.set_index('track_id_int')[['gfp_onset_frame', 'x_at_onset', 'y_at_onset']].to_dict('index')

# Pick 3 earliest + 3 latest cells
df_sorted = df_a2.sort_values('delay_blue_to_red')
selected = pd.concat([df_sorted.head(3), df_sorted.tail(3)]).reset_index(drop=True)
labels_b2r = list(selected['delay_blue_to_red'].round(0).astype(int))
groups = ['early'] * 3 + ['late'] * 3
print(f'  Selected cells b2r (min): {labels_b2r}')

# ── Load model ─────────────────────────────────────────────────────────────
print('Loading model...', flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  Using device: {device}')
model = models.CellposeModel(gpu=(device == 'cuda'), pretrained_model=MODEL_PATH)
enc = model.net.encoder

# ── Attention hook ─────────────────────────────────────────────────────────
# Monkey-patch each block's Attention.forward to capture softmax attention maps.
# Skip rel_pos for speed (positional bias has minimal effect on spatial importance).
import types

def make_patched_forward(attn_module):
    def patched_forward(x):
        B, H, W, _ = x.shape
        N = H * W
        qkv = attn_module.qkv(x)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each (B, heads, N, head_dim)
        attn = (q * attn_module.scale) @ k.transpose(-2, -1)  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        # store mean-over-heads: (N, N) on CPU float32
        attn_module._saved_attn = attn.mean(dim=1).squeeze(0).detach().cpu().float()
        x = (attn @ v).reshape(B, H, W, -1)
        return attn_module.proj(x)
    return patched_forward

for blk in enc.blocks:
    blk.attn.forward = make_patched_forward(blk.attn)

def compute_rollout():
    """Attention rollout across all 24 blocks → (32, 32) saliency map."""
    n = enc.blocks[0].attn._saved_attn.shape[0]  # 1024
    rollout = torch.eye(n)
    for blk in enc.blocks:
        A = blk.attn._saved_attn        # (1024, 1024)
        A_hat = 0.5 * A + 0.5 * torch.eye(n)
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)   # renormalise rows
        rollout = A_hat @ rollout
    # mean attention received per spatial position → saliency
    saliency = rollout.mean(dim=0).numpy()   # (1024,)
    return saliency.reshape(32, 32)

# ── Load GFP TIFF ──────────────────────────────────────────────────────────
print('Memory-mapping GFP TIFF...', flush=True)
tif = tifffile.memmap(str(GFP_TIFF), mode='r')   # (289, 9357, 9051)

# ── Process each cell ─────────────────────────────────────────────────────
print('Processing cells...', flush=True)
results = []

for i, row in selected.iterrows():
    tid = int(row['track_id'])
    info = onset_map.get(tid)
    if info is None:
        print(f'  WARNING: no onset info for track {tid}'); continue
    frame = int(info['gfp_onset_frame'])
    cx = int(round(float(info['x_at_onset']) / PIXEL_SCALE))
    cy = int(round(float(info['y_at_onset']) / PIXEL_SCALE))

    H, W = tif.shape[1], tif.shape[2]
    y0, y1 = max(0, cy - CROP), min(H, cy + CROP)
    x0, x1 = max(0, cx - CROP), min(W, cx + CROP)
    raw_crop = np.array(tif[frame, y0:y1, x0:x1]).astype(np.float32)

    # zero-pad to 256x256 if near boundary
    pad_crop = np.zeros((256, 256), dtype=np.float32)
    ph = min(raw_crop.shape[0], 256)
    pw = min(raw_crop.shape[1], 256)
    pad_crop[:ph, :pw] = raw_crop[:ph, :pw]

    # run forward pass with attention hooks active
    model.eval([pad_crop], diameter=40)

    # compute rollout
    saliency = compute_rollout()    # (32, 32)

    # upsample to 256x256
    scale = 256 / 32
    sal_up = zoom(saliency, scale, order=1)

    results.append({
        'crop':    pad_crop,
        'saliency': sal_up,
        'b2r':     int(labels_b2r[i]),
        'group':   groups[i],
        'track_id': tid,
    })
    print(f'  Cell {tid} ({groups[i]}, b2r={labels_b2r[i]} min) — saliency range '
          f'[{sal_up.min():.4f}, {sal_up.max():.4f}]')

# ── Figure ────────────────────────────────────────────────────────────────
print('Making figure...', flush=True)

n_cells = len(results)
n_cols  = 3   # crop | heatmap | overlay per cell
fig, axes = plt.subplots(n_cells, n_cols, figsize=(n_cols * 3, n_cells * 3))
fig.subplots_adjust(hspace=0.08, wspace=0.05)

col_titles = ['GFP crop', 'Attention rollout', 'Overlay']
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=11, fontweight='bold', pad=6)

for i, res in enumerate(results):
    crop = res['crop']
    sal  = res['saliency']
    group = res['group']
    b2r   = res['b2r']

    # normalise crop for display
    vmin, vmax = np.percentile(crop, 1), np.percentile(crop, 99)
    crop_norm = np.clip((crop - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # crop
    ax = axes[i, 0]
    ax.imshow(crop_norm, cmap='gray', interpolation='nearest')
    ax.set_ylabel(f'{group}\nb2r={b2r} min', fontsize=9, rotation=0,
                  labelpad=70, va='center')
    ax.set_xticks([]); ax.set_yticks([])

    # heatmap
    sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    ax = axes[i, 1]
    ax.imshow(sal_norm, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])

    # overlay
    ax = axes[i, 2]
    ax.imshow(crop_norm, cmap='gray', interpolation='nearest')
    ax.imshow(sal_norm, cmap='hot', alpha=0.55, interpolation='bilinear', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])

# add group separator line
mid = n_cells // 2 - 0.5
for ax_row in axes:
    for ax in ax_row:
        ax.axhline(-0.5, color='white', lw=0.3)

# colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Attention (normalised)', fontsize=9)

out = FIGURES / 'attention_rollout_b2r.png'
fig.savefig(str(out), dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')

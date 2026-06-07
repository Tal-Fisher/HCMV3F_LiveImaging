#!/usr/bin/env python3
"""
02_extract_bf_embeddings.py

Extract 256-dim Cellpose SAM neck embeddings from the brightfield channel,
10 frames BEFORE each cell's GFP onset.

Cell selection:
  - Confident, unambiguous BF↔GFP matches (from 01_bf_gfp_overlap.py)
  - BF allspot track must reach at least onset_frame - 10
  - Result: ~538 cells

Crop centre: BF allspot centroid at (onset_frame - 10), in µm → pixels.
Model: retrained Cellpose SAM on brightfield data.
Hook: model.net.encoder.neck  →  global avg pool  →  256-dim vector.

Outputs:
  embeddings/A2_bf_embeddings_m10.npz   -- gfp_track_ids (N,), embeddings (N,256)
  embeddings/A2_bf_embeddings_m10.csv   -- gfp_track_id + metadata + emb_0..255
  figures/sample_crops_bf_m10.png       -- 10 random 256×256 BF crops
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')

BF_TIFF  = LIVEIMG / 'CompleteImage' / 'A2_BrightField_raw.tif'

MODEL_PATH  = str(BASE / 'models' / 'cpsam_BrightField')
MATCHES_CSV = BASE / 'bf_gfp_matches.csv'
ALLSPOTS    = LIVEIMG / 'CompleteImage' / 'A2_BrightField_allspots.csv'

OUT_DIR  = BASE / 'embeddings'
FIG_DIR  = BASE / 'figures'
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

PIXEL_SCALE    = 0.2871   # µm / px  (same as GFP extraction)
CROP_SIZE      = 256
HALF           = CROP_SIZE // 2
DIAMETER       = 40
LOOKBACK       = 10       # frames before GFP onset to use
MATCH_TIERS    = {'confident', 'plausible'}   # dist < 1.0 × BF radius
N_SAMPLE_CROPS = 10
RANDOM_SEED    = 42

# ── Guard: image file must exist ───────────────────────────────────────────────
if not BF_TIFF.exists():
    raise FileNotFoundError(
        f"BF raw tif not found: {BF_TIFF}\n"
        "Set BF_TIFF at the top of this script to the correct path."
    )

# ── Load Cellpose model ────────────────────────────────────────────────────────
print('Loading Cellpose BF model...', flush=True)
from cellpose import models
model = models.CellposeModel(pretrained_model=MODEL_PATH, gpu=True)
device = next(model.net.parameters()).device
print(f'  device: {device}', flush=True)

# ── Register neck hook (same as GFP extraction) ────────────────────────────────
# model.eval() style vector (result[2]) is always zeros in Cellpose 4 SAM.
# Real embedding captured from model.net.encoder.neck via forward hook.
neck_outputs = []

def _hook_fn(module, inp, output):
    neck_outputs.append(output.detach().cpu().float())

hook_handle = model.net.encoder.neck.register_forward_hook(_hook_fn)

# ── Load BF↔GFP matches ────────────────────────────────────────────────────────
print('Loading match table...', flush=True)
matches = pd.read_csv(MATCHES_CSV)

# Unambiguous matches within 1.0 × BF radius (confident + plausible tiers)
matches["bf_track_id"] = pd.to_numeric(matches["bf_track_id"], errors="coerce")
matches["bf_earliest_frame"] = pd.to_numeric(matches["bf_earliest_frame"], errors="coerce")

cells = matches[
    (matches["match_tier"].isin(MATCH_TIERS)) &
    (~matches["is_ambiguous"]) &
    matches["bf_track_id"].notna() &
    (matches["bf_earliest_frame"] <= matches["onset_frame"] - LOOKBACK)
].copy()
cells["bf_track_id"] = cells["bf_track_id"].astype(int)
cells["gfp_track_id"] = cells["gfp_track_id"].astype(int)
cells["target_frame"] = cells["onset_frame"] - LOOKBACK

print(f'  Cells eligible (tiers={MATCH_TIERS}, unambiguous, {LOOKBACK}-frame BF history): {len(cells)}', flush=True)

# ── Load BF allspot positions at target_frame ──────────────────────────────────
print('Loading BF allspot positions for target frames...', flush=True)
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
bf_all["FRAME"] = bf_all["FRAME"].astype(int)

# Build lookup: (bf_track_id, frame) → (x_um, y_um)
bf_pos = bf_all.set_index(["TRACK_ID", "FRAME"])[["POSITION_X", "POSITION_Y"]].to_dict("index")
print(f'  BF allspot positions loaded: {len(bf_pos)} entries', flush=True)

# ── Memmap BF TIFF ─────────────────────────────────────────────────────────────
print(f'Memmapping {BF_TIFF.name}...', flush=True)
bf_img = tifffile.memmap(str(BF_TIFF))
if bf_img.ndim == 4:
    # Multi-channel tif: assume BF is the last channel (index -1) or first
    # Update the slice below if channel index differs
    raise ValueError(
        f"BF tif has {bf_img.ndim} dims (shape {bf_img.shape}). "
        "Expected T×Y×X. Set the correct channel slice in get_crop()."
    )
T, H, W = bf_img.shape
print(f'  Shape: {bf_img.shape}, dtype: {bf_img.dtype}', flush=True)

def get_crop(frame, cx, cy):
    """256×256 crop centred on (cx, cy) pixels, zero-padded at boundaries."""
    y0, y1 = cy - HALF, cy + HALF
    x0, x1 = cx - HALF, cx + HALF
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    patch = bf_img[frame, iy0:iy1, ix0:ix1]
    crop  = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    crop[iy0 - y0 : iy0 - y0 + (iy1 - iy0),
         ix0 - x0 : ix0 - x0 + (ix1 - ix0)] = patch
    return crop

# ── Extract embeddings ─────────────────────────────────────────────────────────
rng        = np.random.default_rng(RANDOM_SEED)
sample_ids = set(
    rng.choice(cells["gfp_track_id"].values, size=min(N_SAMPLE_CROPS, len(cells)), replace=False)
    .tolist()
)

track_ids_out  = []
embeddings_out = []
meta_out       = []
sample_crops   = {}   # gfp_track_id → (crop, target_frame)

print(f'Extracting embeddings (n={len(cells)})...', flush=True)
n_total   = len(cells)
n_skipped = 0

for i, (_, row) in enumerate(cells.iterrows()):
    gfp_tid    = int(row["gfp_track_id"])
    bf_tid     = int(row["bf_track_id"])
    target_f   = int(row["target_frame"])
    onset_f    = int(row["onset_frame"])

    key = (bf_tid, target_f)
    if key not in bf_pos:
        print(f'  WARNING: BF track {bf_tid} not found at frame {target_f} — skipping', flush=True)
        n_skipped += 1
        continue

    x_um = bf_pos[key]["POSITION_X"]
    y_um = bf_pos[key]["POSITION_Y"]
    cx   = int(round(x_um / PIXEL_SCALE))
    cy   = int(round(y_um / PIXEL_SCALE))

    if not (0 <= cx < W and 0 <= cy < H):
        print(f'  WARNING: track {gfp_tid} centroid ({cx},{cy}) outside image — skipping', flush=True)
        n_skipped += 1
        continue

    crop = get_crop(target_f, cx, cy)

    if gfp_tid in sample_ids:
        sample_crops[gfp_tid] = (crop.copy(), target_f, onset_f)

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
        print(f'  WARNING: no neck output for GFP track {gfp_tid}', flush=True)
        n_skipped += 1
        continue

    # (batch, 256, H', W') → global avg pool → (256,)
    emb = neck_outputs[0].mean(dim=[2, 3])[0].numpy().astype(np.float32)

    track_ids_out.append(gfp_tid)
    embeddings_out.append(emb)
    meta_out.append({"gfp_track_id": gfp_tid, "bf_track_id": bf_tid,
                     "onset_frame": onset_f, "target_frame": target_f,
                     "x_um": x_um, "y_um": y_um})

    if i == 0:
        print(f'  First cell — emb shape: {emb.shape}, std: {emb.std():.4f}  [sanity]', flush=True)
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{n_total} cells done  ({n_skipped} skipped so far)', flush=True)

hook_handle.remove()

# ── Save embeddings ────────────────────────────────────────────────────────────
track_ids_arr  = np.array(track_ids_out,  dtype=np.int64)
embeddings_arr = np.array(embeddings_out, dtype=np.float32)

print(f'\nEmbeddings shape : {embeddings_arr.shape}')
print(f'Embedding stats  : mean={embeddings_arr.mean():.4f}  std={embeddings_arr.std():.4f}')
print(f'Skipped          : {n_skipped}')

npz_path = OUT_DIR / 'A2_bf_embeddings_m10_relaxed.npz'
np.savez(str(npz_path), gfp_track_ids=track_ids_arr, embeddings=embeddings_arr)
print(f'Saved: {npz_path}')

# CSV: gfp_track_id + metadata + emb_0..255
dim    = embeddings_arr.shape[1]
meta_df = pd.DataFrame(meta_out)
emb_df  = pd.DataFrame(embeddings_arr, columns=[f'emb_{k}' for k in range(dim)])
out_df  = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)
csv_path = OUT_DIR / 'A2_bf_embeddings_m10_relaxed.csv'
out_df.to_csv(str(csv_path), index=False)
print(f'Saved: {csv_path}')

# ── Save 10 sample crop PNGs ───────────────────────────────────────────────────
print(f'\nSaving {len(sample_crops)} sample crops...')
ncols = 5
nrows = int(np.ceil(len(sample_crops) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for ax, (tid, (crop, target_f, onset_f)) in zip(axes, sample_crops.items()):
    vmin = np.percentile(crop, 0.5) if crop.max() > 0 else 0
    vmax = np.percentile(crop, 99.5) if crop.max() > 0 else 1
    ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'GFP track {tid}\nframe {target_f} (onset-{LOOKBACK})', fontsize=8)
    ax.axis('off')

# hide unused axes
for ax in axes[len(sample_crops):]:
    ax.axis('off')

fig.suptitle(f'BF crops 256×256 px  —  {LOOKBACK} frames before GFP onset', fontsize=11)
plt.tight_layout()
png_path = FIG_DIR / 'sample_crops_bf_m10_relaxed.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {png_path}')

print('\nDone.', flush=True)

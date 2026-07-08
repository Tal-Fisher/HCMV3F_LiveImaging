#!/usr/bin/env python3
"""
03_saliency_maps.py

Gradient (vanilla-gradient) saliency maps from the from-scratch dual-branch
CNN (02_train_cnn_multitask.py), for the SAME 10 A2 cells (5 "fast"/5 "slow"
by blue-to-red delay) used in the existing Cellpose-embedding attention-map
analysis (BrightFieldEmbedding/Attention_maps/01_attention_maps.py), so the
two saliency analyses can be compared side by side for the same cell
identities.

Cell selection (mirrors 01_attention_maps.py EXACTLY, same source files/seed)
-------------------------------------------------------------------------
01_attention_maps.py does NOT pick the most extreme (fastest/slowest) cells.
It:
  1. Restricts to A2 cells present in the BF-embedding cache
     (BrightFieldEmbedding/embeddings/A2_bf_embeddings_m10_relaxed.npz,
     key 'gfp_track_ids') with a finite blue-to-red delay
     (results/elasticnet_extended2/model_df_extended2.csv).
  2. Sorts that pool by track_id ascending, splits it at CUT_B2R=1094 min
     into "early" (<=1094) and "med+late" (>1094) buckets.
  3. Draws 5 cells uniformly at random (np.random.default_rng(42).choice,
     replace=False) from EACH bucket -- NOT the 5 most extreme in either
     direction.
This script reproduces that exact selection code (same CSVs, same seed, same
call order) to get the identical 10 track_ids, then looks those track_ids up
in THIS pipeline's own A2 population (CNN/patches/A2_patches_all.npz, joined
with model_df.csv under the productive+HMF filter used by
02_train_cnn_multitask.py's load_dataset()). If a selected track_id fell
outside that population (e.g. filtered by the half-movie filter), it is
skipped with a warning -- see the printed summary for how many were found.
Cells are labelled 'fast' (b2r<=1094) / 'slow' (b2r>1094) here, equivalent to
the reference script's 'early' / 'med+late' (same 1094 min cutoff).

Out-of-fold checkpoint selection
---------------------------------
For each selected cell, the CNN/checkpoints/A2_dual_branch_multitask_fold{0..4}.pt
checkpoint whose 'test_track_ids' contains that cell's track_id is used --
i.e. saliency is always computed from a model for which that cell was
held out (never trained on), avoiding memorization artifacts.

Saliency method
----------------
Vanilla gradient saliency (same core method as 01_attention_maps.py's
"Approach 1 -- Gradient saliency": requires_grad_(True) on the input,
backward from a scalar score, saliency = |d(score)/d(input)|, Gaussian-
smoothed with sigma=4). Adaptation for this network: the reference script's
score is a fixed weighted sum over embedding dimensions (because its "model"
only outputs a 256-dim embedding, not a task head); this network has actual
task heads, so the natural score is either the raw classification logit
(pre-sigmoid) or the raw (standardized) regression output -- both are
computed here as two separate saliency maps per cell. The gradient is over a
genuine 2-channel (GFP, BF) pixel input rather than a 3-channel RGB-cast
embedding-model input, so the two channels are reported separately instead
of being averaged.

Outputs
-------
  figures/saliency_comparison.png   (10 rows: 5 fast, 5 slow, divided by a
                                      horizontal rule; 6 cols: GFP crop,
                                      BF crop, GFP cls-saliency, BF
                                      cls-saliency, GFP reg-saliency,
                                      BF reg-saliency)
  stdout summary table: track_id, status, b2r (min), cls probability,
                         regression prediction (min), fold used.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BASE      = Path('/home/labs/ginossar/talfis/LiveImaging/CNN')
LIVEIMG   = Path('/home/labs/ginossar/talfis/LiveImaging')
MODEL_DF  = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
PATCH_DIR = BASE / 'patches'
CKPT_DIR  = BASE / 'checkpoints'
FIG_DIR   = BASE / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- cell-selection sources, identical to 01_attention_maps.py --
ATTN_BASE  = LIVEIMG / 'BrightFieldEmbedding'
EXT_DF_CSV = LIVEIMG / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'
EMB_NPZ    = ATTN_BASE / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'

CUT_B2R     = 1094   # minutes -- fast/slow cutoff (same constant as everywhere else)
N_PER_CLASS = 5
SEED        = 42
N_FOLDS     = 5
DS_LABEL    = 'A2'   # scope matches 01_attention_maps.py (A2 only)

PATCH_FILES = {ds: PATCH_DIR / f'{ds}_patches_all.npz' for ds in ('A2', 'A3')}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

mdf = pd.read_csv(MODEL_DF)


# ═══════════════════════════════════════════════════════════════════════════
# NETWORK -- duplicated VERBATIM from 02_train_cnn_multitask.py.
# Must stay byte-for-byte identical to that file's ConvTower/DualBranchCNN,
# or checkpoint state_dicts will fail to load / silently mismatch.
# ═══════════════════════════════════════════════════════════════════════════

class ConvTower(nn.Module):
    """Conv(k5,s2)+BN+ReLU -> Conv(k3,s2)+BN+ReLU -> Conv(k3,s2)+BN+ReLU
    -> Conv(k3,s2)+BN+ReLU -> GlobalAvgPool. 256->128->64->32->16, ch 8/16/32/32."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),    nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),   nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.net(x)
        return self.gap(x).flatten(1)   # (B, 32)


class DualBranchCNN(nn.Module):
    """Two separate 1-channel conv towers (GFP, BF; separate weights) ->
    concat(64) -> shared head Linear(64->32)+ReLU+Dropout -> cls + reg heads."""
    def __init__(self, multitask=True, drop=0.45):
        super().__init__()
        self.gfp_tower = ConvTower(in_ch=1)
        self.bf_tower  = ConvTower(in_ch=1)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(drop))
        self.cls_head = nn.Linear(32, 1)
        self.reg_head = nn.Linear(32, 1) if multitask else None

    def forward(self, x):
        g = self.gfp_tower(x[:, 0:1])
        b = self.bf_tower(x[:, 1:2])
        h = self.head(torch.cat([g, b], dim=1))
        cls_logit = self.cls_head(h).squeeze(1)
        reg_out   = self.reg_head(h).squeeze(1) if self.reg_head is not None else None
        return cls_logit, reg_out


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING -- duplicated from 02_train_cnn_multitask.py's load_dataset()
# (same join/filter logic) so track_id -> row indices line up EXACTLY with
# what run_cv() used when it wrote 'test_track_ids' into the checkpoints.
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(ds_list):
    records = []
    patch_chunks = []
    norm_p1, norm_p995 = {}, {}
    offset = 0

    for ds in ds_list:
        npz_path = PATCH_FILES[ds]
        if not npz_path.exists():
            raise FileNotFoundError(
                f'Patch cache missing: {npz_path} -- run 01_extract_raw_patches.py '
                f'--dataset {ds} first.')
        d = np.load(str(npz_path))
        tids = d['track_ids']
        for i, tid in enumerate(tids):
            records.append({'dataset': ds, 'track_id': int(tid), 'row': offset + i})
        patch_chunks.append(d['patches'])
        norm_p1[ds]   = d['norm_p1'].astype(np.float32)
        norm_p995[ds] = d['norm_p995'].astype(np.float32)
        offset += len(tids)

    PATCHES  = np.concatenate(patch_chunks, axis=0)   # uint8 (N, 2, 256, 256)
    patch_df = pd.DataFrame(records)

    meta_chunks = []
    for ds in ds_list:
        sub = mdf[mdf['dataset'] == ds].copy()
        sub['track_id'] = sub['Track.ID'].str.replace(f'{ds}_', '', regex=False).astype(int)
        sub['b2r']      = sub['delay_green_to_red'] - sub['delay_green_to_blue']
        meta_chunks.append(sub)
    meta = pd.concat(meta_chunks).reset_index(drop=True)

    merged = meta.merge(patch_df, on=['dataset', 'track_id'], how='inner')
    # see 02_train_cnn_multitask.py's load_dataset() for why isfinite (not notna)
    # is required: our patch cache includes non-productive cells whose b2r can be
    # +inf (not NaN) rather than simply absent.
    merged = merged[np.isfinite(merged['b2r'])]
    if 'abs_gfp_onset_min' in merged.columns and 'movie_half_min' in merged.columns:
        merged = merged[merged['abs_gfp_onset_min'] <= merged['movie_half_min']]
    else:
        print('  WARNING: half-movie filter columns not found -- skipping filter', flush=True)
    merged = merged.sort_values(['dataset', 'track_id']).reset_index(drop=True)

    X_raw = PATCHES[merged['row'].values]           # uint8 (n, 2, 256, 256)
    X01   = np.zeros(X_raw.shape, dtype=np.float32)  # stage-1 normalized, [0,1]

    for ds in ds_list:
        mask = (merged['dataset'] == ds).values
        if not mask.any():
            continue
        p1, p995 = norm_p1[ds], norm_p995[ds]
        for c in range(2):
            lo, hi = float(p1[c]), float(p995[c])
            chan = X_raw[mask, c].astype(np.float32)
            chan = np.clip(chan, lo, hi)
            chan = (chan - lo) / max(hi - lo, 1e-6)
            X01[mask, c] = chan

    y_reg = merged['b2r'].values.astype(np.float32)
    y_cls = (y_reg <= CUT_B2R).astype(np.int64)
    return X01, y_cls, y_reg, merged


# ═══════════════════════════════════════════════════════════════════════════
# CELL SELECTION -- reproduces 01_attention_maps.py's selection code exactly
# (same source files, same seed, same call order) to get identical track_ids.
# ═══════════════════════════════════════════════════════════════════════════

print('\nSelecting cells (replicating 01_attention_maps.py selection)...', flush=True)

emb_data      = np.load(str(EMB_NPZ))
emb_track_ids = set(emb_data['gfp_track_ids'].tolist())

ext = pd.read_csv(EXT_DF_CSV)
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id']          = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
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
# 'fast'/'slow' here == reference script's 'early'/'med+late' (same 1094 min cutoff)
sel_cells['status'] = (['fast'] * N_PER_CLASS + ['slow'] * N_PER_CLASS)

print(f'  Selected {len(sel_cells)} cells (same identities as 01_attention_maps.py): '
      f'{N_PER_CLASS} fast, {N_PER_CLASS} slow')
for _, row in sel_cells.iterrows():
    print(f'    track {int(row.track_id):6d}  b2r={row.delay_blue_to_red:.0f} min  [{row.status}]')


# ═══════════════════════════════════════════════════════════════════════════
# LOAD THIS PIPELINE'S A2 POPULATION + MATCH SELECTED TRACK_IDS INTO IT
# ═══════════════════════════════════════════════════════════════════════════

print(f'\nLoading {DS_LABEL} patch cache + productive/HMF-filtered population...', flush=True)
X01, y_cls, y_reg, merged = load_dataset([DS_LABEL])
pop_row_of = {int(tid): i for i, tid in enumerate(merged['track_id'].values)}
print(f'  Population size: {len(merged)} cells', flush=True)

found_rows = []
for _, row in sel_cells.iterrows():
    tid = int(row['track_id'])
    if tid not in pop_row_of:
        print(f'  [WARN] track {tid} not in {DS_LABEL} CNN population '
              f'(productive+HMF filter) -- skipping', flush=True)
        continue
    found_rows.append(dict(track_id=tid, b2r=float(row['delay_blue_to_red']),
                            status=row['status'], pop_idx=pop_row_of[tid]))

if not found_rows:
    raise RuntimeError('None of the 10 selected cells were found in the CNN A2 population.')

print(f'  {len(found_rows)}/10 selected cells found in the CNN population.', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ALL 5 FOLD CHECKPOINTS, BUILD track_id -> fold LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

print('\nLoading checkpoints...', flush=True)
checkpoints = {}
fold_of_track = {}
for fold in range(N_FOLDS):
    ckpt_path = CKPT_DIR / f'{DS_LABEL}_dual_branch_multitask_fold{fold}.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Checkpoint missing: {ckpt_path} -- run 02_train_cnn_multitask.py first '
            f'(it must finish and save all {N_FOLDS} fold checkpoints for {DS_LABEL}).')
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    checkpoints[fold] = ckpt
    for tid in np.asarray(ckpt['test_track_ids']).tolist():
        fold_of_track[int(tid)] = fold
print(f'  Loaded {N_FOLDS} checkpoints.', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# GRADIENT SALIENCY
# ═══════════════════════════════════════════════════════════════════════════

def compute_saliency(model, x01_2ch, pix_mean, pix_std, target):
    """x01_2ch: (2,256,256) float32 in [0,1], stage-1 normalized.
    pix_mean/pix_std: (1,2,1,1) float32 arrays from the checkpoint.
    target: 'cls' (backprop from cls_logit) or 'reg' (backprop from reg_out).
    Returns (saliency (2,256,256) float32 abs-gradient Gaussian-smoothed,
             cls_prob float, reg_val float [raw standardized reg_out]).
    Vanilla-gradient saliency, mirroring 01_attention_maps.py's
    gradient_saliency(): |d(score)/d(input)|, Gaussian sigma=4 -- adapted
    here to backprop from an actual task head instead of a fixed weighted
    embedding-dim sum, and reported per-channel (GFP, BF) rather than
    averaged over channels (the input here is genuine 2-channel pixels, not
    a 3-channel RGB-cast embedding-model input).
    """
    mean = pix_mean.reshape(1, 2, 1, 1).astype(np.float32)
    std  = pix_std.reshape(1, 2, 1, 1).astype(np.float32)
    x_z  = (x01_2ch[None] - mean) / std   # (1,2,256,256)
    x_t  = torch.tensor(x_z, dtype=torch.float32, device=device)
    x_t.requires_grad_(True)

    model.zero_grad()
    cls_logit, reg_out = model(x_t)
    cls_prob = torch.sigmoid(cls_logit).item()
    reg_val  = reg_out.item()

    score = cls_logit.squeeze() if target == 'cls' else reg_out.squeeze()
    score.backward()

    grad = x_t.grad.detach().cpu().numpy()[0]   # (2,256,256)
    saliency = np.abs(grad)
    for c in range(2):
        saliency[c] = gaussian_filter(saliency[c], sigma=4)
    return saliency, cls_prob, reg_val


print('\nComputing saliency maps for each cell (out-of-fold checkpoint)...', flush=True)

results = []
for rec in found_rows:
    tid = rec['track_id']
    if tid not in fold_of_track:
        print(f'  [WARN] track {tid} not held out in any fold checkpoint -- skipping', flush=True)
        continue
    fold = fold_of_track[tid]
    ckpt = checkpoints[fold]

    model = DualBranchCNN(multitask=True).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # only the input carries gradients

    x01 = X01[rec['pop_idx']]   # (2,256,256), stage-1 normalized [0,1]

    sal_cls, cls_prob, _        = compute_saliency(model, x01, ckpt['pix_mean'], ckpt['pix_std'], target='cls')
    sal_reg, _, reg_out_std     = compute_saliency(model, x01, ckpt['pix_mean'], ckpt['pix_std'], target='reg')
    reg_pred_min = reg_out_std * ckpt['y_reg_std'] + ckpt['y_reg_mean']

    results.append(dict(
        track_id=tid, status=rec['status'], b2r=rec['b2r'], fold=fold,
        cls_prob=cls_prob, reg_pred_min=reg_pred_min,
        gfp_crop=x01[0], bf_crop=x01[1],
        sal_cls_gfp=sal_cls[0], sal_cls_bf=sal_cls[1],
        sal_reg_gfp=sal_reg[0], sal_reg_bf=sal_reg[1],
    ))
    print(f'  track {tid:6d}  [{rec["status"]}]  fold={fold}  '
          f'cls_p={cls_prob:.3f}  reg_pred={reg_pred_min:.0f} min', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

summary_df = pd.DataFrame([{
    'track_id': r['track_id'], 'status': r['status'], 'b2r_min': round(r['b2r'], 1),
    'cls_prob': round(r['cls_prob'], 3), 'reg_pred_min': round(r['reg_pred_min'], 1),
    'fold': r['fold'],
} for r in results])
print('\n' + '=' * 78)
print('SUMMARY')
print('=' * 78)
print(summary_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: 10 rows (5 fast, 5 slow, divided) x 6 cols
# ═══════════════════════════════════════════════════════════════════════════

print('\nGenerating figure...', flush=True)

fast_results = [r for r in results if r['status'] == 'fast']
slow_results = [r for r in results if r['status'] == 'slow']
ordered = fast_results + slow_results
n_cells = len(ordered)
n_fast  = len(fast_results)

CMAP_SAL = 'inferno'

col_titles = ['GFP crop', 'BF crop', 'GFP cls-saliency', 'BF cls-saliency',
              'GFP reg-saliency', 'BF reg-saliency']

fig, axes = plt.subplots(n_cells, 6, figsize=(18, 3.0 * n_cells), constrained_layout=True)
if n_cells == 1:
    axes = axes[None]

for j, ct in enumerate(col_titles):
    axes[0, j].set_title(ct, fontsize=11, fontweight='bold', pad=8)

def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)

for row_i, r in enumerate(ordered):
    row_label = (f'track {r["track_id"]}\nb2r={r["b2r"]:.0f} min  [{r["status"]}]\n'
                 f'fold {r["fold"]}  cls_p={r["cls_prob"]:.2f}\nreg={r["reg_pred_min"]:.0f} min')

    axes[row_i, 0].imshow(r['gfp_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[row_i, 0].set_ylabel(row_label, fontsize=7.5, labelpad=4)

    axes[row_i, 1].imshow(r['bf_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    axes[row_i, 2].imshow(r['gfp_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[row_i, 2].imshow(norm01(r['sal_cls_gfp']), cmap=CMAP_SAL, alpha=0.55,
                           interpolation='bilinear', vmin=0, vmax=1)

    axes[row_i, 3].imshow(r['bf_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[row_i, 3].imshow(norm01(r['sal_cls_bf']), cmap=CMAP_SAL, alpha=0.55,
                           interpolation='bilinear', vmin=0, vmax=1)

    axes[row_i, 4].imshow(r['gfp_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[row_i, 4].imshow(norm01(r['sal_reg_gfp']), cmap=CMAP_SAL, alpha=0.55,
                           interpolation='bilinear', vmin=0, vmax=1)

    axes[row_i, 5].imshow(r['bf_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[row_i, 5].imshow(norm01(r['sal_reg_bf']), cmap=CMAP_SAL, alpha=0.55,
                           interpolation='bilinear', vmin=0, vmax=1)

    for j in range(6):
        axes[row_i, j].set_xticks([]); axes[row_i, j].set_yticks([])

# shared colourbar for saliency columns
sm_sal = ScalarMappable(cmap=CMAP_SAL, norm=Normalize(0, 1))
sm_sal.set_array([])
cbar = fig.colorbar(sm_sal, ax=axes[:, 2:6], shrink=0.5, pad=0.02)
cbar.set_label('normalised |∇ score|', fontsize=9)

# horizontal divider + group labels between fast/slow blocks
if 0 < n_fast < n_cells:
    fig.canvas.draw()
    y_bottom_fast = axes[n_fast - 1, 0].get_position().y0
    y_top_slow    = axes[n_fast, 0].get_position().y1
    y_div = (y_bottom_fast + y_top_slow) / 2
    fig.add_artist(Line2D([0.0, 1.0], [y_div, y_div], transform=fig.transFigure,
                           color='black', lw=1.5, linestyle='--'))
    fig.text(0.005, (y_div + 1.0) / 2, 'FAST\n(b2r ≤ 1094 min)', fontsize=9,
              fontweight='bold', rotation=90, va='center', ha='left')
    fig.text(0.005, y_div / 2, 'SLOW\n(b2r > 1094 min)', fontsize=9,
              fontweight='bold', rotation=90, va='center', ha='left')

fig.suptitle(
    'From-scratch dual-branch CNN — gradient saliency (out-of-fold checkpoints)\n'
    'cls-saliency = |∇ cls_logit|  |  reg-saliency = |∇ reg_out (standardized)|  |  '
    f'A2, cut={CUT_B2R} min  |  same 10 cells as BrightFieldEmbedding/Attention_maps',
    fontsize=11, y=1.01,
)

out_path = FIG_DIR / 'saliency_comparison.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')

print('\nDone.', flush=True)

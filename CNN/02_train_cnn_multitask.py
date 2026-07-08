#!/usr/bin/env python3
"""
02_train_cnn_multitask.py

From-scratch CNN on raw GFP+BF pixel patches (256x256, at GFP onset frame),
predicting both fast/slow b2r classification and the blue-to-red delay in
minutes (regression), jointly (multi-task) via a shared conv backbone.

This is deliberately NOT expected to beat the existing Cellpose-embedding
pipeline (LogReg fusion AUC=0.712, BF embedding onset-10 AUC=0.876) -- the
point of this experiment is diagnostic: does a network trained from scratch,
with no segmentation-pretraining confound, discover similar morphological
signal (checked later via saliency maps, 03_saliency_maps.py) to what the
Cellpose-embedding pipeline implicitly uses?

Requires: CNN/01_extract_raw_patches.py must have been run for A2 and A3
(produces CNN/patches/{DATASET}_patches_all.npz).

Architectures
-------------
  DualBranchCNN  : two separate 1-channel conv towers (GFP, BF) -> concat(64)
                   -> shared head -> classification + regression heads.
  EarlyFusionCNN : single 2-channel conv tower -> head -> both heads.

Conv tower (per branch): Conv(k5,s2)+BN+ReLU (256->128, ch 1->8)
                          -> Conv(k3,s2)+BN+ReLU (128->64, ch 8->16)
                          -> Conv(k3,s2)+BN+ReLU (64->32, ch 16->32)
                          -> Conv(k3,s2)+BN+ReLU (32->16, ch 32->32)
                          -> GlobalAvgPool -> 32-dim

Normalization (two-stage, see plan)
------------------------------------
  1. Fixed per-movie, per-channel percentile clip+rescale (norm_p1/norm_p995
     stored in each dataset's npz), applied once when building the tensor.
  2. Per-fold z-score standardization (mean/std computed on train-fold pixels
     only), matching this repo's per-fold-scaler-on-train-only convention.

Loss: BCEWithLogitsLoss(pos_weight) + lambda * SmoothL1Loss (on per-fold
z-scored regression target). Label smoothing 0.08 on classification target.
Lambda swept over {0.3, 1.0, 3.0} for the DualBranchCNN-multitask config only.

Cross-validation: StratifiedKFold(5, shuffle, seed=42) on y_cls, over
conditions A2 / A3 / A2+A3, matching GFP_BF_Fusion/09_slim_net.py exactly.

Outputs
-------
  results/raw_cnn_multitask_metrics.csv
  results/raw_cnn_multitask_oof.csv
  figures/raw_cnn_auc_bars.png
  figures/raw_cnn_roc.png
  checkpoints/*.pt   (primary config only: dual_branch/multitask/lambda=1.0,
                       one file per fold per condition -- for a later
                       03_saliency_maps.py to load)
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                              balanced_accuracy_score, matthews_corrcoef, r2_score)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

warnings.filterwarnings('ignore')
torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150,
                     help='max epochs per fold (default 150)')
parser.add_argument('--patience', type=int, default=25,
                     help='early-stopping patience on val AUC (default 25)')
parser.add_argument('--batch-size', type=int, default=16,
                     help='training batch size (default 16, small N)')
parser.add_argument('--quick', action='store_true',
                     help='smoke-test mode: A2 only, dual_branch/multitask/'
                          'lambda=1.0 only, <=5 epochs -- for fast sanity checks '
                          'before submitting the full sweep')
args = parser.parse_args()

if args.quick:
    args.epochs = min(args.epochs, 5)
    args.patience = min(args.patience, 5)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BASE      = Path('/home/labs/ginossar/talfis/LiveImaging/CNN')
LIVEIMG   = Path('/home/labs/ginossar/talfis/LiveImaging')
MODEL_DF  = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
PATCH_DIR = BASE / 'patches'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
CKPT_DIR    = BASE / 'checkpoints'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CUT_B2R  = 1094   # minutes -- fast/slow cutoff, matches GFP_BF_Fusion convention
SEED     = 42
N_SPLITS = 5
LAMBDA_SWEEP = [0.3, 1.0, 3.0]   # diagnostic sweep, DualBranchCNN-multitask only

PATCH_FILES = {ds: PATCH_DIR / f'{ds}_patches_all.npz' for ds in ('A2', 'A3')}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

mdf = pd.read_csv(MODEL_DF)

print('=' * 78, flush=True)
print('WORKING HYPOTHESIS (H0): a from-scratch CNN trained directly on raw GFP+BF '
      'pixel patches at the GFP onset frame will NOT exceed the existing tabular/'
      'embedding baselines at this sample size (~500 labeled cells). Success here '
      'is measured by diagnostic/interpretability value -- whether saliency maps '
      'from this from-scratch network agree with the Cellpose-embedding pipeline\'s '
      'implicit signal (see 03_saliency_maps.py) -- not by a new best AUC.', flush=True)
print('=' * 78, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(ds_list):
    """Load + join patch caches with model_df labels, exactly mirroring
    GFP_BF_Fusion/09_slim_net.py's load_dataset() join logic, adapted from
    embeddings to raw patches. Applies stage-1 (fixed, per-movie, per-channel
    percentile clip+rescale) normalization here; stage-2 (per-fold z-score)
    is applied later, inside run_cv, on train-fold pixels only.
    """
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
    # NOTE: unlike GFP_BF_Fusion/09_slim_net.py (whose upstream embedding npz was
    # already extracted productive-only, making .notna() a no-op), our patch cache
    # (01_extract_raw_patches.py) intentionally includes ALL cells. b2r = g2r - g2b
    # is +inf (not NaN) whenever red never appeared but blue did, so .notna() alone
    # would silently admit non-productive cells with an infinite regression target
    # -> NaN loss during training. Must filter on finiteness explicitly.
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


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cls_metrics(y, score, pred):
    """score = continuous ranking score for AUC/AP (probability, or -reg_min);
    pred = already-thresholded binary prediction."""
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, score)
        ap  = average_precision_score(y, score)
    else:
        auc = ap = float('nan')
    sens = int(((pred == 1) & (y == 1)).sum()) / max(int(y.sum()), 1)
    spec = int(((pred == 0) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
    bal  = balanced_accuracy_score(y, pred)
    mcc  = matthews_corrcoef(y, pred)
    return dict(auc=round(float(auc), 3), ap=round(float(ap), 3),
                sens=round(sens, 3), spec=round(spec, 3),
                bal_acc=round(bal, 3), mcc=round(mcc, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# ON-THE-FLY AUGMENTATION + DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class PatchDataset(Dataset):
    """Wraps stage-1-normalized ([0,1]) patches. Applies, for the train split
    only: geometric aug (flip/rotate/translate) + brightness/contrast jitter
    in [0,1] space, THEN the fixed per-fold z-score standardization (pix_mean/
    pix_std, computed on train-fold pixels only, identical at train and eval
    time), THEN small additive Gaussian noise in standardized units (train
    only). No scale/RandomResizedCrop augmentation -- cell/nuclear size is a
    real predictive feature.
    """
    def __init__(self, X01, y_cls, y_reg_std, pix_mean, pix_std, train=False,
                 translate_px=12, jitter_pct=0.15, noise_sigma=1.5):
        self.X01      = torch.tensor(X01, dtype=torch.float32)          # (N,2,256,256)
        self.y_cls    = torch.tensor(y_cls, dtype=torch.float32)
        self.y_reg    = torch.tensor(y_reg_std, dtype=torch.float32)
        self.pix_mean = torch.tensor(np.asarray(pix_mean).reshape(2, 1, 1), dtype=torch.float32)
        self.pix_std  = torch.tensor(np.asarray(pix_std).reshape(2, 1, 1),  dtype=torch.float32)
        self.train    = train
        self.jitter_pct  = jitter_pct
        self.noise_sigma = noise_sigma
        if train:
            self.geo_aug = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAffine(degrees=360,
                                 translate=(translate_px / 256, translate_px / 256),
                                 fill=0),
            ])

    def __len__(self):
        return self.X01.shape[0]

    def __getitem__(self, idx):
        x = self.X01[idx].clone()
        if self.train:
            x = self.geo_aug(x)
            b = 1.0 + (torch.rand(2, 1, 1) * 2 - 1) * self.jitter_pct   # brightness
            c = 1.0 + (torch.rand(2, 1, 1) * 2 - 1) * self.jitter_pct   # contrast
            mean_c = x.mean(dim=(1, 2), keepdim=True)
            x = (x - mean_c) * c + mean_c
            x = (x * b).clamp(0.0, 1.0)
        x = (x - self.pix_mean) / self.pix_std
        if self.train:
            x = x + torch.randn_like(x) * self.noise_sigma
        return x, self.y_cls[idx], self.y_reg[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

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


class EarlyFusionCNN(nn.Module):
    """Single 2-channel conv tower (GFP+BF stacked at input) -> head
    Linear(32->32)+ReLU+Dropout -> cls + reg heads. Secondary comparison
    variant against DualBranchCNN's late fusion."""
    def __init__(self, multitask=True, drop=0.45):
        super().__init__()
        self.tower = ConvTower(in_ch=2)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Dropout(drop))
        self.cls_head = nn.Linear(32, 1)
        self.reg_head = nn.Linear(32, 1) if multitask else None

    def forward(self, x):
        h = self.head(self.tower(x))
        cls_logit = self.cls_head(h).squeeze(1)
        reg_out   = self.reg_head(h).squeeze(1) if self.reg_head is not None else None
        return cls_logit, reg_out


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_ds, Xv_t, yv_cls, yv_reg_std, pos_weight,
               lam=1.0, max_epochs=150, patience=25, batch_size=16,
               lr=1e-3, weight_decay=3e-3, label_smooth=0.08):
    model = model.to(device)
    multitask = model.reg_head is not None

    criterion_cls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # BatchNorm needs >1 sample per batch; drop a trailing singleton batch.
    drop_last = len(train_ds) > batch_size
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    yv_cls_np = np.asarray(yv_cls)
    best_auc, best_state, no_improve, stopped = -1.0, None, 0, max_epochs

    for epoch in range(max_epochs):
        model.train()
        for xb, yb_cls, yb_reg in loader:
            xb, yb_cls = xb.to(device), yb_cls.to(device)
            optimizer.zero_grad()
            logit, reg_pred = model(xb)
            yb_cls_smooth = yb_cls * (1 - label_smooth) + 0.5 * label_smooth
            loss = criterion_cls(logit, yb_cls_smooth)
            if multitask:
                loss = loss + lam * criterion_reg(reg_pred, yb_reg.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logit_v, _ = model(Xv_t)
            probs_v = torch.sigmoid(logit_v).cpu().numpy()
        val_auc = roc_auc_score(yv_cls_np, probs_v) if len(np.unique(yv_cls_np)) > 1 else 0.5

        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped = epoch + 1
                break

    model.load_state_dict(best_state)
    return model, best_auc, stopped


def run_cv(X01, y_cls, y_reg, model_factory, lam=1.0,
           max_epochs=150, patience=25, batch_size=16,
           save_ckpt_dir=None, ckpt_prefix=None, track_ids=None):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    n = len(y_cls)
    oof_prob    = np.zeros(n, dtype=np.float64)
    oof_reg_min = np.full(n, np.nan, dtype=np.float64)
    has_reg = None

    for fold, (tr, te) in enumerate(skf.split(X01, y_cls)):
        pix_mean = X01[tr].mean(axis=(0, 2, 3), keepdims=True)   # (1,2,1,1), train-fold only
        pix_std  = X01[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6

        model = model_factory().to(device)
        multitask = model.reg_head is not None
        has_reg = multitask

        if multitask:
            y_reg_mean = float(y_reg[tr].mean())
            y_reg_sd   = float(y_reg[tr].std() + 1e-6)
            y_reg_tr_std = (y_reg[tr] - y_reg_mean) / y_reg_sd
            y_reg_te_std = (y_reg[te] - y_reg_mean) / y_reg_sd
        else:
            y_reg_mean, y_reg_sd = 0.0, 1.0
            y_reg_tr_std = np.zeros(len(tr), dtype=np.float32)
            y_reg_te_std = np.zeros(len(te), dtype=np.float32)

        train_ds = PatchDataset(
            X01[tr], y_cls[tr].astype(np.float32), y_reg_tr_std.astype(np.float32),
            pix_mean, pix_std, train=True)

        val_std = (X01[te] - pix_mean) / pix_std
        Xv_t = torch.tensor(val_std, dtype=torch.float32).to(device)

        n_slow_tr = int((y_cls[tr] == 0).sum())
        n_fast_tr = int((y_cls[tr] == 1).sum())
        pos_w = n_slow_tr / max(n_fast_tr, 1)

        model, best_auc, n_ep = train_fold(
            model, train_ds, Xv_t, y_cls[te], y_reg_te_std, pos_w,
            lam=lam, max_epochs=max_epochs, patience=patience, batch_size=batch_size)

        model.eval()
        with torch.no_grad():
            logit_te, reg_te = model(Xv_t)
            oof_prob[te] = torch.sigmoid(logit_te).cpu().numpy()
            if multitask:
                oof_reg_min[te] = reg_te.cpu().numpy() * y_reg_sd + y_reg_mean

        if save_ckpt_dir is not None:
            torch.save({
                'state_dict': model.state_dict(),
                'pix_mean': pix_mean, 'pix_std': pix_std,
                'y_reg_mean': y_reg_mean, 'y_reg_std': y_reg_sd,
                'test_idx': te,
                'test_track_ids': track_ids[te] if track_ids is not None else te,
            }, save_ckpt_dir / f'{ckpt_prefix}_fold{fold}.pt')

        fold_auc = roc_auc_score(y_cls[te], oof_prob[te]) if len(np.unique(y_cls[te])) > 1 else 0.5
        print(f'      fold {fold+1}: OOF={fold_auc:.3f}  best_val={best_auc:.3f}  ep={n_ep}', flush=True)

    if not has_reg:
        oof_reg_min = None
    return oof_prob, oof_reg_min


# ═══════════════════════════════════════════════════════════════════════════════
# RUN SPECS
# ═══════════════════════════════════════════════════════════════════════════════

def build_run_specs(quick=False):
    """(a) DualBranchCNN multitask with the lambda sweep (primary);
    (b) EarlyFusionCNN multitask, lambda=1.0 fixed (comparison);
    (c) DualBranchCNN single-task classification-only (comparison).
    Lambda is swept only for (a), kept fixed for (b), unused for (c)."""
    if quick:
        return [dict(arch='dual_branch', mode='multitask', lam=1.0)]
    specs = [dict(arch='dual_branch', mode='multitask', lam=lam) for lam in LAMBDA_SWEEP]
    specs.append(dict(arch='early_fusion', mode='multitask', lam=1.0))
    specs.append(dict(arch='dual_branch', mode='single_task', lam=None))
    return specs


def make_model_factory(arch, mode):
    multitask = (mode == 'multitask')
    if arch == 'dual_branch':
        return lambda: DualBranchCNN(multitask=multitask)
    if arch == 'early_fusion':
        return lambda: EarlyFusionCNN(multitask=multitask)
    raise ValueError(f'Unknown arch: {arch}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = [(['A2'], 'A2')] if args.quick else \
             [(['A2'], 'A2'), (['A3'], 'A3'), (['A2', 'A3'], 'A2+A3')]

metric_rows = []
oof_rows    = []
oof_store   = {}   # (ds_label, arch, mode, lam) -> dict(oof_prob, oof_reg_min, y_cls, y_reg)

for ds_list, ds_label in CONDITIONS:
    print(f'\n{"═"*70}', flush=True)
    print(f'DATASET: {ds_label}', flush=True)
    print(f'{"═"*70}', flush=True)

    X01, y_cls, y_reg, merged = load_dataset(ds_list)
    n, n_fast = len(y_cls), int(y_cls.sum())
    track_ids = merged['track_id'].values
    print(f'  n={n}  fast={n_fast}  slow={n - n_fast}', flush=True)

    for spec in build_run_specs(quick=args.quick):
        arch, mode, lam = spec['arch'], spec['mode'], spec['lam']
        run_lam = lam if mode == 'multitask' else 1.0   # lam unused for single_task loss
        label = f'{arch}/{mode}' + (f'/lambda={lam}' if mode == 'multitask' else '')
        print(f'\n  [{label}]', flush=True)

        model_factory = make_model_factory(arch, mode)
        n_params = sum(p.numel() for p in model_factory().parameters())
        print(f'    trainable params: {n_params}', flush=True)

        save_ckpt = (arch == 'dual_branch' and mode == 'multitask' and lam == 1.0)
        ckpt_dir    = CKPT_DIR if save_ckpt else None
        ckpt_prefix = f'{ds_label.replace("+", "_")}_{arch}_{mode}' if save_ckpt else None

        oof_prob, oof_reg_min = run_cv(
            X01, y_cls, y_reg, model_factory, lam=run_lam,
            max_epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            save_ckpt_dir=ckpt_dir, ckpt_prefix=ckpt_prefix, track_ids=track_ids)

        # ── cls_head metrics row ──
        pred_cls = (oof_prob >= 0.5).astype(int)
        m_cls = compute_cls_metrics(y_cls, oof_prob, pred_cls)
        metric_rows.append(dict(dataset=ds_label, arch=arch, mode=mode,
                                 **{'lambda': lam if mode == 'multitask' else np.nan},
                                 source='cls_head', n=n, n_fast=n_fast, **m_cls,
                                 pearson_r=np.nan, r2=np.nan))
        print(f'    [cls_head] AUC={m_cls["auc"]:.3f}  AP={m_cls["ap"]:.3f}  '
              f'BalAcc={m_cls["bal_acc"]:.3f}  MCC={m_cls["mcc"]:.3f}', flush=True)

        # ── per-cell OOF rows ──
        oof_reg_col = oof_reg_min if oof_reg_min is not None else np.full(n, np.nan)
        for i in range(n):
            oof_rows.append(dict(dataset=ds_label, arch=arch, mode=mode,
                                  **{'lambda': lam if mode == 'multitask' else np.nan},
                                  track_id=int(track_ids[i]), y_cls=int(y_cls[i]),
                                  oof_prob=float(oof_prob[i]), y_reg=float(y_reg[i]),
                                  oof_reg_min=float(oof_reg_col[i])))

        # ── reg_head_thresholded cross-check row (multitask only) ──
        if oof_reg_min is not None:
            pred_reg = (oof_reg_min <= CUT_B2R).astype(int)
            m_reg = compute_cls_metrics(y_cls, -oof_reg_min, pred_reg)
            pr, _ = pearsonr(oof_reg_min, y_reg)
            r2v = r2_score(y_reg, oof_reg_min)
            metric_rows.append(dict(dataset=ds_label, arch=arch, mode=mode,
                                     **{'lambda': lam}, source='reg_head_thresholded',
                                     n=n, n_fast=n_fast, **m_reg,
                                     pearson_r=round(float(pr), 3), r2=round(float(r2v), 3)))
            print(f'    [reg_head_thresholded] AUC={m_reg["auc"]:.3f}  '
                  f'pearson_r={pr:.3f}  R2={r2v:.3f}', flush=True)

        oof_store[(ds_label, arch, mode, lam)] = dict(
            oof_prob=oof_prob, oof_reg_min=oof_reg_min, y_cls=y_cls, y_reg=y_reg)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

metrics_cols = ['dataset', 'arch', 'mode', 'lambda', 'source', 'n', 'n_fast',
                'auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc', 'pearson_r', 'r2']
metrics_df = pd.DataFrame(metric_rows)[metrics_cols]
metrics_df.to_csv(RESULTS_DIR / 'raw_cnn_multitask_metrics.csv', index=False)
print(f'\nSaved results/raw_cnn_multitask_metrics.csv', flush=True)
print(metrics_df.to_string(index=False), flush=True)

oof_cols = ['dataset', 'arch', 'mode', 'lambda', 'track_id', 'y_cls', 'oof_prob',
            'y_reg', 'oof_reg_min']
oof_df = pd.DataFrame(oof_rows)[oof_cols]
oof_df.to_csv(RESULTS_DIR / 'raw_cnn_multitask_oof.csv', index=False)
print('Saved results/raw_cnn_multitask_oof.csv', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

ds_labels_all = ['A2', 'A3', 'A2+A3']
ds_labels = [d for d in ds_labels_all if any(k[0] == d for k in oof_store)]

RUN_CONFIGS = [
    ('dual_branch', 'multitask', 0.3),
    ('dual_branch', 'multitask', 1.0),
    ('dual_branch', 'multitask', 3.0),
    ('early_fusion', 'multitask', 1.0),
    ('dual_branch', 'single_task', None),
]
RUN_CONFIGS = [c for c in RUN_CONFIGS if any(k[1:] == c for k in oof_store)]

RUN_COLOURS = {
    ('dual_branch', 'multitask', 0.3):  '#90CAF9',
    ('dual_branch', 'multitask', 1.0):  '#1E88E5',
    ('dual_branch', 'multitask', 3.0):  '#0D47A1',
    ('early_fusion', 'multitask', 1.0): '#FF9800',
    ('dual_branch', 'single_task', None): '#9E9E9E',
}
RUN_LABELS = {
    ('dual_branch', 'multitask', 0.3):  'dual_branch λ=0.3',
    ('dual_branch', 'multitask', 1.0):  'dual_branch λ=1.0',
    ('dual_branch', 'multitask', 3.0):  'dual_branch λ=3.0',
    ('early_fusion', 'multitask', 1.0): 'early_fusion λ=1.0',
    ('dual_branch', 'single_task', None): 'dual_branch single-task',
}

# ── AUC bar chart (cls_head, grouped by dataset condition) ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ds_labels))
n_cfg = max(len(RUN_CONFIGS), 1)
width = 0.7 / n_cfg
offsets = np.linspace(-(n_cfg - 1) / 2 * width, (n_cfg - 1) / 2 * width, n_cfg)

for offset, cfg in zip(offsets, RUN_CONFIGS):
    aucs = []
    for ds in ds_labels:
        key = (ds,) + cfg
        row = metrics_df[(metrics_df['dataset'] == ds) & (metrics_df['arch'] == cfg[0]) &
                          (metrics_df['mode'] == cfg[1]) & (metrics_df['source'] == 'cls_head') &
                          (metrics_df['lambda'].isna() if cfg[2] is None else metrics_df['lambda'] == cfg[2])]
        aucs.append(row['auc'].values[0] if len(row) else float('nan'))
    bars = ax.bar(x + offset, aucs, width * 0.9, label=RUN_LABELS[cfg],
                  color=RUN_COLOURS[cfg], edgecolor='white', linewidth=0.5)
    for bar, a in zip(bars, aucs):
        if not np.isnan(a):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f'{a:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(ds_labels, fontsize=11)
ax.set_ylabel('AUC (5-fold OOF, cls head)', fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.set_title('Raw-pixel from-scratch CNN — b2r classification (cls head)\n'
             f'half-movie filter  |  cut={CUT_B2R} min', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'raw_cnn_auc_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/raw_cnn_auc_bars.png', flush=True)

# ── ROC curves, 1 subplot per dataset condition ────────────────────────────────
fig, axes = plt.subplots(1, len(ds_labels), figsize=(5 * len(ds_labels), 4.5), squeeze=False)
axes = axes[0]
for ax, ds in zip(axes, ds_labels):
    for cfg in RUN_CONFIGS:
        key = (ds,) + cfg
        if key not in oof_store:
            continue
        d = oof_store[key]
        if len(np.unique(d['y_cls'])) < 2:
            continue
        fpr, tpr, _ = roc_curve(d['y_cls'], d['oof_prob'])
        auc = roc_auc_score(d['y_cls'], d['oof_prob'])
        ax.plot(fpr, tpr, lw=1.6, color=RUN_COLOURS[cfg], label=f'{RUN_LABELS[cfg]} ({auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_title(ds, fontsize=11)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=6.5, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('ROC — raw-pixel from-scratch CNN (5-fold OOF, cls head)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'raw_cnn_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/raw_cnn_roc.png', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON (honest context per H0)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 78, flush=True)
print('BASELINE COMPARISON (per stated H0 -- diagnostic value, not a new best AUC)', flush=True)
print('=' * 78, flush=True)
print('  [Reference] LogReg GFP+BF fusion (A2, Cellpose embeddings)  AUC = 0.712', flush=True)
print('  [Reference] BF embedding, onset-10 (strongest single predictor) AUC = 0.876', flush=True)

mt_cls = metrics_df[(metrics_df['source'] == 'cls_head') & (metrics_df['mode'] == 'multitask')]
for ds in ds_labels:
    sub = mt_cls[mt_cls['dataset'] == ds]
    if len(sub):
        best = sub.loc[sub['auc'].idxmax()]
        print(f'  This script  ({ds:>5}, best multitask config: {best["arch"]} '
              f'λ={best["lambda"]}): AUC = {best["auc"]:.3f}', flush=True)

print('\nDone.', flush=True)

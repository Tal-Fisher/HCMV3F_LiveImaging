#!/usr/bin/env python3
"""
10_new_arch.py

New FC architectures for b2r classification on GFP+BF (512-dim) fused embeddings.
Runs only on A2+A3 (pooled) to keep run time short.

New architectures (512-dim input):
  slim_32   : FC(512→32) → ReLU → Drop(0.5) → FC(32→1)
  wide_64_8 : FC(512→64) → ReLU → Drop(0.4) → FC(64→8) → ReLU → Drop(0.3) → FC(8→1)

Baselines carried from 09_slim_net.py (all rescaled to 512-dim input):
  baseline   : FC(512→64) → ReLU → Drop(0.4) → FC(64→20) → ReLU → Drop(0.3) → FC(20→1)
  slim_16     : FC(512→16) → ReLU → Drop(0.5) → FC(16→1)
  slim_8      : FC(512→8)  → ReLU → Drop(0.5) → FC(8→1)
  slim_16_d06 : FC(512→16) → ReLU → Drop(0.6) → FC(16→1)
  slim_idrop  : Drop(0.1) → FC(512→16) → ReLU → Drop(0.5) → FC(16→1)

Outputs
-------
  results/new_arch_metrics.csv
  results/new_arch_oof.csv
  figures/new_arch_roc.png
  figures/new_arch_auc_bars.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
torch.manual_seed(42)

BASE     = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')
CPE      = LIVEIMG / 'CellposeEmbedding' / 'embeddings'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

GFP_FILES = {
    'A2': CPE / 'A2_cell_embeddings.npz',
    'A3': CPE / 'A3_cell_embeddings.npz',
}
BF_FILES = {
    'A2': BASE / 'embeddings' / 'A2_bf_at_gfp_onset.npz',
    'A3': BASE / 'embeddings' / 'A3_bf_at_gfp_onset.npz',
}

CUT_B2R  = 1094
SEED     = 42
N_SPLITS = 5

mdf    = pd.read_csv(MODEL_DF)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(ds_list):
    gfp_records, bf_records = [], []
    gfp_emb_list, bf_emb_list = [], []
    gfp_off = bf_off = 0

    for ds in ds_list:
        d_gfp = np.load(str(GFP_FILES[ds]))
        d_bf  = np.load(str(BF_FILES[ds]))
        ng = len(d_gfp['track_ids'])
        nb = len(d_bf['track_ids'])
        for i, tid in enumerate(d_gfp['track_ids']):
            gfp_records.append({'dataset': ds, 'track_id': int(tid), 'gfp_row': gfp_off + i})
        for i, tid in enumerate(d_bf['track_ids']):
            bf_records.append({'dataset': ds, 'track_id': int(tid), 'bf_row': bf_off + i})
        gfp_emb_list.append(d_gfp['embeddings'].astype(np.float32))
        bf_emb_list.append(d_bf['embeddings'].astype(np.float32))
        gfp_off += ng
        bf_off  += nb

    GFP_EMB = np.vstack(gfp_emb_list)
    BF_EMB  = np.vstack(bf_emb_list)
    gfp_df  = pd.DataFrame(gfp_records)
    bf_df   = pd.DataFrame(bf_records)

    meta_chunks = []
    for ds in ds_list:
        sub = mdf[mdf['dataset'] == ds].copy()
        sub['track_id'] = sub['Track.ID'].str.replace(f'{ds}_', '', regex=False).astype(int)
        sub['b2r']      = sub['delay_green_to_red'] - sub['delay_green_to_blue']
        meta_chunks.append(sub)
    meta = pd.concat(meta_chunks).reset_index(drop=True)

    merged = (meta
              .merge(gfp_df, on=['dataset', 'track_id'], how='inner')
              .merge(bf_df,  on=['dataset', 'track_id'], how='inner'))
    merged = merged[merged['b2r'].notna()]
    if 'abs_gfp_onset_min' in merged.columns and 'movie_half_min' in merged.columns:
        merged = merged[merged['abs_gfp_onset_min'] <= merged['movie_half_min']]
    merged = merged.sort_values(['dataset', 'track_id']).reset_index(drop=True)

    X_GFP = GFP_EMB[merged['gfp_row'].values]
    X_BF  = BF_EMB[merged['bf_row'].values]
    y     = (merged['b2r'].values <= CUT_B2R).astype(int)
    return X_GFP, X_BF, y, merged


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y, oof):
    auc  = roc_auc_score(y, oof)
    ap   = average_precision_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    sens = int(((pred == 1) & (y == 1)).sum()) / max(int(y.sum()), 1)
    spec = int(((pred == 0) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
    bal  = balanced_accuracy_score(y, pred)
    mcc  = matthews_corrcoef(y, pred)
    return dict(auc=round(auc, 3), ap=round(ap, 3),
                sens=round(sens, 3), spec=round(spec, 3),
                bal_acc=round(bal, 3), mcc=round(mcc, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class Head20Net(nn.Module):
    """Baseline: in→64→20→1"""
    def __init__(self, in_dim, drop1=0.4, drop2=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(64, 20),     nn.ReLU(), nn.Dropout(drop2),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x)


class SlimNet(nn.Module):
    """Single hidden layer with optional input dropout."""
    def __init__(self, in_dim, hidden=16, drop=0.5, input_drop=0.0):
        super().__init__()
        layers = []
        if input_drop > 0:
            layers.append(nn.Dropout(input_drop))
        layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
                   nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoLayerNet(nn.Module):
    """Two hidden layers: in→h1→h2→1"""
    def __init__(self, in_dim, h1=64, h2=8, drop1=0.4, drop2=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(drop2),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_architectures(in_dim):
    return [
        # --- baselines from 09_slim_net ---
        ('baseline',    lambda: Head20Net(in_dim)),
        ('slim_16',     lambda: SlimNet(in_dim, hidden=16, drop=0.5)),
        ('slim_8',      lambda: SlimNet(in_dim, hidden=8,  drop=0.5)),
        ('slim_16_d06', lambda: SlimNet(in_dim, hidden=16, drop=0.6)),
        ('slim_idrop',  lambda: SlimNet(in_dim, hidden=16, drop=0.5, input_drop=0.1)),
        # --- new architectures ---
        ('slim_32',     lambda: SlimNet(in_dim, hidden=32, drop=0.5)),
        ('wide_64_8',   lambda: TwoLayerNet(in_dim, h1=64, h2=8, drop1=0.4, drop2=0.3)),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(model, X_tr, y_tr, X_val, y_val,
               max_epochs=300, patience=30, batch_size=32):
    n_slow = int((y_tr == 0).sum())
    n_fast = int(y_tr.sum())
    pos_w  = n_slow / max(n_fast, 1)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    ds_tr  = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    Xv_t   = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_auc, best_state, no_improve, stopped = -1.0, None, 0, max_epochs
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb).squeeze(1), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xv_t).squeeze(1)).cpu().numpy()
        val_auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped = epoch + 1
                break

    model.load_state_dict(best_state)
    return model, best_auc, stopped


def run_cv(X, y, arch_label, model_factory):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X[tr])
        X_te_s = sc.transform(X[te])

        model = model_factory().to(device)
        model, best_auc, n_ep = train_fold(
            model, X_tr_s, y[tr].astype(np.float32), X_te_s, y[te])

        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(
                model(torch.tensor(X_te_s, dtype=torch.float32).to(device)).squeeze(1)
            ).cpu().numpy()

        fold_auc = roc_auc_score(y[te], oof[te]) if len(np.unique(y[te])) > 1 else 0.5
        print(f'      fold {fold+1}: OOF={fold_auc:.3f}  best_val={best_auc:.3f}  ep={n_ep}',
              flush=True)

    m = compute_metrics(y, oof)
    print(f'    [{arch_label}] AUC={m["auc"]:.3f}  AP={m["ap"]:.3f}  '
          f'BalAcc={m["bal_acc"]:.3f}', flush=True)
    return oof, m


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — A2+A3, GFP+BF (512) only
# ═══════════════════════════════════════════════════════════════════════════════

print('\nLoading A2+A3 dataset...', flush=True)
X_GFP, X_BF, y, eligible = load_dataset(['A2', 'A3'])
X_in = np.concatenate([X_GFP, X_BF], axis=1)   # 512-dim
n, n_fast, n_slow = len(y), int(y.sum()), int((y == 0).sum())
print(f'  n={n}  fast={n_fast}  slow={n_slow}  in_dim={X_in.shape[1]}', flush=True)

arch_list   = make_architectures(X_in.shape[1])
all_records = []
oof_store   = {}

for arch_label, model_factory_fn in arch_list:
    print(f'\n  [{arch_label}]', flush=True)
    oof, m = run_cv(X_in, y, arch_label, model_factory_fn)
    m.update(arch=arch_label, n=n, n_fast=n_fast)
    all_records.append(m)
    oof_store[arch_label] = (oof, y.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

cols    = ['arch', 'n', 'n_fast', 'auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc']
results = pd.DataFrame(all_records)[cols].sort_values('auc', ascending=False)
results.to_csv(RESULTS_DIR / 'new_arch_metrics.csv', index=False)
print(f'\nSaved results/new_arch_metrics.csv')
print(results.to_string(index=False))

oof_rows = []
for arch, (oof, y_arr) in oof_store.items():
    for i, (o, yi) in enumerate(zip(oof, y_arr)):
        oof_rows.append({'arch': arch, 'idx': i, 'oof_prob': o, 'y': yi})
pd.DataFrame(oof_rows).to_csv(RESULTS_DIR / 'new_arch_oof.csv', index=False)
print('Saved results/new_arch_oof.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

NEW_ARCHS = {'slim_32', 'wide_64_8'}

arch_colours = {
    'baseline':    '#9E9E9E',
    'slim_16':     '#2196F3',
    'slim_8':      '#4CAF50',
    'slim_16_d06': '#FF9800',
    'slim_idrop':  '#E91E63',
    'slim_32':     '#9C27B0',
    'wide_64_8':   '#F44336',
}

arch_display = {
    'baseline':    'baseline (→64→20→1)',
    'slim_16':     'slim_16 (→16→1)',
    'slim_8':      'slim_8 (→8→1)',
    'slim_16_d06': 'slim_16_d06 (→16→1, d=0.6)',
    'slim_idrop':  'slim_idrop (id→16→1)',
    'slim_32':     'slim_32 (→32→1)  ★',
    'wide_64_8':   'wide_64_8 (→64→8→1)  ★',
}

# --- AUC bar chart ---
arch_order = [r['arch'] for r in all_records]  # original run order
sorted_results = results.copy()

fig, ax = plt.subplots(figsize=(10, 5))
aucs   = [results.loc[results['arch'] == a, 'auc'].values[0] for a in arch_order]
colors = [arch_colours.get(a, '#607D8B') for a in arch_order]
bars   = ax.bar(range(len(arch_order)), aucs, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.6)
for bar, auc, arch in zip(bars, aucs, arch_order):
    weight = 'bold' if arch in NEW_ARCHS else 'normal'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{auc:.3f}', ha='center', va='bottom', fontsize=9, fontweight=weight)

ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6)
ax.set_xticks(range(len(arch_order)))
ax.set_xticklabels([arch_display.get(a, a) for a in arch_order],
                   rotation=25, ha='right', fontsize=9)
ax.set_ylabel('AUC (5-fold OOF)', fontsize=11)
ax.set_title(f'Architecture comparison — GFP+BF (512-dim) — A2+A3  (n={n}, {n_fast} fast)\n'
             f'b2r classification  |  cut=1094 min  |  ★ = new architectures', fontsize=11)
ax.set_ylim(0.4, 0.95)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'new_arch_auc_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/new_arch_auc_bars.png')

# --- ROC curves ---
fig, ax = plt.subplots(figsize=(7, 6))
for arch in arch_order:
    if arch not in oof_store:
        continue
    oof, y_arr = oof_store[arch]
    fpr, tpr, _ = roc_curve(y_arr, oof)
    auc  = roc_auc_score(y_arr, oof)
    lw   = 2.5 if arch in NEW_ARCHS else 1.5
    ls   = '-'  if arch in NEW_ARCHS else '--'
    label = f'{arch_display.get(arch, arch)}  ({auc:.3f})'
    ax.plot(fpr, tpr, color=arch_colours.get(arch, '#607D8B'),
            lw=lw, ls=ls, label=label)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('FPR', fontsize=11)
ax.set_ylabel('TPR', fontsize=11)
ax.set_title(f'ROC — all architectures — GFP+BF 512-dim — A2+A3  (n={n}, {n_fast} fast)\n'
             f'solid = new  |  dashed = baselines  |  5-fold OOF', fontsize=10)
ax.legend(fontsize=7.5, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'new_arch_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/new_arch_roc.png')

print('\nDone.', flush=True)

#!/usr/bin/env python3
"""
08_fullEmb_head20_network.py

FC network starting from the full 256-dim (or 512-dim concat) embedding,
with a 20-node penultimate layer before the output.

Architecture:
  single-stream (in_dim=256):  FC(256→64) → ReLU → Drop(0.4) → FC(64→20) → ReLU → Drop(0.3) → FC(20→1)
  concat (in_dim=512):         FC(512→64) → ReLU → Drop(0.4) → FC(64→20) → ReLU → Drop(0.3) → FC(20→1)

Half-movie filter applied (abs_gfp_onset_min <= movie_half_min).

Outputs
-------
  results/fullEmb_head20_metrics.csv
  results/fullEmb_head20_oof.csv
  figures/fullEmb_head20_roc.png
  figures/fullEmb_head20_auc_bars.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
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

CUT_B2R  = 1094
SEED     = 42
TOP_N    = 20
N_SPLITS = 5

GFP_FILES = {
    'A2': CPE / 'A2_cell_embeddings.npz',
    'A3': CPE / 'A3_cell_embeddings.npz',
}
BF_FILES = {
    'A2': BASE / 'embeddings' / 'A2_bf_at_gfp_onset.npz',
    'A3': BASE / 'embeddings' / 'A3_bf_at_gfp_onset.npz',
}

LR_PARAMS = dict(
    penalty='elasticnet', solver='saga',
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    Cs=np.logspace(-3, 1, 20),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
    class_weight='balanced', scoring='roc_auc',
    max_iter=2000, random_state=SEED, n_jobs=-1,
)

mdf    = pd.read_csv(MODEL_DF)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (with half-movie filter)
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
    y_reg = merged['b2r'].values.astype(float)
    return X_GFP, X_BF, y, y_reg, merged


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
# FC NETWORK — full embedding input, 20-node bottleneck before output
# ═══════════════════════════════════════════════════════════════════════════════

class Head20Net(nn.Module):
    """
    single-stream (256): FC(256→64) → ReLU → Drop(0.4) → FC(64→20) → ReLU → Drop(0.3) → FC(20→1)
    concat     (512): FC(512→64) → ReLU → Drop(0.4) → FC(64→20) → ReLU → Drop(0.3) → FC(20→1)
    """
    def __init__(self, in_dim, drop1=0.4, drop2=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(64, 20),     nn.ReLU(), nn.Dropout(drop2),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_net_fold(X_tr, y_tr, X_val, y_val, in_dim,
                   max_epochs=200, patience=20, batch_size=32):
    n_slow = int((y_tr == 0).sum())
    n_fast = int(y_tr.sum())
    pos_w  = n_slow / max(n_fast, 1)

    model     = Head20Net(in_dim).to(device)
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


def run_cv_net(X, y, label=''):
    in_dim = X.shape[1]
    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof    = np.zeros(len(y))

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X[tr])
        X_te_s = sc.transform(X[te])

        model, best_auc, n_ep = train_net_fold(
            X_tr_s, y[tr].astype(np.float32), X_te_s, y[te], in_dim)

        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(
                model(torch.tensor(X_te_s, dtype=torch.float32).to(device)).squeeze(1)
            ).cpu().numpy()

        fold_auc = roc_auc_score(y[te], oof[te]) if len(np.unique(y[te])) > 1 else 0.5
        print(f'      fold {fold+1}: OOF AUC={fold_auc:.3f}  '
              f'best_val={best_auc:.3f}  epoch={n_ep}', flush=True)

    m = compute_metrics(y, oof)
    if label:
        print(f'    {label}: AUC={m["auc"]:.3f}  AP={m["ap"]:.3f}  '
              f'BalAcc={m["bal_acc"]:.3f}', flush=True)
    return oof, m


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = [(['A2'], 'A2'), (['A3'], 'A3'), (['A2', 'A3'], 'A2+A3')]

all_records = []
oof_store   = {}

for ds_list, ds_label in CONDITIONS:
    print(f'\n{"═"*65}', flush=True)
    print(f'DATASET: {ds_label}', flush=True)

    X_GFP, X_BF, y, y_reg, eligible = load_dataset(ds_list)
    n, n_fast, n_slow = len(y), int(y.sum()), int((y == 0).sum())
    print(f'  n={n}  fast={n_fast}  slow={n_slow}', flush=True)

    X_concat = np.concatenate([X_GFP, X_BF], axis=1)  # (n, 512)

    net_runs = [
        (X_GFP,    'GFP (256)'),
        (X_BF,     'BF (256)'),
        (X_concat, 'GFP+BF (512)'),
    ]

    print('\n  [Head20Net — full embedding → 64 → 20 → 1]', flush=True)
    for X_in, feat in net_runs:
        print(f'\n    {feat}  (in_dim={X_in.shape[1]})...', flush=True)
        oof, m = run_cv_net(X_in, y, label=feat)
        m.update(dataset=ds_label, model_type='Head20Net', features=feat, n=n, n_fast=n_fast)
        all_records.append(m)
        oof_store[(ds_label, feat)] = (oof, y.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

cols    = ['dataset', 'model_type', 'features', 'n', 'n_fast',
           'auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc']
results = pd.DataFrame(all_records)[cols]
results.to_csv(RESULTS_DIR / 'fullEmb_head20_metrics.csv', index=False)
print(f'\nSaved results/fullEmb_head20_metrics.csv')
print(results.to_string(index=False))

oof_rows = []
for (ds_label, feat), (oof, y) in oof_store.items():
    for i, (o, yi) in enumerate(zip(oof, y)):
        oof_rows.append({'dataset': ds_label, 'features': feat, 'idx': i, 'oof_prob': o, 'y': yi})
pd.DataFrame(oof_rows).to_csv(RESULTS_DIR / 'fullEmb_head20_oof.csv', index=False)
print('Saved results/fullEmb_head20_oof.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

ds_labels    = ['A2', 'A3', 'A2+A3']
model_labels = ['GFP (256)', 'BF (256)', 'GFP+BF (512)']
colours      = {'GFP (256)': '#4CAF50', 'BF (256)': '#2196F3', 'GFP+BF (512)': '#FF9800'}

# AUC bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x       = np.arange(len(ds_labels))
width   = 0.25
offsets = [-width, 0, width]
ns      = {ds: results.loc[results['dataset'] == ds, 'n'].iloc[0] for ds in ds_labels}

for offset, feat in zip(offsets, model_labels):
    aucs = [results.loc[(results['dataset'] == ds) & (results['features'] == feat), 'auc'].values[0]
            for ds in ds_labels]
    bars = ax.bar(x + offset, aucs, width * 0.9, label=feat,
                  color=colours[feat], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6, label='Random (AUC=0.500)')
ax.set_xticks(x)
ax.set_xticklabels([f'{ds}\n(n={ns[ds]})' for ds in ds_labels], fontsize=11)
ax.set_ylabel('AUC (5-fold OOF)', fontsize=11)
ax.set_title('Head20Net: full 256-dim → 64 → 20 → 1\nb2r classification  |  half-movie filter  |  cut=1094 min', fontsize=11)
ax.set_ylim(0.4, 0.92)
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'fullEmb_head20_auc_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/fullEmb_head20_auc_bars.png')

# ROC curves per dataset
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, ds in zip(axes, ds_labels):
    for feat, colour in colours.items():
        key = (ds, feat)
        if key not in oof_store:
            continue
        oof, y = oof_store[key]
        fpr, tpr, _ = roc_curve(y, oof)
        auc = roc_auc_score(y, oof)
        ax.plot(fpr, tpr, color=colour, lw=1.5, label=f'{feat} ({auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    n_sub = results.loc[results['dataset'] == ds, 'n'].iloc[0]
    n_f   = results.loc[results['dataset'] == ds, 'n_fast'].iloc[0]
    ax.set_title(f'{ds}  (n={n_sub}, {n_f} fast)', fontsize=11)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('ROC curves — Head20Net full embedding  |  5-fold OOF', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'fullEmb_head20_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/fullEmb_head20_roc.png')

print('\nDone.', flush=True)

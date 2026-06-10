#!/usr/bin/env python3
"""
07_top40_network_hmf.py

FC network on the top-40 concatenated features (top-20 GFP + top-20 BF),
with half-movie filter applied.

Directly comparable to the best GLM result (GFP+BF top-40 LogReg).
Also runs GFP-only top-20 and BF-only top-20 network variants for completeness.

Architecture (2 hidden FC layers, small to match feature count):
  GFP/BF top-20 (in_dim=20): FC(20→16) → ReLU → Drop(0.3) → FC(16→8) → ReLU → Drop(0.2) → FC(8→1)
  GFP+BF top-40 (in_dim=40): FC(40→32) → ReLU → Drop(0.3) → FC(32→16) → ReLU → Drop(0.2) → FC(16→1)

Outputs
-------
  results/top40_net_hmf_metrics.csv
  results/top40_net_hmf_oof.csv
  figures/top40_net_hmf_roc.png
  figures/top40_net_hmf_auc_bars.png
"""

import pickle
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
# FEATURE SELECTION & METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def select_top_dims(X, y_reg, y_cls, top_n=TOP_N, label=''):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                      cv=5, max_iter=10000, n_jobs=-1, random_state=SEED).fit(Xs, y_reg)
    lr = LogisticRegressionCV(**LR_PARAMS).fit(Xs, y_cls)
    rank_r = pd.Series(np.abs(en.coef_),    index=np.arange(X.shape[1])).rank(ascending=False)
    rank_c = pd.Series(np.abs(lr.coef_[0]), index=np.arange(X.shape[1])).rank(ascending=False)
    dims = ((rank_r + rank_c) / 2).sort_values().head(top_n).index.tolist()
    if label:
        print(f'    top-{top_n} {label}: {dims}', flush=True)
    return dims


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
# GLM (for reference, same top-40 features)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv_glm(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR_PARAMS)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof, compute_metrics(y, oof)


# ═══════════════════════════════════════════════════════════════════════════════
# FC NETWORK — sized to match top-k input dims
# ═══════════════════════════════════════════════════════════════════════════════

class TopKNet(nn.Module):
    """Two-hidden-layer FC network sized for small top-k inputs.

    in_dim=20: FC(20→16) → ReLU → Drop(0.3) → FC(16→8) → ReLU → Drop(0.2) → FC(8→1)
    in_dim=40: FC(40→32) → ReLU → Drop(0.3) → FC(32→16) → ReLU → Drop(0.2) → FC(16→1)
    """
    def __init__(self, in_dim, drop1=0.3, drop2=0.2):
        super().__init__()
        if in_dim <= 24:
            h1, h2 = 16, 8
        else:
            h1, h2 = 32, 16
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(h1, h2),    nn.ReLU(), nn.Dropout(drop2),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_net_fold(X_tr, y_tr, X_val, y_val, in_dim,
                   max_epochs=300, patience=30, batch_size=32):
    n_slow = int((y_tr == 0).sum())
    n_fast = int(y_tr.sum())
    pos_w  = n_slow / max(n_fast, 1)

    model     = TopKNet(in_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

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
# MAIN LOOP (with per-dataset checkpointing)
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = [(['A2'], 'A2'), (['A3'], 'A3'), (['A2', 'A3'], 'A2+A3')]

all_records = []
oof_store   = {}


def ckpt_path(ds_label):
    return RESULTS_DIR / f'top40_net_hmf_ckpt_{ds_label.replace("+", "")}.pkl'


def save_ckpt(ds_label, records, store_slice):
    with open(ckpt_path(ds_label), 'wb') as f:
        pickle.dump({'records': records, 'oof_store': store_slice}, f)
    print(f'  [checkpoint saved: {ckpt_path(ds_label).name}]', flush=True)


def load_ckpt(ds_label):
    p = ckpt_path(ds_label)
    if p.exists():
        with open(p, 'rb') as f:
            data = pickle.load(f)
        print(f'  [checkpoint loaded: {p.name} — skipping]', flush=True)
        return data['records'], data['oof_store']
    return None, None


for ds_list, ds_label in CONDITIONS:
    print(f'\n{"═"*65}', flush=True)
    print(f'DATASET: {ds_label}', flush=True)

    ckpt_records, ckpt_store = load_ckpt(ds_label)
    if ckpt_records is not None:
        all_records.extend(ckpt_records)
        oof_store.update(ckpt_store)
        continue

    X_GFP, X_BF, y, y_reg, eligible = load_dataset(ds_list)
    n, n_fast, n_slow = len(y), int(y.sum()), int((y == 0).sum())
    print(f'  n={n}  fast={n_fast}  slow={n_slow}', flush=True)

    # Dims pre-computed from 05_a2a3_combined_hmf.py (job 476322) — skip recomputation
    PRECOMPUTED_DIMS = {
        'A2': {
            'gfp': [204, 198, 212, 127, 60, 148, 64, 66, 26, 218, 5, 205, 77, 85, 163, 223, 239, 241, 254, 247],
            'bf':  [158, 40, 140, 100, 61, 209, 28, 120, 6, 187, 125, 14, 45, 165, 39, 21, 117, 109, 2, 255],
        },
        'A3': {
            'gfp': [148, 237, 24, 202, 218, 4, 66, 38, 168, 8, 10, 9, 3, 5, 6, 2, 15, 16, 18, 17],
            'bf':  [6, 2, 27, 0, 52, 171, 140, 217, 141, 235, 162, 142, 177, 158, 85, 65, 225, 10, 167, 126],
        },
        'A2+A3': {
            'gfp': [204, 5, 237, 18, 85, 168, 95, 148, 66, 118, 77, 78, 127, 64, 6, 59, 241, 0, 94, 217],
            'bf':  [158, 40, 171, 65, 28, 60, 0, 123, 172, 85, 68, 14, 165, 21, 37, 162, 168, 113, 167, 164],
        },
    }
    gfp_top = PRECOMPUTED_DIMS[ds_label]['gfp']
    bf_top  = PRECOMPUTED_DIMS[ds_label]['bf']
    print(f'  Using pre-computed top-20 GFP dims: {gfp_top}', flush=True)
    print(f'  Using pre-computed top-20 BF dims:  {bf_top}', flush=True)

    X_gfp_t = X_GFP[:, gfp_top]                                  # (n, 20)
    X_bf_t  = X_BF[:,  bf_top]                                    # (n, 20)
    X_top40 = np.concatenate([X_gfp_t, X_bf_t], axis=1)          # (n, 40)

    ds_records = []
    ds_store   = {}

    runs = [
        (X_gfp_t, 'GFP (top-20)'),
        (X_bf_t,  'BF (top-20)'),
        (X_top40, 'GFP+BF (top-40)'),
    ]

    # ── TopKNet ──────────────────────────────────────────────────────────────
    print('\n  [TopKNet — FC on top-k dims]', flush=True)
    for X_in, feat in runs:
        print(f'    {feat}...', flush=True)
        oof, m = run_cv_net(X_in, y, label=feat)
        m.update(dataset=ds_label, model_type='TopKNet', features=feat, n=n, n_fast=n_fast)
        ds_records.append(m)
        ds_store[(ds_label, 'TopKNet', feat)] = (oof, y.copy())

    save_ckpt(ds_label, ds_records, ds_store)
    all_records.extend(ds_records)
    oof_store.update(ds_store)

# ── Save results ─────────────────────────────────────────────────────────────
col_order = ['dataset', 'model_type', 'features', 'n', 'n_fast',
             'auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc']
results = pd.DataFrame(all_records)[col_order]
results.to_csv(RESULTS_DIR / 'top40_net_hmf_metrics.csv', index=False)
print(f'\nSaved results/top40_net_hmf_metrics.csv', flush=True)

oof_rows = []
for (ds_label, model_type, feat), (oof, y_arr) in oof_store.items():
    for p, label in zip(oof, y_arr):
        oof_rows.append(dict(dataset=ds_label, model_type=model_type, features=feat,
                             y_true=int(label), oof_prob=round(float(p), 6)))
pd.DataFrame(oof_rows).to_csv(RESULTS_DIR / 'top40_net_hmf_oof.csv', index=False)
print('Saved results/top40_net_hmf_oof.csv', flush=True)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f'\n{"─"*80}', flush=True)
print(f'  {"Dataset":<8}  {"Type":<10}  {"Features":<18}  '
      f'{"n":>5}  {"Fast":>5}  {"AUC":>6}  {"AP":>6}  {"BalAcc":>7}  {"MCC":>6}')
print(f'{"─"*80}', flush=True)
for _, row in results.iterrows():
    print(f'  {row["dataset"]:<8}  {row["model_type"]:<10}  {row["features"]:<18}  '
          f'{row["n"]:>5}  {row["n_fast"]:>5}  {row["auc"]:>6.3f}  '
          f'{row["ap"]:>6.3f}  {row["bal_acc"]:>7.3f}  {row["mcc"]:>6.3f}')
print(f'{"─"*80}', flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

DS_LABELS  = ['A2', 'A3', 'A2+A3']
FEAT_KEYS  = ['GFP (top-20)', 'BF (top-20)', 'GFP+BF (top-40)']
COLOURS    = {'GFP (top-20)': '#4CAF50', 'BF (top-20)': '#2196F3', 'GFP+BF (top-40)': '#FF9800'}

# Load GLM results from the existing HMF metrics CSV for comparison
glm_ref = pd.read_csv(RESULTS_DIR / 'a2a3_combined_hmf_metrics.csv')
glm_ref = glm_ref[glm_ref['model_type'] == 'GLM']

# ── ROC curves: 3 panels (one per dataset), TopKNet only ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for col_i, ds_label in enumerate(DS_LABELS):
    ax = axes[col_i]
    for feat in FEAT_KEYS:
        oof, y_arr = oof_store[(ds_label, 'TopKNet', feat)]
        auc = roc_auc_score(y_arr, oof)
        fpr, tpr, _ = roc_curve(y_arr, oof)
        lbl = feat.split(' ')[0]
        ax.plot(fpr, tpr, lw=2, color=COLOURS[feat], label=f'{lbl}  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ref_y = oof_store[(ds_label, 'TopKNet', FEAT_KEYS[0])][1]
    ax.set_title(f'{ds_label} (n={len(ref_y)}, {int(ref_y.sum())} fast)', fontsize=10)
    ax.set_xlabel('1 − Specificity', fontsize=9)
    if col_i == 0:
        ax.set_ylabel('Sensitivity', fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(f'TopKNet ROC — FC network on top-k dims  |  cut={CUT_B2R} min  |  '
             f'5-fold OOF  |  half-movie filter',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'top40_net_hmf_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/top40_net_hmf_roc.png', flush=True)

# ── AUC bar chart: GLM vs TopKNet side by side ────────────────────────────────
x           = np.arange(len(DS_LABELS))
width       = 0.13
feat_offsets = np.array([-2, -1, 0, 1, 2, 3]) * width  # 3 feats × 2 models

fig, ax = plt.subplots(figsize=(12, 5))
bar_specs = [
    ('GLM',     'GFP (top-20)',    '-',  0.85),
    ('GLM',     'BF (top-20)',     '-',  0.85),
    ('GLM',     'GFP+BF (top-40)','-',  0.85),
    ('TopKNet', 'GFP (top-20)',    '/', 0.65),
    ('TopKNet', 'BF (top-20)',     '/', 0.65),
    ('TopKNet', 'GFP+BF (top-40)','/', 0.65),
]
for gi, (model_type, feat, hatch, alpha) in enumerate(bar_specs):
    if model_type == 'GLM':
        aucs = [glm_ref.loc[(glm_ref['dataset'] == ds) &
                            (glm_ref['features'] == feat), 'auc'].values[0]
                for ds in DS_LABELS]
    else:
        aucs = [results.loc[(results['dataset'] == ds) &
                            (results['features'] == feat), 'auc'].values[0]
                for ds in DS_LABELS]
    colour = COLOURS[feat]
    lbl = f'{"GLM" if model_type=="GLM" else "Net"} {feat.split(" ")[0]}'
    bars = ax.bar(x + feat_offsets[gi], aucs, width * 0.92,
                  label=lbl, color=colour, alpha=alpha,
                  hatch=hatch if model_type == 'TopKNet' else '',
                  edgecolor='white', linewidth=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color='#333333')

ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6)
ax.set_xticks(x + width * 0.5)
ns = {ds: int(glm_ref.loc[glm_ref['dataset'] == ds, 'n'].iloc[0]) for ds in DS_LABELS}
ax.set_xticklabels([f'{ds}\n(n={ns[ds]})' for ds in DS_LABELS], fontsize=10)
ax.set_ylabel('AUC (5-fold OOF)', fontsize=10)
ax.set_title(f'GLM (solid) vs TopKNet (hatched) on top-k dims  |  cut={CUT_B2R} min  |  HMF',
             fontsize=10, fontweight='bold')
ax.set_ylim(0.40, 0.95)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'top40_net_hmf_auc_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/top40_net_hmf_auc_bars.png', flush=True)

print('\nDone.', flush=True)

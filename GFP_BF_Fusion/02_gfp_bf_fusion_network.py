#!/usr/bin/env python3
"""
02_gfp_bf_fusion_network.py

Combine GFP and BF-at-GFP-onset Cellpose embeddings to predict fast vs slow
(delay_blue_to_red <= 1094 min) using a dual-stream FC network.

Requires: 01_extract_bf_at_gfp_coords.py must have been run first.

Sections
--------
  0. Load & merge data + cell statistics
  1. Option A — LogReg on concat top-20 GFP + top-20 BF (40-dim, baseline)
  2. Ablation — GFP alone / BF alone (each top-20, same cells)
  3. Option B — DualStreamNet 256+256  (PyTorch, 5-fold stratified CV)
  4. Summary: ROC comparison figure + CSV

Architecture (Option B)
-----------------------
  GFP emb (256) -> Linear(256->64) + ReLU + Dropout(0.4)  -|
                                                             +-> concat(128)
  BF  emb (256) -> Linear(256->64) + ReLU + Dropout(0.4)  -|
                       -> Linear(128->32) + ReLU + Dropout(0.3) -> Linear(32->1)
  Loss: BCEWithLogitsLoss(pos_weight=n_slow/n_fast)
  Opt:  Adam lr=1e-3, weight_decay=1e-3
  CV:   5-fold stratified, early stopping patience=20 on val AUC

Outputs
-------
  results/fusion_cell_stats.csv
  results/fusion_logreg_oof.csv        (OOF probabilities, all methods)
  results/fusion_network_oof.csv
  results/fusion_summary.csv
  figures/fusion_roc_comparison.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

GFP_NPZ    = LIVEIMG / 'CellposeEmbedding' / 'embeddings' / 'A2_cell_embeddings.npz'
BF_NPZ     = BASE / 'embeddings' / 'A2_bf_at_gfp_onset.npz'
MODEL_DF   = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_B2R  = 1094   # minutes — GMM Bayes-optimal cutoff (fast = early)
SEED     = 42
TOP_N    = 20     # dims per embedding for Option A baseline
N_SPLITS = 5

# ── Check inputs ───────────────────────────────────────────────────────────────
for p in [GFP_NPZ, BF_NPZ, MODEL_DF]:
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — LOAD & MERGE
# ═══════════════════════════════════════════════════════════════════════════════
print('═'*60, flush=True)
print('SECTION 0 — LOAD & MERGE', flush=True)
print('═'*60, flush=True)

# GFP embeddings
d_gfp = np.load(str(GFP_NPZ))
gfp_ids  = d_gfp['track_ids'].astype(int)
gfp_embs = d_gfp['embeddings'].astype(np.float32)
gfp_map  = {tid: i for i, tid in enumerate(gfp_ids)}
print(f'  GFP embeddings : {gfp_embs.shape}  ({len(gfp_map)} cells)')

# BF-at-GFP-onset embeddings
d_bf = np.load(str(BF_NPZ))
bf_ids  = d_bf['track_ids'].astype(int)
bf_embs = d_bf['embeddings'].astype(np.float32)
bf_map  = {tid: i for i, tid in enumerate(bf_ids)}
print(f'  BF  embeddings : {bf_embs.shape}  ({len(bf_map)} cells)')

# Labels and filters from model_df
df = pd.read_csv(MODEL_DF)
a2 = df[df['dataset'] == 'A2'].copy()
a2['track_id']           = a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
a2['delay_blue_to_red']  = a2['delay_green_to_red'] - a2['delay_green_to_blue']

# Filters
prod_mask = np.isfinite(a2['delay_green_to_red']) & np.isfinite(a2['delay_green_to_blue'])
a2_prod   = a2[prod_mask].copy()

# Half-movie filter (if columns available)
if 'abs_gfp_onset_min' in a2_prod.columns and 'movie_half_min' in a2_prod.columns:
    hm_mask  = a2_prod['abs_gfp_onset_min'] <= a2_prod['movie_half_min']
    a2_prod  = a2_prod[hm_mask].copy()
    print(f'  After half-movie filter: {len(a2_prod)} cells')
else:
    print('  WARNING: half-movie filter columns not found — skipping filter')

# Inner join: must have GFP + BF + finite b2r
eligible = a2_prod[
    a2_prod['track_id'].isin(gfp_map) &
    a2_prod['track_id'].isin(bf_map)  &
    np.isfinite(a2_prod['delay_blue_to_red'])
].sort_values('track_id').reset_index(drop=True)

n = len(eligible)
if n < 50:
    raise RuntimeError(f'Only {n} cells after all filters — check inputs.')

gfp_rows = [gfp_map[tid] for tid in eligible['track_id']]
bf_rows  = [bf_map[tid]  for tid in eligible['track_id']]

X_gfp = gfp_embs[gfp_rows]   # (n, 256)
X_bf  = bf_embs[bf_rows]      # (n, 256)

y_b2r = eligible['delay_blue_to_red'].values
y_cls = (y_b2r <= CUT_B2R).astype(int)   # 1 = fast/early, 0 = slow/med+late

n_fast = int(y_cls.sum())
n_slow = int((y_cls == 0).sum())

print(f'\n  Cells used       : {n}')
print(f'  Fast (b2r<=1094) : {n_fast}  ({100*n_fast/n:.1f}%)')
print(f'  Slow (b2r>1094)  : {n_slow}  ({100*n_slow/n:.1f}%)')
print(f'  b2r mean={y_b2r.mean():.0f}  median={np.median(y_b2r):.0f}  std={y_b2r.std():.0f} min')

pd.DataFrame([dict(
    n=n, n_fast=n_fast, n_slow=n_slow,
    pct_fast=round(100*n_fast/n, 1),
    b2r_mean=round(y_b2r.mean(), 1),
    b2r_median=round(float(np.median(y_b2r)), 1),
    b2r_std=round(y_b2r.std(), 1),
    cut_b2r=CUT_B2R,
)]).to_csv(RESULTS_DIR / 'fusion_cell_stats.csv', index=False)

# ── Helper: select top-N dims via global ElasticNet rank ──────────────────────
def select_top_dims(X, y_reg, y_cls, top_n=TOP_N, label=''):
    """Global top-N dim selection by average |beta| rank across regression + classify."""
    en_params = dict(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                     cv=5, max_iter=10000, n_jobs=-1, random_state=SEED)
    lr_params = dict(penalty='elasticnet', solver='saga',
                     l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                     Cs=np.logspace(-3, 1, 20),
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED+1),
                     class_weight='balanced', scoring='roc_auc',
                     max_iter=2000, random_state=SEED, n_jobs=-1)
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    en  = ElasticNetCV(**en_params).fit(Xs, y_reg)
    lr  = LogisticRegressionCV(**lr_params).fit(Xs, y_cls)
    rank_r = pd.Series(np.abs(en.coef_),   index=np.arange(X.shape[1])).rank(ascending=False)
    rank_c = pd.Series(np.abs(lr.coef_[0]),index=np.arange(X.shape[1])).rank(ascending=False)
    avg_rank = (rank_r + rank_c) / 2
    dims = avg_rank.sort_values().head(top_n).index.tolist()
    if label:
        print(f'    {label} top-{top_n} dims: {dims}', flush=True)
    return dims

# ── Helper: 5-fold stratified LogReg CV ──────────────────────────────────────
def cv_logreg(X, y, label=''):
    lr_params = dict(penalty='elasticnet', solver='saga',
                     l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                     Cs=np.logspace(-3, 1, 20),
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED+1),
                     class_weight='balanced', scoring='roc_auc',
                     max_iter=2000, random_state=SEED, n_jobs=-1)
    skf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**lr_params)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    auc  = roc_auc_score(y, oof)
    pred = (oof >= 0.5).astype(int)
    sens = ((pred==1)&(y==1)).sum() / max(y.sum(), 1)
    spec = ((pred==0)&(y==0)).sum() / max((y==0).sum(), 1)
    bal  = balanced_accuracy_score(y, pred)
    if label:
        print(f'    {label}: AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}',
              flush=True)
    return oof, auc, sens, spec, bal

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OPTION A: LOGREG ON TOP-20 + TOP-20 (40-DIM CONCAT BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 1 — OPTION A: LogReg BASELINES (top-20 per embedding)', flush=True)
print('═'*60, flush=True)

print('\n  Selecting top-20 GFP dims...', flush=True)
gfp_top20 = select_top_dims(X_gfp, y_b2r, y_cls, label='GFP')

print('\n  Selecting top-20 BF dims...', flush=True)
bf_top20 = select_top_dims(X_bf, y_b2r, y_cls, label='BF')

X_gfp_t20 = X_gfp[:, gfp_top20]
X_bf_t20  = X_bf[:, bf_top20]
X_concat  = np.concatenate([X_gfp_t20, X_bf_t20], axis=1)   # (n, 40)

print('\n  GFP alone (top-20):', flush=True)
oof_gfp, auc_gfp, sens_gfp, spec_gfp, bal_gfp = cv_logreg(X_gfp_t20, y_cls, '5-fold CV')

print('\n  BF alone (top-20, at GFP onset):', flush=True)
oof_bf, auc_bf, sens_bf, spec_bf, bal_bf = cv_logreg(X_bf_t20, y_cls, '5-fold CV')

print('\n  GFP + BF concat (top-20 each → 40-dim):', flush=True)
oof_concat, auc_concat, sens_concat, spec_concat, bal_concat = cv_logreg(X_concat, y_cls, '5-fold CV')

logreg_oof = pd.DataFrame({
    'track_id': eligible['track_id'].values,
    'y_true':   y_cls,
    'oof_gfp':    oof_gfp,
    'oof_bf':     oof_bf,
    'oof_concat': oof_concat,
})
logreg_oof.to_csv(RESULTS_DIR / 'fusion_logreg_oof.csv', index=False)
print('\n  Saved fusion_logreg_oof.csv', flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — OPTION B: DUALSTREAMNET (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 2 — OPTION B: DualStreamNet (5-fold stratified CV)', flush=True)
print('═'*60, flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  Device: {device}', flush=True)


class DualStreamNet(nn.Module):
    def __init__(self, drop1=0.4, drop2=0.3):
        super().__init__()
        self.gfp_branch = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(drop1)
        )
        self.bf_branch = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(drop1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(drop2),
            nn.Linear(32, 1)
        )

    def forward(self, gfp, bf):
        return self.fusion(torch.cat([self.gfp_branch(gfp), self.bf_branch(bf)], dim=1))


def train_fold(X_gfp_tr, X_bf_tr, y_tr,
               X_gfp_val, X_bf_val, y_val,
               pos_weight, max_epochs=200, patience=20, batch_size=32):
    model = DualStreamNet().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    ds_tr = TensorDataset(
        torch.tensor(X_gfp_tr, dtype=torch.float32),
        torch.tensor(X_bf_tr,  dtype=torch.float32),
        torch.tensor(y_tr,     dtype=torch.float32),
    )
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    gfp_val_t = torch.tensor(X_gfp_val, dtype=torch.float32).to(device)
    bf_val_t  = torch.tensor(X_bf_val,  dtype=torch.float32).to(device)

    best_auc   = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for g_b, b_b, y_b in loader:
            g_b, b_b, y_b = g_b.to(device), b_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(g_b, b_b).squeeze(1), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(gfp_val_t, bf_val_t).squeeze(1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

        if len(np.unique(y_val)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(y_val, probs)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc, epoch + 1


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_net = np.zeros(n)
pos_w   = n_slow / n_fast

print(f'  pos_weight = {pos_w:.2f}  ({n_slow} slow / {n_fast} fast)', flush=True)

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_gfp, y_cls)):
    # Per-fold normalisation (no leakage)
    sc_gfp = StandardScaler()
    sc_bf  = StandardScaler()

    Xg_tr = sc_gfp.fit_transform(X_gfp[tr_idx])
    Xb_tr = sc_bf.fit_transform(X_bf[tr_idx])
    Xg_te = sc_gfp.transform(X_gfp[te_idx])
    Xb_te = sc_bf.transform(X_bf[te_idx])

    model_fold, best_auc, n_epochs = train_fold(
        Xg_tr, Xb_tr, y_cls[tr_idx].astype(np.float32),
        Xg_te, Xb_te, y_cls[te_idx],
        pos_weight=pos_w,
    )

    model_fold.eval()
    with torch.no_grad():
        logits = model_fold(
            torch.tensor(Xg_te, dtype=torch.float32).to(device),
            torch.tensor(Xb_te, dtype=torch.float32).to(device),
        ).squeeze(1).cpu().numpy()
    oof_net[te_idx] = 1 / (1 + np.exp(-logits))

    fold_auc = roc_auc_score(y_cls[te_idx], oof_net[te_idx])
    print(f'  Fold {fold+1}/{N_SPLITS}: OOF AUC={fold_auc:.3f}  '
          f'best_val_AUC={best_auc:.3f}  stopped@epoch={n_epochs}', flush=True)

auc_net  = roc_auc_score(y_cls, oof_net)
pred_net = (oof_net >= 0.5).astype(int)
sens_net = ((pred_net==1)&(y_cls==1)).sum() / max(y_cls.sum(), 1)
spec_net = ((pred_net==0)&(y_cls==0)).sum() / max((y_cls==0).sum(), 1)
bal_net  = balanced_accuracy_score(y_cls, pred_net)
print(f'\n  DualStreamNet OOF: AUC={auc_net:.3f}  Sens={sens_net:.3f}  '
      f'Spec={spec_net:.3f}  BalAcc={bal_net:.3f}', flush=True)

net_oof = pd.DataFrame({
    'track_id': eligible['track_id'].values,
    'y_true':   y_cls,
    'oof_net':  oof_net,
})
net_oof.to_csv(RESULTS_DIR / 'fusion_network_oof.csv', index=False)
print('  Saved fusion_network_oof.csv', flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60, flush=True)
print('SECTION 3 — SUMMARY', flush=True)
print('═'*60, flush=True)

summary = pd.DataFrame([
    dict(model='GFP_alone_top20',     n=n, n_fast=n_fast,
         auc=round(auc_gfp,3),    sens=round(sens_gfp,3),    spec=round(spec_gfp,3),    bal_acc=round(bal_gfp,3)),
    dict(model='BF_at_onset_top20',   n=n, n_fast=n_fast,
         auc=round(auc_bf,3),     sens=round(sens_bf,3),     spec=round(spec_bf,3),     bal_acc=round(bal_bf,3)),
    dict(model='Concat_top40_LogReg', n=n, n_fast=n_fast,
         auc=round(auc_concat,3), sens=round(sens_concat,3), spec=round(spec_concat,3), bal_acc=round(bal_concat,3)),
    dict(model='DualStreamNet_256x2', n=n, n_fast=n_fast,
         auc=round(auc_net,3),    sens=round(sens_net,3),    spec=round(spec_net,3),    bal_acc=round(bal_net,3)),
])
summary.to_csv(RESULTS_DIR / 'fusion_summary.csv', index=False)

print(f'\n  n={n}  |  {n_fast} fast / {n_slow} slow  |  cut={CUT_B2R} min')
print(f'  {"Model":<28} {"AUC":>7} {"Sens":>7} {"Spec":>7} {"BalAcc":>8}')
for _, row in summary.iterrows():
    print(f'  {row["model"]:<28} {row["auc"]:>7.3f} {row["sens"]:>7.3f} '
          f'{row["spec"]:>7.3f} {row["bal_acc"]:>8.3f}')

# Reference baselines (from prior analyses)
print(f'\n  [Reference] BF_m10_top20 (onset-10)  : AUC=0.876  (BrightFieldEmbedding/07)')
print(f'  [Reference] GFP_emb_top20  (onset)   : AUC=0.742  (CellposeEmbedding/08)')
print(f'  [Reference] Tabular_45feat            : AUC~0.685')

# ── ROC comparison figure ──────────────────────────────────────────────────────
colours = {'GFP alone':       ('#4CAF50', oof_gfp,    auc_gfp),
           'BF at onset':     ('#2196F3', oof_bf,     auc_bf),
           'Concat top-40':   ('#FF9800', oof_concat, auc_concat),
           'DualStreamNet':   ('#9C27B0', oof_net,    auc_net)}

fig, ax = plt.subplots(figsize=(6, 6))
for label, (colour, oof, auc) in colours.items():
    fpr, tpr, _ = roc_curve(y_cls, oof)
    ax.plot(fpr, tpr, lw=2, color=colour, label=f'{label}  AUC={auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random  AUC=0.500')
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title(
    f'GFP + BF Fusion — b2r Classification\n'
    f'n={n}, {n_fast} fast, cut={CUT_B2R} min  (5-fold OOF)',
    fontsize=10
)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
png_path = FIGURES_DIR / 'fusion_roc_comparison.png'
fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  Saved {png_path}', flush=True)

print('\nDone.', flush=True)

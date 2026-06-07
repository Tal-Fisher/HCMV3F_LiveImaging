#!/usr/bin/env python3
"""
Three-way elastic net classification: early vs medium+late (A2 cells only).
Uses LogisticRegressionCV (elasticnet, saga) — the sklearn equivalent of glmnet.

Models:
  A) Tabular only   (29 features)
  B) Embedding only (PCA 50 of Cellpose neck embeddings)
  C) Combined       (tabular + embedding PCA, independently scaled per fold)

Reference from script 13 (A2+A3): AUC=0.684  Sens=0.597  Spec=0.710

Output:
  results/classify_cv_metrics.csv
  figures/classify_roc.png
  figures/classify_probas.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ  = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CUT_EARLY  = 911
PCA_COMPS  = 50
N_FOLDS    = 5
SEED       = 42

LR_CV = dict(
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    Cs=np.logspace(-3, 1, 20),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1),
    class_weight='balanced',
    scoring='roc_auc',
    max_iter=2000,
    random_state=SEED,
    n_jobs=-1,
)

# ── Load data ──────────────────────────────────────────────────────────────
print('Loading embeddings...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float64)

print('Loading tabular features...', flush=True)
df = pd.read_csv(MODEL_DF)
df = df[df['abs_gfp_onset_min'] <= df['movie_half_min']].reset_index(drop=True)

NON_FEAT = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
            'green_onset_min', 'track_start_min', 'abs_gfp_onset_min',
            'movie_half_min', 'y', 'gfp_snr_mean', 'bf_snr_mean'}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_a2 = df_a2[np.isfinite(df_a2['delay_green_to_red'])].copy()

emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}
df_a2 = df_a2[df_a2['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
emb_rows = [emb_id_to_row[tid] for tid in df_a2['track_id']]

X_emb_raw = embeddings[emb_rows]
X_tab_raw = df_a2[feat_cols].values.astype(float)
delay     = df_a2['delay_green_to_red'].values
y         = (delay <= CUT_EARLY).astype(int)

print(f'  A2 cells: {len(y)}  |  early: {y.sum()}  |  med+late: {(y==0).sum()}')
print(f'  Tabular features: {len(feat_cols)}')

# ── Nested CV ──────────────────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds    = list(outer_cv.split(X_tab_raw, y))

def run_nested_cv(X_raw, label, pca_components=None, impute=False):
    print(f'\n── {label} ──', flush=True)
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(folds):
        Xtr, Xte = X_raw[tr].copy(), X_raw[te].copy()
        if impute:
            imp = SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(Xtr)
            Xte = imp.transform(Xte)
        if pca_components is not None:
            pca = PCA(n_components=pca_components, random_state=SEED)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        m = LogisticRegressionCV(**LR_CV)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]

        if fold == 0:
            best_C = m.C_[0]; best_l1 = m.l1_ratio_[0]
            print(f'  fold 1: AUC={roc_auc_score(y[te], oof[te]):.3f}  '
                  f'C={best_C:.4f}  l1_ratio={best_l1:.2f}', flush=True)

    auc     = roc_auc_score(y, oof)
    pred    = (oof >= 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fn = int(((pred==0)&(y==1)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fp = int(((pred==1)&(y==0)).sum())
    sens    = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = balanced_accuracy_score(y, pred)
    print(f'  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal_acc:.3f}')
    return oof, dict(model=label, n=len(y), n_early=int(y.sum()),
                     auc=round(auc,3), sens=round(sens,3),
                     spec=round(spec,3), bal_acc=round(bal_acc,3))

oof_A, met_A = run_nested_cv(X_tab_raw, 'Tabular only', impute=True)
oof_B, met_B = run_nested_cv(X_emb_raw, 'Embedding only', pca_components=PCA_COMPS)

# Combined: scale each block per fold independently
print('\n── Combined ──', flush=True)
oof_C = np.zeros(len(y))
for fold, (tr, te) in enumerate(folds):
    imp = SimpleImputer(strategy='median')
    Xtab_tr = imp.fit_transform(X_tab_raw[tr])
    Xtab_te = imp.transform(X_tab_raw[te])

    pca = PCA(n_components=PCA_COMPS, random_state=SEED)
    Xemb_tr = pca.fit_transform(X_emb_raw[tr])
    Xemb_te = pca.transform(X_emb_raw[te])

    sc_tab = StandardScaler()
    Xtab_tr = sc_tab.fit_transform(Xtab_tr); Xtab_te = sc_tab.transform(Xtab_te)

    sc_emb = StandardScaler()
    Xemb_tr = sc_emb.fit_transform(Xemb_tr); Xemb_te = sc_emb.transform(Xemb_te)

    Xtr = np.hstack([Xtab_tr, Xemb_tr])
    Xte = np.hstack([Xtab_te, Xemb_te])

    m = LogisticRegressionCV(**LR_CV)
    m.fit(Xtr, y[tr])
    oof_C[te] = m.predict_proba(Xte)[:, 1]
    if fold == 0:
        print(f'  fold 1: AUC={roc_auc_score(y[te], oof_C[te]):.3f}  '
              f'C={m.C_[0]:.4f}  l1_ratio={m.l1_ratio_[0]:.2f}', flush=True)

auc_C   = roc_auc_score(y, oof_C)
pred_C  = (oof_C >= 0.5).astype(int)
tp = int(((pred_C==1)&(y==1)).sum()); fn = int(((pred_C==0)&(y==1)).sum())
tn = int(((pred_C==0)&(y==0)).sum()); fp = int(((pred_C==1)&(y==0)).sum())
sens_C  = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_C  = tn / (tn + fp) if (tn + fp) > 0 else 0
bal_C   = balanced_accuracy_score(y, pred_C)
print(f'  AUC={auc_C:.3f}  Sens={sens_C:.3f}  Spec={spec_C:.3f}  BalAcc={bal_C:.3f}')
met_C = dict(model='Combined', n=len(y), n_early=int(y.sum()),
             auc=round(auc_C,3), sens=round(sens_C,3),
             spec=round(spec_C,3), bal_acc=round(bal_C,3))

# ── Save & report ──────────────────────────────────────────────────────────
print('\n── Summary ──')
metrics_df = pd.DataFrame([met_A, met_B, met_C])
metrics_df['ref_auc_A2plusA3'] = 0.684
print(metrics_df[['model','n','n_early','auc','sens','spec','bal_acc']].to_string(index=False))
metrics_df.to_csv(str(RESULTS_DIR / 'classify_cv_metrics.csv'), index=False)
print(f'Saved: {RESULTS_DIR}/classify_cv_metrics.csv')

# ── ROC curves ─────────────────────────────────────────────────────────────
colours = {'Tabular only': '#2196F3', 'Embedding only': '#FF9800', 'Combined': '#4CAF50'}
fig, ax = plt.subplots(figsize=(5.5, 5))
for label, oof, m in [('Tabular only', oof_A, met_A),
                      ('Embedding only', oof_B, met_B),
                      ('Combined', oof_C, met_C)]:
    fpr, tpr, _ = roc_curve(y, oof)
    ax.plot(fpr, tpr, lw=2, color=colours[label],
            label=f'{label}  AUC={m["auc"]:.3f}')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random  AUC=0.500')
ax.axhline(0.684, color='gray', lw=1, ls=':', alpha=0.7,
           label='Ref (A2+A3 tabular)  AUC=0.684')
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'Early vs Med+Late — A2  (n={len(y)}, {y.sum()} early)\n'
             f'5-fold nested CV, LogisticRegressionCV, class_weight=balanced')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'classify_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {FIGURES_DIR}/classify_roc.png')

# ── Probability distributions ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (label, oof, m) in zip(axes, [('Tabular only', oof_A, met_A),
                                       ('Embedding only', oof_B, met_B),
                                       ('Combined', oof_C, met_C)]):
    bins = np.linspace(0, 1, 25)
    ax.hist(oof[y==0], bins=bins, alpha=0.6, color='#607D8B', label='Med+Late', density=True)
    ax.hist(oof[y==1], bins=bins, alpha=0.6, color='#E91E63', label='Early',    density=True)
    ax.axvline(0.5, color='black', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('P(early)'); ax.set_ylabel('Density')
    ax.set_title(f'{label}\nAUC={m["auc"]:.3f}  Sens={m["sens"]:.3f}  Spec={m["spec"]:.3f}')
    ax.legend(fontsize=8)
fig.suptitle('OOF predicted probabilities by true class', fontsize=11)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'classify_probas.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {FIGURES_DIR}/classify_probas.png')

print('\nDone.', flush=True)

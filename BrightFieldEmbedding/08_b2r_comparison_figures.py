#!/usr/bin/env python3
"""
08_b2r_comparison_figures.py

Comparison figures + confusion matrices for the b2r analysis.
Re-runs all CVs to get OOF predictions (fast, CPU only).

Models compared:
  GFP embedding top-20   (n=291)
  BF  embedding top-20   (n=151, same cells as tabular)
  BF  tabular 45 feat    (n=151, same cells as BF emb)

Outputs
-------
  BrightFieldEmbedding/figures/b2r_roc_all.png           — 3 ROC curves
  BrightFieldEmbedding/figures/b2r_confusion_all.png     — 3×2 confusion matrices
  BrightFieldEmbedding/figures/b2r_summary_bars.png      — metric bar chart
  BrightFieldEmbedding/figures/b2r_scatter_all.png       — 3 regression scatter panels
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error,
                             roc_auc_score, roc_curve,
                             confusion_matrix, balanced_accuracy_score)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_GFP = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
BASE_BF  = Path('/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding')
LIVEIMG  = Path('/home/labs/ginossar/talfis/LiveImaging')

GFP_EMB_NPZ  = BASE_GFP / 'embeddings' / 'A2_cell_embeddings.npz'
BF_EMB_NPZ   = BASE_BF  / 'embeddings' / 'A2_bf_embeddings_m10_relaxed.npz'
MODEL_DF_CSV = LIVEIMG  / 'cache' / 'python_export' / 'model_df.csv'
EXT_DF_CSV   = LIVEIMG  / 'results' / 'elasticnet_extended2' / 'model_df_extended2.csv'
TOP20_BF_CSV = BASE_BF  / 'results' / 'b2r_top20_dims.csv'
FIG_DIR      = BASE_BF  / 'figures'
FIG_DIR.mkdir(exist_ok=True)

GFP_TOP20 = [212, 148, 127, 204, 237, 200, 241, 198, 77, 60, 66, 247,
             11, 163, 223, 78, 205, 44, 190, 249]

CUT_B2R = 1094
SEED    = 42

META_COLS = {'Track.ID', 'dataset', 'delay_green_to_red', 'delay_green_to_blue',
             'abs_gfp_onset_min', 'movie_half_min'}
EXTRAS_18 = {'cell_aspect_start', 'cell_aspect_mean', 'bfp_nuc_frac_start',
             'nuc_ratio_start', 'nuc_ratio_end',
             'bf_ctrst_start', 'bf_ctrst_end', 'bf_ctrst_slope'}

EN_PARAMS = dict(l1_ratio=[0.5, 0.9, 1.0], alphas=np.logspace(-2, 3, 20),
                 cv=5, max_iter=10000, n_jobs=-1, random_state=SEED)
LR_OUTER  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
LR_INNER  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
LR_PARAMS = dict(penalty='elasticnet', solver='saga',
                 l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                 Cs=np.logspace(-3, 1, 20), cv=LR_INNER,
                 class_weight='balanced', scoring='roc_auc',
                 max_iter=2000, random_state=SEED, n_jobs=-1)

# ── CV helpers ─────────────────────────────────────────────────────────────────
def cv_regression_oof(X, y):
    kf  = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        en = ElasticNetCV(**EN_PARAMS)
        en.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = en.predict(sc.transform(X[te]))
    return oof

def cv_classify_oof(X, y):
    oof = np.zeros(len(y))
    for tr, te in LR_OUTER.split(X, y):
        sc = StandardScaler()
        m  = LogisticRegressionCV(**LR_PARAMS)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof

def youden_threshold(y_true, y_score):
    """Threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])

# ── Load and prepare datasets ──────────────────────────────────────────────────
print('Loading data...', flush=True)

# GFP
d_gfp = np.load(str(GFP_EMB_NPZ))
gfp_emb   = d_gfp['embeddings'].astype(np.float64)
gfp_ids   = {int(t): i for i, t in enumerate(d_gfp['track_ids'])}
mdf = pd.read_csv(MODEL_DF_CSV)
mdf = mdf[mdf['dataset'] == 'A2'].copy()
mdf['track_id'] = mdf['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
mdf['b2r'] = mdf['delay_green_to_red'] - mdf['delay_green_to_blue']
gfp_df = mdf[mdf['track_id'].isin(gfp_ids) & np.isfinite(mdf['b2r'])
             ].sort_values('track_id').reset_index(drop=True)
X_gfp  = gfp_emb[[gfp_ids[t] for t in gfp_df['track_id']]][:, GFP_TOP20]
y_gfp_r = gfp_df['b2r'].values
y_gfp_c = (y_gfp_r <= CUT_B2R).astype(int)
print(f'  GFP: n={len(y_gfp_r)}  early={y_gfp_c.sum()}  med+late={(y_gfp_c==0).sum()}')

# BF embeddings + tabular
d_bf   = np.load(str(BF_EMB_NPZ))
bf_emb = d_bf['embeddings'].astype(np.float64)
bf_ids = {int(t): i for i, t in enumerate(d_bf['gfp_track_ids'])}
ext    = pd.read_csv(EXT_DF_CSV)
tab_cols = [c for c in ext.columns if c not in META_COLS and c not in EXTRAS_18]
ext = ext[ext['dataset'] == 'A2'].copy()
ext['track_id'] = ext['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
ext['b2r'] = ext['delay_green_to_red'] - ext['delay_green_to_blue']
bf_df  = ext[ext['track_id'].isin(bf_ids) & np.isfinite(ext['b2r'])
             ].sort_values('track_id').reset_index(drop=True)
top20_bf = pd.read_csv(TOP20_BF_CSV)['dim'].tolist()
X_bf   = bf_emb[[bf_ids[t] for t in bf_df['track_id']]][:, top20_bf]
X_tab_raw = bf_df[tab_cols].values.astype(float)
col_med   = np.nanmedian(X_tab_raw, axis=0)
for j in range(X_tab_raw.shape[1]):
    bad = ~np.isfinite(X_tab_raw[:, j])
    X_tab_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
X_tab  = X_tab_raw
y_bf_r = bf_df['b2r'].values
y_bf_c = (y_bf_r <= CUT_B2R).astype(int)
print(f'  BF:  n={len(y_bf_r)}  early={y_bf_c.sum()}  med+late={(y_bf_c==0).sum()}')

# ── Run all CVs ────────────────────────────────────────────────────────────────
print('Running CVs...', flush=True)

print('  GFP embedding top-20 regression...', flush=True)
oof_gfp_r = cv_regression_oof(X_gfp, y_gfp_r)
print('  GFP embedding top-20 classification...', flush=True)
oof_gfp_c = cv_classify_oof(X_gfp, y_gfp_c)

print('  BF embedding top-20 regression...', flush=True)
oof_bf_r  = cv_regression_oof(X_bf, y_bf_r)
print('  BF embedding top-20 classification...', flush=True)
oof_bf_c  = cv_classify_oof(X_bf, y_bf_c)

print('  Tabular 45 feat regression...', flush=True)
oof_tab_r = cv_regression_oof(X_tab, y_bf_r)
print('  Tabular 45 feat classification...', flush=True)
oof_tab_c = cv_classify_oof(X_tab, y_bf_c)

print('  CVs done.', flush=True)

# Compute metrics
def reg_metrics(y, oof):
    r2  = r2_score(y, oof)
    r   = pearsonr(y, oof)[0]
    mae = mean_absolute_error(y, oof)
    return r2, r, mae

def cls_metrics(y, oof, thr=0.5):
    auc  = roc_auc_score(y, oof)
    pred = (oof >= thr).astype(int)
    bal  = balanced_accuracy_score(y, pred)
    cm   = confusion_matrix(y, pred)
    tp = cm[1,1]; fn = cm[1,0]; fp = cm[0,1]; tn = cm[0,0]
    sens = tp/(tp+fn) if tp+fn>0 else 0
    spec = tn/(tn+fp) if tn+fp>0 else 0
    return auc, bal, sens, spec, cm

def cls_at_youden(y, oof):
    thr = youden_threshold(y, oof)
    return cls_metrics(y, oof, thr=thr), thr

r2_gfp, r_gfp, mae_gfp = reg_metrics(y_gfp_r, oof_gfp_r)
r2_bf,  r_bf,  mae_bf  = reg_metrics(y_bf_r,  oof_bf_r)
r2_tab, r_tab, mae_tab = reg_metrics(y_bf_r,  oof_tab_r)

(auc_gfp, bal_gfp, sens_gfp, spec_gfp, cm_gfp), thr_gfp = cls_at_youden(y_gfp_c, oof_gfp_c)
(auc_bf,  bal_bf,  sens_bf,  spec_bf,  cm_bf),  thr_bf  = cls_at_youden(y_bf_c,  oof_bf_c)
(auc_tab, bal_tab, sens_tab, spec_tab, cm_tab), thr_tab = cls_at_youden(y_bf_c,  oof_tab_c)

print(f'\n  GFP emb: r={r_gfp:.3f}  AUC={auc_gfp:.3f}  Sens={sens_gfp:.3f}  Spec={spec_gfp:.3f}  thr={thr_gfp:.3f}')
print(f'  BF  emb: r={r_bf:.3f}   AUC={auc_bf:.3f}  Sens={sens_bf:.3f}  Spec={spec_bf:.3f}  thr={thr_bf:.3f}')
print(f'  Tabular: r={r_tab:.3f}  AUC={auc_tab:.3f}  Sens={sens_tab:.3f}  Spec={spec_tab:.3f}  thr={thr_tab:.3f}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ROC curves
# ══════════════════════════════════════════════════════════════════════════════
print('\nFigure 1: ROC comparison...', flush=True)

fig, ax = plt.subplots(figsize=(6, 5.5))
colour_gfp = '#2196F3'
colour_bf  = '#9C27B0'
colour_tab = '#FF5722'

for y, oof, auc, lbl, col in [
    (y_gfp_c, oof_gfp_c, auc_gfp, f'GFP emb top-20  (n={len(y_gfp_c)}, {y_gfp_c.sum()} early)', colour_gfp),
    (y_bf_c,  oof_bf_c,  auc_bf,  f'BF emb top-20   (n={len(y_bf_c)}, {y_bf_c.sum()} early)', colour_bf),
    (y_bf_c,  oof_tab_c, auc_tab, f'Tabular 45 feat  (n={len(y_bf_c)}, {y_bf_c.sum()} early)', colour_tab),
]:
    fpr, tpr, _ = roc_curve(y, oof)
    ax.plot(fpr, tpr, lw=2, color=col, label=f'{lbl}\nAUC={auc:.3f}')

ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random  AUC=0.500')
ax.set_xlabel('1 − Specificity (FPR)', fontsize=11)
ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
ax.set_title(f'ROC: early vs med+late (b2r ≤ {CUT_B2R} min)\n5-fold CV OOF · Youden threshold applied to confusion matrices',
             fontsize=9)
ax.legend(fontsize=7.5, loc='lower right')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
p = FIG_DIR / 'b2r_roc_all.png'
fig.savefig(str(p), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {p.name}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Confusion matrices (Youden threshold, 3 columns × 2 rows)
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 2: Confusion matrices...', flush=True)

MODELS = [
    ('GFP emb\ntop-20', cm_gfp, auc_gfp, sens_gfp, spec_gfp, thr_gfp,
     len(y_gfp_c), y_gfp_c.sum(), colour_gfp),
    ('BF emb\ntop-20',  cm_bf,  auc_bf,  sens_bf,  spec_bf,  thr_bf,
     len(y_bf_c), y_bf_c.sum(), colour_bf),
    ('Tabular\n45 feat', cm_tab, auc_tab, sens_tab, spec_tab, thr_tab,
     len(y_bf_c), y_bf_c.sum(), colour_tab),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (name, cm, auc, sens, spec, thr, n, n_early, col) in zip(axes, MODELS):
    # Row-normalise to percentages for the colour scale
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = 100 * cm / np.where(row_sums > 0, row_sums, 1)
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred\nMed+Late', 'Pred\nEarly'], fontsize=9)
    ax.set_yticklabels(['True\nMed+Late', 'True\nEarly'], fontsize=9)
    for i in range(2):
        for j in range(2):
            pct = cm_pct[i, j]
            ax.text(j, i, f'{pct:.0f}%\n(n={cm[i,j]})',
                    ha='center', va='center', fontsize=10,
                    color='white' if pct > 50 else 'black',
                    fontweight='bold')
    ax.set_title(
        f'{name}\n'
        f'AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}\n'
        f'thr={thr:.2f}  n={n}  ({n_early} early)',
        fontsize=8.5, color=col
    )
    ax.set_xlabel('Predicted label', fontsize=8)
    ax.set_ylabel('True label', fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('% of true class', fontsize=7)

fig.suptitle(f'Confusion matrices at Youden\'s J threshold — b2r ≤ {CUT_B2R} min (early)\nColour = row % (per true class)',
             fontsize=11)
plt.tight_layout()
p = FIG_DIR / 'b2r_confusion_all.png'
fig.savefig(str(p), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {p.name}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Summary bar chart (regression r + AUC)
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 3: Summary bars...', flush=True)

labels   = ['GFP emb\ntop-20\n(n=291)', 'BF emb\ntop-20\n(n=151)', 'Tabular\n45 feat\n(n=151)']
r_vals   = [r_gfp,   r_bf,   r_tab]
auc_vals = [auc_gfp, auc_bf, auc_tab]
colours  = [colour_gfp, colour_bf, colour_tab]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for ax, vals, ylabel, title, ref in [
    (axes[0], r_vals,   'Pearson r (5-fold CV OOF)', 'Regression: delay_blue_to_red', 0.0),
    (axes[1], auc_vals, 'AUC (5-fold CV OOF)',        'Classification: early vs med+late', 0.5),
]:
    bars = ax.bar(labels, vals, color=colours, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.axhline(ref, color='gray', lw=1, ls='--', alpha=0.6,
               label='Random' if ref == 0.5 else 'No correlation')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005 if val >= 0 else val - 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    if ref == 0.5:
        ax.set_ylim([0, min(1.05, max(auc_vals)+0.1)])
    else:
        ymin = min(0, min(r_vals)-0.05)
        ax.set_ylim([ymin, max(r_vals)+0.07])

fig.suptitle('B2R analysis: Cellpose embeddings vs hand-crafted tabular features\n'
             '5-fold nested CV  ·  no half-movie filter  ·  b2r classification cut=1094 min',
             fontsize=10)
plt.tight_layout()
p = FIG_DIR / 'b2r_summary_bars.png'
fig.savefig(str(p), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {p.name}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Regression scatter (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 4: Regression scatter...', flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tertile_gfp = np.array(pd.qcut(y_gfp_r, q=3, labels=['early','medium','late']).astype(str))
tertile_bf  = np.array(pd.qcut(y_bf_r,  q=3, labels=['early','medium','late']).astype(str))
t_cols = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}

for ax, y, oof, r2, r, lbl, tertile, col in [
    (axes[0], y_gfp_r, oof_gfp_r, r2_gfp, r_gfp,
     f'GFP emb top-20\n(n={len(y_gfp_r)})', tertile_gfp, colour_gfp),
    (axes[1], y_bf_r,  oof_bf_r,  r2_bf,  r_bf,
     f'BF emb top-20\n(n={len(y_bf_r)})',  tertile_bf,  colour_bf),
    (axes[2], y_bf_r,  oof_tab_r, r2_tab, r_tab,
     f'Tabular 45 feat\n(n={len(y_bf_r)})', tertile_bf, colour_tab),
]:
    lims = [y.min()-150, y.max()+150]
    for t in ['early','medium','late']:
        m = tertile == t
        ax.scatter(y[m], oof[m], c=t_cols[t], label=t, alpha=0.65, s=18, edgecolors='none')
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual delay_blue_to_red (min)', fontsize=9)
    ax.set_ylabel('Predicted (min)', fontsize=9)
    ax.set_title(f'{lbl}\nr={r:.3f}  R²={r2:.3f}', fontsize=9, color=col)
    ax.legend(title='Tertile', fontsize=7, loc='upper left')

fig.suptitle('delay_blue_to_red regression — 5-fold CV OOF predictions', fontsize=11)
plt.tight_layout()
p = FIG_DIR / 'b2r_scatter_all.png'
fig.savefig(str(p), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {p.name}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Print summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*70, flush=True)
print('SUMMARY', flush=True)
print('═'*70, flush=True)
print(f'  {"Model":<28} {"n":>5} {"early":>6} {"Regr r":>8} {"Regr R²":>8} '
      f'{"AUC":>7} {"Sens":>7} {"Spec":>7}', flush=True)
for name, n, npos, r, r2, auc, sens, spec in [
    ('GFP emb top-20', len(y_gfp_r), y_gfp_c.sum(), r_gfp, r2_gfp, auc_gfp, sens_gfp, spec_gfp),
    ('BF emb top-20',  len(y_bf_r),  y_bf_c.sum(),  r_bf,  r2_bf,  auc_bf,  sens_bf,  spec_bf),
    ('Tabular 45 feat', len(y_bf_r), y_bf_c.sum(),  r_tab, r2_tab, auc_tab, sens_tab, spec_tab),
]:
    print(f'  {name:<28} {n:>5} {npos:>6} {r:>8.3f} {r2:>8.3f} '
          f'{auc:>7.3f} {sens:>7.3f} {spec:>7.3f}', flush=True)
print('\nDone.', flush=True)

#!/usr/bin/env python3
"""
XGBoost regression (delay_blue_to_red) + classification (early vs med+late)
using the top-20 embedding dims.

Evaluation: nested 5-fold CV (outer OOF, inner RandomizedSearch for hyperparams).
Compares to ElasticNet top-20 results (r=0.297, AUC=0.742).

Output:
  results/top20_xgb_metrics.csv
  figures/top20_xgb_regression_importance.png
  figures/top20_xgb_classify_importance.png
  figures/top20_xgb_scatter_regression.png
  figures/top20_xgb_roc.png
  figures/embedding_comparison_figure2.png   (updated summary)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (r2_score, mean_absolute_error,
                             roc_auc_score, roc_curve, balanced_accuracy_score)
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings('ignore')

BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ     = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF    = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'

TOP20_DIMS = [212, 148, 127, 204, 237, 200, 241, 198, 77, 60, 66, 247, 11, 163, 223, 78, 205, 44, 190, 249]
FEAT_NAMES = [f'dim_{d}' for d in TOP20_DIMS]
CUT_EARLY  = 911
SEED       = 42

# ── Load data ──────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float64)
X_top = embeddings[:, TOP20_DIMS]
print(f'  Feature matrix: {X_top.shape}')

df = pd.read_csv(MODEL_DF)
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}

# Regression dataset
df_reg = df[df['dataset']=='A2'].copy()
df_reg['track_id'] = df_reg['Track.ID'].str.replace('A2_','',regex=False).astype(int)
df_reg = df_reg[np.isfinite(df_reg['delay_green_to_red']) &
                np.isfinite(df_reg['delay_green_to_blue'])].copy()
df_reg = df_reg[df_reg['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_reg = [emb_id_to_row[tid] for tid in df_reg['track_id']]
X_reg = X_top[rows_reg]
y_reg = (df_reg['delay_green_to_red'] - df_reg['delay_green_to_blue']).values
print(f'  Regression: n={len(y_reg)}  mean={y_reg.mean():.0f}  std={y_reg.std():.0f} min')

# Classification dataset
df_cls = df[df['abs_gfp_onset_min'] <= df['movie_half_min']].reset_index(drop=True)
df_cls = df_cls[df_cls['dataset']=='A2'].copy()
df_cls['track_id'] = df_cls['Track.ID'].str.replace('A2_','',regex=False).astype(int)
df_cls = df_cls[np.isfinite(df_cls['delay_green_to_red'])].copy()
df_cls = df_cls[df_cls['track_id'].isin(emb_id_to_row)].sort_values('track_id').reset_index(drop=True)
rows_cls = [emb_id_to_row[tid] for tid in df_cls['track_id']]
X_cls = X_top[rows_cls]
y_cls = (df_cls['delay_green_to_red'].values <= CUT_EARLY).astype(int)
print(f'  Classification: n={len(y_cls)}  ({y_cls.sum()} early / {(y_cls==0).sum()} med+late)')

# ── Hyperparameter grid (shared structure, task-specific estimator) ─────────
XGB_GRID = {
    'n_estimators':  [100, 200, 400],
    'max_depth':     [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample':     [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma':         [0, 0.1, 0.3],
}

# ── Feature importance bar plot ────────────────────────────────────────────
def plot_importance(importances, title, path):
    order  = np.argsort(importances)[::-1]
    vals   = importances[order]
    labels = [FEAT_NAMES[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(vals)), vals, color='#1565C0', height=0.7)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean gain importance')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# REGRESSION
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ REGRESSION (XGBoost, top-20 dims) ═══', flush=True)

outer_kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_kf   = KFold(n_splits=5, shuffle=True, random_state=SEED+1)
oof_reg    = np.zeros(len(y_reg))
best_params_r = []

for fold, (tr, te) in enumerate(outer_kf.split(X_reg)):
    xgb_r = XGBRegressor(tree_method='hist', random_state=SEED,
                          eval_metric='rmse', verbosity=0)
    search = RandomizedSearchCV(xgb_r, XGB_GRID, n_iter=40, cv=inner_kf,
                                scoring='neg_mean_absolute_error',
                                random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X_reg[tr], y_reg[tr])
    oof_reg[te] = search.best_estimator_.predict(X_reg[te])
    best_params_r.append(search.best_params_)
    print(f'  Fold {fold+1}: best_params={search.best_params_}', flush=True)

cv_r2  = r2_score(y_reg, oof_reg)
cv_r   = pearsonr(y_reg, oof_reg)[0]
cv_mae = mean_absolute_error(y_reg, oof_reg)
print(f'  5-fold CV:  R²={cv_r2:.3f}  r={cv_r:.3f}  MAE={cv_mae:.1f} min')
print(f'  ElasticNet top-20 reference: r=0.297')

# full-data model for importances
xgb_rf = XGBRegressor(tree_method='hist', random_state=SEED, verbosity=0,
                       **{k: pd.Series([p[k] for p in best_params_r]).mode()[0]
                          for k in XGB_GRID})
xgb_rf.fit(X_reg, y_reg)
imp_r = xgb_rf.feature_importances_   # gain by default

plot_importance(imp_r, f'XGBoost regression importance\ndel_blue_to_red  CV r={cv_r:.3f}',
                FIGURES_DIR/'top20_xgb_regression_importance.png')

# scatter
tertile = np.array(pd.qcut(y_reg, q=3, labels=['early','medium','late']).astype(str))
colours = {'early':'#2196F3','medium':'#FF9800','late':'#F44336'}
fig, ax = plt.subplots(figsize=(5, 5))
for t in ['early','medium','late']:
    m = tertile == t
    ax.scatter(y_reg[m], oof_reg[m], c=colours[t], label=t, alpha=0.7, s=25, edgecolors='none')
lims = [y_reg.min()-100, y_reg.max()+100]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual delay_blue_to_red (min)')
ax.set_ylabel('Predicted (min)')
ax.set_title(f'XGBoost — delay_blue_to_red (top-20)\nCV R²={cv_r2:.3f}  r={cv_r:.3f}  (n={len(y_reg)})')
ax.legend(title='Tertile', fontsize=8)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR/'top20_xgb_scatter_regression.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved figures/top20_xgb_regression_importance.png + top20_xgb_scatter_regression.png')

# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ CLASSIFICATION (XGBoost, top-20 dims) ═══', flush=True)

scale_pos = float((y_cls == 0).sum()) / float(y_cls.sum())   # ~5.5
print(f'  scale_pos_weight = {scale_pos:.2f}')

outer_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+1)
oof_cls    = np.zeros(len(y_cls))
best_params_c = []

for fold, (tr, te) in enumerate(outer_skf.split(X_cls, y_cls)):
    xgb_c = XGBClassifier(tree_method='hist', random_state=SEED,
                           scale_pos_weight=scale_pos,
                           eval_metric='auc', use_label_encoder=False, verbosity=0)
    search = RandomizedSearchCV(xgb_c, XGB_GRID, n_iter=40, cv=inner_skf,
                                scoring='roc_auc',
                                random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X_cls[tr], y_cls[tr])
    oof_cls[te] = search.best_estimator_.predict_proba(X_cls[te])[:, 1]
    best_params_c.append(search.best_params_)
    print(f'  Fold {fold+1}: best_params={search.best_params_}', flush=True)

auc  = roc_auc_score(y_cls, oof_cls)
pred = (oof_cls >= 0.5).astype(int)
tp = int(((pred==1)&(y_cls==1)).sum()); fn = int(((pred==0)&(y_cls==1)).sum())
tn = int(((pred==0)&(y_cls==0)).sum()); fp = int(((pred==1)&(y_cls==0)).sum())
sens = tp/(tp+fn) if (tp+fn)>0 else 0
spec = tn/(tn+fp) if (tn+fp)>0 else 0
bal  = balanced_accuracy_score(y_cls, pred)
print(f'  5-fold CV:  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}')
print(f'  ElasticNet top-20 reference: AUC=0.742')

# full-data model for importances
xgb_cf = XGBClassifier(tree_method='hist', random_state=SEED, verbosity=0,
                        scale_pos_weight=scale_pos,
                        **{k: pd.Series([p[k] for p in best_params_c]).mode()[0]
                           for k in XGB_GRID})
xgb_cf.fit(X_cls, y_cls)
imp_c = xgb_cf.feature_importances_

plot_importance(imp_c, f'XGBoost classification importance\nearly vs med+late  CV AUC={auc:.3f}',
                FIGURES_DIR/'top20_xgb_classify_importance.png')

# ROC
fpr, tpr, _ = roc_curve(y_cls, oof_cls)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, color='#1565C0', label=f'XGBoost top-20  AUC={auc:.3f}')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random AUC=0.500')
ax.axhline(0.742, color='#42A5F5', lw=1, ls=':', label='ElasticNet top-20 AUC=0.742')
ax.axhline(0.674, color='#EF6C00', lw=1, ls=':', label='Tabular AUC=0.674')
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'Early vs Med+Late — XGBoost top-20\n(n={len(y_cls)}, {y_cls.sum()} early)')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR/'top20_xgb_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved figures/top20_xgb_classify_importance.png + top20_xgb_roc.png')

# ── Save metrics ───────────────────────────────────────────────────────────
pd.DataFrame([
    dict(task='regression_blue_to_red', model='XGBoost', n_dims=20, n=len(y_reg),
         cv_r2=round(cv_r2,4), cv_r=round(cv_r,4), cv_mae=round(cv_mae,1),
         ref_elasticnet_r=0.297, ref_tabular_r=0.359),
    dict(task='classify_early_vs_rest', model='XGBoost', n_dims=20, n=len(y_cls),
         n_early=int(y_cls.sum()), auc=round(auc,3),
         sens=round(sens,3), spec=round(spec,3), bal_acc=round(bal,3),
         ref_elasticnet_auc=0.742, ref_tabular_auc=0.674),
]).to_csv(str(RESULTS_DIR/'top20_xgb_metrics.csv'), index=False)

# ── Updated summary ────────────────────────────────────────────────────────
print('\n── Full progression summary ──')
print(f'  {"Model":<22}  {"Regr r":>8}  {"Classif AUC":>12}')
print(f'  {"256 raw (ElasticNet)":<22}  {0.277:>8.3f}  {0.637:>12.3f}')
print(f'  {"116 (ElasticNet)":<22}  {0.271:>8.3f}  {0.668:>12.3f}')
print(f'  {"Top-20 (ElasticNet)":<22}  {0.297:>8.3f}  {0.742:>12.3f}')
print(f'  {"Top-20 (XGBoost)":<22}  {cv_r:>8.3f}  {auc:>12.3f}')
print(f'  {"Tabular (reference)":<22}  {0.359:>8.3f}  {0.674:>12.3f}')

# ── Updated comparison figure ──────────────────────────────────────────────
import matplotlib.patches as mpatches

labels  = ['256 dims\n(raw EN)', '116 dims\n(EN)', '20 dims\n(EN)',
           '20 dims\n(XGBoost)', 'Tabular\n(ref)']
regr_r  = [0.277, 0.271, 0.297, cv_r,  0.359]
clf_auc = [0.637, 0.668, 0.742, auc,   0.674]
colors  = ['#5C6BC0','#42A5F5','#1565C0','#2E7D32','#EF6C00']

x = np.arange(len(labels))
w = 0.55

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.35)

for ax, vals, ylabel, title, chance, ref_val, ref_lbl in [
    (axes[0], regr_r,  'Pearson r', 'Regression\n(delay BFP→mCherry)',
     None, 0.359, 'Tabular r = 0.359'),
    (axes[1], clf_auc, 'AUC',       'Classification\n(early vs med+late)',
     0.500, 0.674, 'Tabular AUC = 0.674'),
]:
    bars = ax.bar(x, vals, width=w, color=colors, zorder=3,
                  edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if chance is not None:
        ax.axhline(chance, color='#9E9E9E', lw=1.2, ls='--', zorder=2,
                   label=f'Chance AUC = {chance:.3f}')
    ax.axhline(ref_val, color='#EF6C00', lw=1.5, ls=':', zorder=2, alpha=0.8,
               label=ref_lbl)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', lw=0.5, alpha=0.4, zorder=0)
    ax.legend(fontsize=8, loc='lower right' if chance else 'upper left')
    best_emb = int(np.argmax(vals[:4]))
    bars[best_emb].set_edgecolor('#FFD600'); bars[best_emb].set_linewidth(2.5)
    ymin = (min(vals)-0.04) if chance is None else min(chance-0.05, min(vals)-0.04)
    ax.set_ylim(max(0, ymin), max(vals)*1.12)

patches = [mpatches.Patch(color='#42A5F5', label='ElasticNet embedding'),
           mpatches.Patch(color='#2E7D32', label='XGBoost embedding'),
           mpatches.Patch(color='#EF6C00', label='Tabular features')]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Cellpose Neck Embeddings — A2 Dataset\nMorphology at GFP onset frame',
             fontsize=13, fontweight='bold', y=1.02)

out2 = FIGURES_DIR / 'embedding_comparison_figure2.png'
fig.savefig(str(out2), dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out2}')

print('\nDone.', flush=True)

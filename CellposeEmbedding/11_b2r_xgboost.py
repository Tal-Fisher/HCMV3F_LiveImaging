#!/usr/bin/env python3
"""
XGBoost regression (delay_blue_to_red) + classification (b2r early vs med+late)
using the top-20 GFP embedding dims. Consistent b2r target throughout.

Classification: early = delay_blue_to_red <= 1094 min (GMM cutoff, BluetoRed analysis).
No half-movie filter. Same dataset for both tasks (n=291 A2 productive cells).

References:
  ElasticNet top-20 b2r: regression r=0.297, classification AUC=0.708
  Tabular (TabICL A2+A3): regression r=0.382, classification AUC=0.678
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
CUT_B2R    = 1094
SEED       = 42

# ── Load data ──────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
d = np.load(str(EMB_NPZ))
emb_track_ids = d['track_ids']
embeddings    = d['embeddings'].astype(np.float64)
X_top = embeddings[:, TOP20_DIMS]
emb_id_to_row = {int(tid): i for i, tid in enumerate(emb_track_ids)}

df = pd.read_csv(MODEL_DF)
df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
df_a2['delay_blue_to_red'] = df_a2['delay_green_to_red'] - df_a2['delay_green_to_blue']
df_b2r = df_a2[
    np.isfinite(df_a2['delay_blue_to_red']) &
    df_a2['track_id'].isin(emb_id_to_row)
].sort_values('track_id').reset_index(drop=True)

rows  = [emb_id_to_row[tid] for tid in df_b2r['track_id']]
X     = X_top[rows]
y_reg = df_b2r['delay_blue_to_red'].values
y_cls = (y_reg <= CUT_B2R).astype(int)

print(f'  n={len(y_reg)}  b2r mean={y_reg.mean():.0f}  std={y_reg.std():.0f} min')
print(f'  Classification: {y_cls.sum()} early / {(y_cls==0).sum()} med+late  (cut={CUT_B2R} min)')

# ── Hyperparameter grid ────────────────────────────────────────────────────
XGB_GRID = {
    'n_estimators':      [100, 200, 400],
    'max_depth':         [2, 3, 4],
    'learning_rate':     [0.01, 0.05, 0.1],
    'subsample':         [0.6, 0.8, 1.0],
    'colsample_bytree':  [0.6, 0.8, 1.0],
    'min_child_weight':  [1, 3, 5],
    'gamma':             [0, 0.1, 0.3],
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
# REGRESSION — delay_blue_to_red
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ REGRESSION (XGBoost, top-20 dims, b2r) ═══', flush=True)

outer_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_kf = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)
oof_reg  = np.zeros(len(y_reg))
best_params_r = []

for fold, (tr, te) in enumerate(outer_kf.split(X)):
    xgb_r  = XGBRegressor(tree_method='hist', random_state=SEED,
                           eval_metric='rmse', verbosity=0)
    search = RandomizedSearchCV(xgb_r, XGB_GRID, n_iter=40, cv=inner_kf,
                                scoring='neg_mean_absolute_error',
                                random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X[tr], y_reg[tr])
    oof_reg[te] = search.best_estimator_.predict(X[te])
    best_params_r.append(search.best_params_)
    print(f'  Fold {fold+1}: {search.best_params_}', flush=True)

cv_r2  = r2_score(y_reg, oof_reg)
cv_r   = pearsonr(y_reg, oof_reg)[0]
cv_mae = mean_absolute_error(y_reg, oof_reg)
print(f'  5-fold CV:  R²={cv_r2:.3f}  r={cv_r:.3f}  MAE={cv_mae:.1f} min')
print(f'  ElasticNet reference: r=0.297  |  Tabular (TabICL): r=0.382')

xgb_rf = XGBRegressor(tree_method='hist', random_state=SEED, verbosity=0,
                       **{k: pd.Series([p[k] for p in best_params_r]).mode()[0]
                          for k in XGB_GRID})
xgb_rf.fit(X, y_reg)
imp_r = xgb_rf.feature_importances_

plot_importance(imp_r, f'XGBoost regression importance\ndelay_blue_to_red  CV r={cv_r:.3f}',
                FIGURES_DIR / 'b2r_xgb_regression_importance.png')

tertile = np.array(pd.qcut(y_reg, q=3, labels=['early', 'medium', 'late']).astype(str))
colours = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}
fig, ax = plt.subplots(figsize=(5, 5))
for t in ['early', 'medium', 'late']:
    m = tertile == t
    ax.scatter(y_reg[m], oof_reg[m], c=colours[t], label=t, alpha=0.7, s=25, edgecolors='none')
lims = [y_reg.min() - 100, y_reg.max() + 100]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.4)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual delay_blue_to_red (min)')
ax.set_ylabel('Predicted (min)')
ax.set_title(f'XGBoost — delay_blue_to_red (top-20)\nCV R²={cv_r2:.3f}  r={cv_r:.3f}  (n={len(y_reg)})')
ax.legend(title='Tertile', fontsize=8)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'b2r_xgb_scatter_regression.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved b2r_xgb_regression_importance.png + b2r_xgb_scatter_regression.png')

# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — b2r early vs med+late
# ══════════════════════════════════════════════════════════════════════════
print('\n═══ CLASSIFICATION (XGBoost, top-20 dims, b2r labels) ═══', flush=True)

scale_pos = float((y_cls == 0).sum()) / float(y_cls.sum())
print(f'  scale_pos_weight = {scale_pos:.2f}')

outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
oof_cls   = np.zeros(len(y_cls))
best_params_c = []

for fold, (tr, te) in enumerate(outer_skf.split(X, y_cls)):
    xgb_c  = XGBClassifier(tree_method='hist', random_state=SEED,
                            scale_pos_weight=scale_pos,
                            eval_metric='auc', use_label_encoder=False, verbosity=0)
    search = RandomizedSearchCV(xgb_c, XGB_GRID, n_iter=40, cv=inner_skf,
                                scoring='roc_auc',
                                random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X[tr], y_cls[tr])
    oof_cls[te] = search.best_estimator_.predict_proba(X[te])[:, 1]
    best_params_c.append(search.best_params_)
    print(f'  Fold {fold+1}: {search.best_params_}', flush=True)

auc  = roc_auc_score(y_cls, oof_cls)
pred = (oof_cls >= 0.5).astype(int)
tp = int(((pred == 1) & (y_cls == 1)).sum()); fn = int(((pred == 0) & (y_cls == 1)).sum())
tn = int(((pred == 0) & (y_cls == 0)).sum()); fp = int(((pred == 1) & (y_cls == 0)).sum())
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
bal  = balanced_accuracy_score(y_cls, pred)
print(f'  5-fold CV:  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}')
print(f'  ElasticNet reference: AUC=0.708  |  Tabular (TabICL): AUC=0.678')

xgb_cf = XGBClassifier(tree_method='hist', random_state=SEED, verbosity=0,
                        scale_pos_weight=scale_pos,
                        **{k: pd.Series([p[k] for p in best_params_c]).mode()[0]
                           for k in XGB_GRID})
xgb_cf.fit(X, y_cls)
imp_c = xgb_cf.feature_importances_

plot_importance(imp_c, f'XGBoost classification importance\nb2r early vs med+late  CV AUC={auc:.3f}',
                FIGURES_DIR / 'b2r_xgb_classify_importance.png')

fpr, tpr, _ = roc_curve(y_cls, oof_cls)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, color='#1565C0', label=f'XGBoost top-20  AUC={auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random AUC=0.500')
ax.axhline(0.708, color='#42A5F5', lw=1, ls=':', label='ElasticNet top-20 AUC=0.708')
ax.axhline(0.678, color='#EF6C00', lw=1, ls=':', label='Tabular (TabICL A2+A3) AUC=0.678')
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'B2R Early vs Med+Late — XGBoost top-20\n(n={len(y_cls)}, {y_cls.sum()} early, cut={CUT_B2R} min)')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'b2r_xgb_roc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved b2r_xgb_classify_importance.png + b2r_xgb_roc.png')

# ── Save metrics ───────────────────────────────────────────────────────────
pd.DataFrame([
    dict(task='regression_b2r', model='XGBoost', n_dims=20, n=len(y_reg),
         cv_r2=round(cv_r2, 4), cv_r=round(cv_r, 4), cv_mae=round(cv_mae, 1),
         ref_elasticnet_r=0.297, ref_tabular_r=0.382),
    dict(task='classify_b2r_early_vs_rest', model='XGBoost', n_dims=20, n=len(y_cls),
         n_early=int(y_cls.sum()), cut=CUT_B2R,
         auc=round(auc, 3), sens=round(sens, 3), spec=round(spec, 3), bal_acc=round(bal, 3),
         ref_elasticnet_auc=0.708, ref_tabular_auc=0.678),
]).to_csv(str(RESULTS_DIR / 'b2r_xgb_metrics.csv'), index=False)

print('\n── Summary ──')
print(f'  {"Model":<25}  {"Regr r":>8}  {"Classif AUC":>12}')
print(f'  {"ElasticNet top-20 (b2r)":<25}  {0.297:>8.3f}  {0.708:>12.3f}')
print(f'  {"XGBoost top-20 (b2r)":<25}  {cv_r:>8.3f}  {auc:>12.3f}')
print(f'  {"Tabular TabICL (b2r)":<25}  {0.382:>8.3f}  {0.678:>12.3f}')
print('\nDone.', flush=True)

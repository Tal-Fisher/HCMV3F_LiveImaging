#!/usr/bin/env python3
"""
PCA → StandardScaler → ElasticNetCV regression on Cellpose neck embeddings.
Target: delay_green_to_red (minutes). Baseline tabular CV R²=0.104, r=0.323.

Output:
  results/metrics.csv          -- R², r, MAE for train/test + baseline comparison
  results/predictions.csv      -- track_id, y_true, y_pred, split, tertile
  figures/scatter_test.png     -- predicted vs actual (test set)
  figures/pca_scree.png        -- PCA explained variance
  models/pca_model.pkl
  models/scaler.pkl
  models/elasticnet_model.pkl
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding')
LIVEIMG = Path('/home/labs/ginossar/talfis/LiveImaging')

EMB_NPZ  = BASE / 'embeddings' / 'A2_cell_embeddings.npz'
MODEL_DF = LIVEIMG / 'cache' / 'python_export' / 'model_df.csv'

RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
MODELS_DIR  = BASE / 'models'
for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

PCA_COMPONENTS  = 50
TEST_FRAC       = 0.25
RANDOM_STATE    = 42
BASELINE_R2     = 0.104
BASELINE_R      = 0.323

# ── Load data ──────────────────────────────────────────────────────────────
print('Loading embeddings...', flush=True)
d = np.load(str(EMB_NPZ))
track_ids  = d['track_ids']          # (291,) int64
embeddings = d['embeddings']         # (291, 256) float32
print(f'  {embeddings.shape[0]} cells × {embeddings.shape[1]} dims')

print('Loading model_df...', flush=True)
df = pd.read_csv(MODEL_DF)
df_a2 = df[df['dataset'] == 'A2'].copy()
df_a2['track_id'] = df_a2['Track.ID'].str.replace('A2_', '', regex=False).astype(int)
a2 = df_a2[np.isfinite(df_a2['delay_green_to_red'])][
    ['track_id', 'delay_green_to_red']
].copy()

# Merge on track_id
emb_df = pd.DataFrame({'track_id': track_ids})
emb_df = emb_df.merge(a2, on='track_id', how='inner')
idx    = emb_df.index  # indices into embeddings array

# Align embeddings to merged order
id_to_row = {int(tid): i for i, tid in enumerate(track_ids)}
emb_rows  = [id_to_row[tid] for tid in emb_df['track_id']]
X = embeddings[emb_rows]                                  # (N, 256)
y = emb_df['delay_green_to_red'].values.astype(np.float32)

# Tertile stratification labels (for balanced train/test split)
emb_df['tertile'] = pd.qcut(y, q=3, labels=['early', 'medium', 'late'])
strat = emb_df['tertile'].astype(str).values

print(f'  Merged cells: {len(y)}')
print(f'  Tertile counts: {pd.Series(strat).value_counts().to_dict()}')

# ── Train / test split ─────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(X, strat))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
ids_train = emb_df['track_id'].values[train_idx]
ids_test  = emb_df['track_id'].values[test_idx]
tert_test = strat[test_idx]

print(f'  Train: {len(y_train)}  Test: {len(y_test)}')

# ── PCA ────────────────────────────────────────────────────────────────────
print(f'PCA({PCA_COMPONENTS})...', flush=True)
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

explained = pca.explained_variance_ratio_
print(f'  Cumulative variance ({PCA_COMPONENTS} PCs): {explained.sum():.3f}')

# ── Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_pca)
X_test_sc  = scaler.transform(X_test_pca)

# ── ElasticNetCV ───────────────────────────────────────────────────────────
print('ElasticNetCV (10-fold CV on train)...', flush=True)
en = ElasticNetCV(
    l1_ratio=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    alphas=np.logspace(-4, 2, 60),
    cv=10,
    max_iter=20000,
    random_state=RANDOM_STATE,
)
en.fit(X_train_sc, y_train)
print(f'  Best alpha={en.alpha_:.4f}  l1_ratio={en.l1_ratio_:.2f}')
print(f'  Non-zero coefs: {(en.coef_ != 0).sum()}/{len(en.coef_)}')

# ── Evaluate ───────────────────────────────────────────────────────────────
y_train_pred = en.predict(X_train_sc)
y_test_pred  = en.predict(X_test_sc)

train_r2  = r2_score(y_train, y_train_pred)
test_r2   = r2_score(y_test,  y_test_pred)
train_r   = pearsonr(y_train, y_train_pred)[0]
test_r    = pearsonr(y_test,  y_test_pred)[0]
test_mae  = mean_absolute_error(y_test, y_test_pred)

print(f'\n  Train R²={train_r2:.3f}  r={train_r:.3f}')
print(f'  Test  R²={test_r2:.3f}  r={test_r:.3f}  MAE={test_mae:.1f} min')
print(f'  Baseline tabular: R²={BASELINE_R2}  r={BASELINE_R}')

# ── Save metrics ───────────────────────────────────────────────────────────
metrics = pd.DataFrame([{
    'n_train':              len(y_train),
    'n_test':               len(y_test),
    'pca_components':       PCA_COMPONENTS,
    'pca_cumvar':           round(float(explained.sum()), 4),
    'best_alpha':           round(en.alpha_, 6),
    'best_l1_ratio':        en.l1_ratio_,
    'n_nonzero_coefs':      int((en.coef_ != 0).sum()),
    'train_r2':             round(train_r2, 4),
    'test_r2':              round(test_r2,  4),
    'train_r':              round(train_r,  4),
    'test_r':               round(test_r,   4),
    'test_mae_min':         round(test_mae, 1),
    'baseline_tabular_r2':  BASELINE_R2,
    'baseline_tabular_r':   BASELINE_R,
}])
metrics_path = RESULTS_DIR / 'metrics.csv'
metrics.to_csv(str(metrics_path), index=False)
print(f'Saved: {metrics_path}')

# ── Save predictions ───────────────────────────────────────────────────────
pred_train = pd.DataFrame({
    'track_id': ids_train,
    'y_true':   y_train,
    'y_pred':   y_train_pred,
    'split':    'train',
    'tertile':  strat[train_idx],
})
pred_test = pd.DataFrame({
    'track_id': ids_test,
    'y_true':   y_test,
    'y_pred':   y_test_pred,
    'split':    'test',
    'tertile':  tert_test,
})
preds = pd.concat([pred_train, pred_test], ignore_index=True)
preds_path = RESULTS_DIR / 'predictions.csv'
preds.to_csv(str(preds_path), index=False)
print(f'Saved: {preds_path}')

# ── Figure 1: scatter test set ─────────────────────────────────────────────
colours = {'early': '#2196F3', 'medium': '#FF9800', 'late': '#F44336'}
fig, ax = plt.subplots(figsize=(5, 5))
for label, grp in pred_test.groupby('tertile'):
    ax.scatter(grp['y_true'], grp['y_pred'],
               c=colours.get(label, 'gray'), label=label, alpha=0.7, s=30, edgecolors='none')
lims = [min(y_test.min(), y_test_pred.min()) - 100,
        max(y_test.max(), y_test_pred.max()) + 100]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual delay (min)')
ax.set_ylabel('Predicted delay (min)')
ax.set_title(f'Test set  R²={test_r2:.3f}  r={test_r:.3f}\n'
             f'(n={len(y_test)}, baseline r={BASELINE_R})')
ax.legend(title='Tertile', fontsize=8)
plt.tight_layout()
scatter_path = FIGURES_DIR / 'scatter_test.png'
fig.savefig(str(scatter_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {scatter_path}')

# ── Figure 2: PCA scree ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(1, PCA_COMPONENTS + 1), explained * 100, color='steelblue', alpha=0.8)
ax2 = ax.twinx()
ax2.plot(range(1, PCA_COMPONENTS + 1), np.cumsum(explained) * 100,
         'r-o', markersize=3, lw=1.5)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained variance (%)')
ax2.set_ylabel('Cumulative (%)', color='red')
ax2.tick_params(axis='y', colors='red')
ax.set_title(f'PCA of 256-dim Cellpose embeddings (cum. {explained.sum()*100:.1f}% in {PCA_COMPONENTS} PCs)')
plt.tight_layout()
scree_path = FIGURES_DIR / 'pca_scree.png'
fig.savefig(str(scree_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {scree_path}')

# ── Save model pickles ─────────────────────────────────────────────────────
for obj, name in [(pca, 'pca_model'), (scaler, 'scaler'), (en, 'elasticnet_model')]:
    p = MODELS_DIR / f'{name}.pkl'
    with open(str(p), 'wb') as f:
        pickle.dump(obj, f)
print(f'Saved models to {MODELS_DIR}/')

print('\nDone.', flush=True)

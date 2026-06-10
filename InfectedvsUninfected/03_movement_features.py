#!/usr/bin/env python3
"""
03_movement_features.py

Extract movement features from the trajectories of the same cells selected
by 01_extract_embeddings.py, then classify infected vs uninfected.

For infected cells  (label=1): track forward TRACK_WINDOW frames from
    target_frame = onset-LOOKBACK.  This covers the pre-GFP-onset window,
    i.e. the cell's motion *before* viral immediate-early gene expression.

For uninfected cells (label=0): track forward TRACK_WINDOW frames from
    target_frame (their temporally-matched extraction frame).

Both windows are the same length so features are directly comparable.

Movement features per cell (9 total):
    mean_speed          – mean frame-to-frame displacement (µm/frame)
    max_speed           – maximum step size
    std_speed           – SD of step sizes
    net_displacement    – distance from first to last position (µm)
    total_path_length   – sum of all step sizes (µm)
    confinement_ratio   – net_displacement / total_path_length
    msd_lag1            – mean squared displacement at lag 1 (µm²)
    msd_lag5            – MSD at lag 5 (µm²)
    speed_trend         – slope of speed vs frame index (µm/frame²)

Outputs (results/):
    movement_features.csv   – per-cell features + label
    movement_metrics.csv    – CV AUC summary
    movement_vs_emb.csv     – movement / embedding / fusion AUC comparison
Outputs (figures/):
    movement_violin.png     – per-feature violin plots
    movement_roc.png        – ROC curves: movement / embedding / fusion
    movement_speed_trace.png – mean ± SEM speed vs relative frame
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, linregress
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent
LIVEIMG = BASE.parent

ALLSPOTS  = LIVEIMG / 'CompleteImage' / 'A2_BrightField_allspots.csv'
EMB_CSV   = BASE / 'embeddings' / 'A2_infected_vs_uninfected.csv'
TOP20_CSV = BASE / 'results' / 'top20_features.csv'

OUT_DIR = BASE / 'results'
FIG_DIR = BASE / 'figures'
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
MAX_RADIUS_UM   = 15.0
AMBIGUITY_RATIO = 2.0
TRACK_WINDOW    = 10   # forward frames used for both classes (equal window)

RANDOM_SEED = 42
N_FOLDS     = 5

rng = np.random.default_rng(RANDOM_SEED)

# ── Load cell metadata from embedding CSV ──────────────────────────────────────
print('Loading cell metadata …')
emb_df = pd.read_csv(EMB_CSV)
# columns: label, track_id, bf_track_id, target_frame, x_um, y_um, emb_0..255
print(f'  {len(emb_df)} cells  ({(emb_df["label"]==1).sum()} infected, '
      f'{(emb_df["label"]==0).sum()} uninfected)')

# ── Load allspots and build per-frame KD-trees ─────────────────────────────────
print('Loading BF allspot detections …')
chunks = []
for chunk in pd.read_csv(
        ALLSPOTS, low_memory=False, chunksize=200_000,
        usecols=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y']):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunks.append(chunk.dropna(subset=['FRAME', 'POSITION_X', 'POSITION_Y']))
allspots = pd.concat(chunks, ignore_index=True)
allspots['FRAME'] = allspots['FRAME'].astype(int)
print(f'  {len(allspots):,} rows, {allspots["FRAME"].nunique()} frames')

print('Building per-frame KD-trees …')
trees  = {}
coords = {}
for f, grp in allspots.groupby('FRAME'):
    xy = grp[['POSITION_X', 'POSITION_Y']].values
    coords[f] = xy
    trees[f]  = cKDTree(xy)
MAX_FRAME = max(trees)
print(f'  Done.')

# ── Strict forward NN tracker ──────────────────────────────────────────────────
def track_forward(start_frame, x0, y0, n_frames):
    """
    Track forward n_frames steps from (start_frame, x0, y0).
    Returns list of (frame, x, y) if all links succeed, else None.
    """
    if start_frame not in trees:
        return None
    k = min(2, len(coords[start_frame]))
    dists, idxs = trees[start_frame].query([x0, y0], k=k)
    dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
    if dists[0] > MAX_RADIUS_UM:
        return None
    x, y = coords[start_frame][idxs[0]]
    positions = [(start_frame, x, y)]

    for f in range(start_frame + 1, start_frame + n_frames + 1):
        if f not in trees or f > MAX_FRAME:
            return None
        k2 = min(2, len(coords[f]))
        d2, i2 = trees[f].query([x, y], k=k2)
        d2 = np.atleast_1d(d2); i2 = np.atleast_1d(i2)
        if d2[0] > MAX_RADIUS_UM:
            return None
        if k2 >= 2 and d2[1] < AMBIGUITY_RATIO * d2[0]:
            return None
        x, y = coords[f][i2[0]]
        positions.append((f, x, y))

    return positions

# ── Movement feature computation ───────────────────────────────────────────────
def compute_features(positions):
    """
    positions: list of (frame, x_um, y_um), length = TRACK_WINDOW+1.
    Returns dict of movement features.
    """
    arr = np.array(positions)          # (N, 3): frame, x, y
    dx  = np.diff(arr[:, 1])
    dy  = np.diff(arr[:, 2])
    steps = np.sqrt(dx**2 + dy**2)    # (N-1,) step sizes

    mean_speed = steps.mean()
    max_speed  = steps.max()
    std_speed  = steps.std() if len(steps) > 1 else 0.0

    net_disp = np.sqrt((arr[-1, 1] - arr[0, 1])**2 +
                       (arr[-1, 2] - arr[0, 2])**2)
    total_path = steps.sum()
    confinement = net_disp / (total_path + 1e-9)

    # MSD at lag τ: mean over all pairs separated by τ steps
    def msd(tau):
        if tau >= len(arr):
            return np.nan
        d = arr[tau:, 1:] - arr[:-tau, 1:]
        return float(np.mean((d**2).sum(axis=1)))

    msd1 = msd(1)
    msd5 = msd(5)

    # Speed trend: slope of step size vs step index
    if len(steps) >= 2:
        slope, *_ = linregress(np.arange(len(steps)), steps)
    else:
        slope = 0.0

    return {
        'mean_speed':       mean_speed,
        'max_speed':        max_speed,
        'std_speed':        std_speed,
        'net_displacement': net_disp,
        'total_path':       total_path,
        'confinement':      confinement,
        'msd_lag1':         msd1,
        'msd_lag5':         msd5,
        'speed_trend':      slope,
    }

# ── Track each cell and compute features ──────────────────────────────────────
print(f'\nTracking cells forward {TRACK_WINDOW} frames from target_frame …')

FEAT_COLS = ['mean_speed', 'max_speed', 'std_speed',
             'net_displacement', 'total_path', 'confinement',
             'msd_lag1', 'msd_lag5', 'speed_trend']

records  = []
traces   = {'infected': [], 'uninfected': []}   # for speed-trace plot
n_fail   = {'infected': 0, 'uninfected': 0}

for _, row in emb_df.iterrows():
    label  = int(row['label'])
    gname  = 'infected' if label == 1 else 'uninfected'
    f0     = int(row['target_frame'])
    x0     = float(row['x_um'])
    y0     = float(row['y_um'])

    positions = track_forward(f0, x0, y0, TRACK_WINDOW)
    if positions is None or len(positions) < TRACK_WINDOW + 1:
        n_fail[gname] += 1
        continue

    feats = compute_features(positions)

    # store per-step speeds for trace plot (relative frame index 0..TRACK_WINDOW-1)
    arr   = np.array(positions)
    steps = np.sqrt(np.diff(arr[:, 1])**2 + np.diff(arr[:, 2])**2)
    traces[gname].append(steps)

    records.append({
        'label':      label,
        'track_id':   row['track_id'],
        'target_frame': f0,
        **feats,
    })

print(f'  Infected:   {(emb_df["label"]==1).sum()} → '
      f'{sum(1 for r in records if r["label"]==1)} tracked '
      f'({n_fail["infected"]} failed)')
print(f'  Uninfected: {(emb_df["label"]==0).sum()} → '
      f'{sum(1 for r in records if r["label"]==0)} tracked '
      f'({n_fail["uninfected"]} failed)')

feat_df = pd.DataFrame(records)
feat_df.to_csv(OUT_DIR / 'movement_features.csv', index=False)
print(f'  Saved: results/movement_features.csv')

# ── Mann-Whitney U tests per feature ──────────────────────────────────────────
print('\nMann-Whitney U tests (infected vs uninfected):')
for col in FEAT_COLS:
    inf_vals = feat_df.loc[feat_df['label']==1, col].dropna()
    non_vals = feat_df.loc[feat_df['label']==0, col].dropna()
    stat, pval = mannwhitneyu(inf_vals, non_vals, alternative='two-sided')
    print(f'  {col:<22} U={stat:.0f}  p={pval:.3g}  '
          f'inf_med={inf_vals.median():.3f}  non_med={non_vals.median():.3f}')

# ── Classifier helpers ─────────────────────────────────────────────────────────
def cv_auc(X, y, random_state=RANDOM_SEED):
    """5-fold stratified CV with LogisticRegressionCV (elasticnet)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=random_state)
    aucs, tprs = [], []
    mean_fpr = np.linspace(0, 1, 200)

    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf = LogisticRegressionCV(
                penalty='elasticnet', solver='saga',
                l1_ratios=[0.1, 0.5, 0.9], cv=3,
                class_weight='balanced',
                max_iter=2000, random_state=random_state,
            )
            clf.fit(X_tr, y[tr])

        proba = clf.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y[va], proba))

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y[va], proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    return float(np.mean(aucs)), float(np.std(aucs)), mean_fpr, np.array(tprs)

def null_auc(X, y, random_state=RANDOM_SEED):
    y_shuf = y.copy()
    rng2 = np.random.default_rng(random_state)
    rng2.shuffle(y_shuf)
    auc, *_ = cv_auc(X, y_shuf, random_state=random_state)
    return auc

# ── Prepare datasets ──────────────────────────────────────────────────────────
# Sort feat_df by (label, track_id, target_frame) for a stable row order
feat_df = feat_df.sort_values(['label', 'track_id', 'target_frame']).reset_index(drop=True)
y_all = feat_df['label'].values
X_mov = feat_df[FEAT_COLS].values

# Align Cellpose top-20 embeddings to the same row order as feat_df
top20    = pd.read_csv(TOP20_CSV)
top_cols = top20['feature_name'].tolist()

emb_sub = feat_df[['track_id', 'target_frame', 'label']].merge(
    emb_df[['track_id', 'target_frame'] + top_cols],
    on=['track_id', 'target_frame'], how='left')

X_emb = emb_sub[top_cols].values

assert len(X_emb) == len(X_mov), \
    f'Cell count mismatch: emb={len(X_emb)}, mov={len(X_mov)}'
assert not np.isnan(X_emb).any(), \
    'NaNs in aligned embeddings — some cells missing from emb CSV'

# Fusion: movement + top-20 embedding
X_fus = np.hstack([X_mov, X_emb])

# ── Run classifiers ────────────────────────────────────────────────────────────
print('\nRunning classifiers …')
print('  [1/4] Movement features …')
auc_mov, std_mov, fpr_mov, tprs_mov = cv_auc(X_mov, y_all)
print(f'        AUC = {auc_mov:.3f} ± {std_mov:.3f}')

print('  [2/4] Top-20 Cellpose embeddings …')
auc_emb, std_emb, fpr_emb, tprs_emb = cv_auc(X_emb, y_emb)
print(f'        AUC = {auc_emb:.3f} ± {std_emb:.3f}')

print('  [3/4] Fusion (movement + embeddings) …')
auc_fus, std_fus, fpr_fus, tprs_fus = cv_auc(X_fus, y_all)
print(f'        AUC = {auc_fus:.3f} ± {std_fus:.3f}')

print('  [4/4] Null (shuffled labels) …')
auc_null = null_auc(X_mov, y_all)
print(f'        AUC = {auc_null:.3f}')

summary = pd.DataFrame([
    {'model': 'movement_only',  'AUC': auc_mov, 'AUC_std': std_mov},
    {'model': 'top20_embeddings', 'AUC': auc_emb, 'AUC_std': std_emb},
    {'model': 'fusion',         'AUC': auc_fus, 'AUC_std': std_fus},
    {'model': 'null',           'AUC': auc_null, 'AUC_std': np.nan},
])
summary.to_csv(OUT_DIR / 'movement_vs_emb.csv', index=False)
print(f'\n  Saved: results/movement_vs_emb.csv')

# ── Figure 1: per-feature violin plots ────────────────────────────────────────
n_feat = len(FEAT_COLS)
n_cols = 3
n_rows = (n_feat + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = axes.flatten()

colors = {'infected': '#e05b5b', 'uninfected': '#5b8be0'}

for ax, col in zip(axes, FEAT_COLS):
    inf_vals = feat_df.loc[feat_df['label']==1, col].dropna().values
    non_vals = feat_df.loc[feat_df['label']==0, col].dropna().values
    stat, pval = mannwhitneyu(inf_vals, non_vals, alternative='two-sided')

    for i, (vals, gname) in enumerate([(non_vals, 'uninfected'), (inf_vals, 'infected')]):
        parts = ax.violinplot(vals, positions=[i], widths=0.6,
                              showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[gname]); pc.set_alpha(0.5)
        parts['cmedians'].set_color('black')
        jitter = rng.uniform(-0.15, 0.15, size=min(len(vals), 300))
        plot_vals = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
        ax.scatter(i + jitter[:len(plot_vals)], plot_vals,
                   s=4, alpha=0.3, color=colors[gname], zorder=3)

    p_str = f'p={pval:.2e}' if pval < 0.001 else f'p={pval:.3f}'
    y_top = max(inf_vals.max(), non_vals.max()) * 1.05
    ax.plot([0, 0, 1, 1], [y_top*0.94, y_top, y_top, y_top*0.94],
            lw=1, color='black')
    ax.text(0.5, y_top * 1.01, p_str, ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Non-inf\n(n={len(non_vals)})', f'Inf\n(n={len(inf_vals)})'],
                       fontsize=8)
    ax.set_title(col.replace('_', ' '), fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

for ax in axes[n_feat:]:
    ax.set_visible(False)

fig.suptitle('Movement features: infected vs uninfected (pre-GFP-onset window)',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / 'movement_violin.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: figures/movement_violin.png')

# ── Figure 2: ROC curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))

for (fpr, tprs, auc, std, label, color) in [
    (fpr_mov, tprs_mov, auc_mov, std_mov, 'Movement only', '#e09b2b'),
    (fpr_emb, tprs_emb, auc_emb, std_emb, 'Top-20 Cellpose', '#7b5be0'),
    (fpr_fus, tprs_fus, auc_fus, std_fus, 'Fusion',          '#2ca02c'),
]:
    mean_tpr = tprs.mean(axis=0)
    ax.plot(fpr, mean_tpr, color=color, lw=2,
            label=f'{label}  AUC={auc:.3f}±{std:.3f}')
    ax.fill_between(fpr, tprs.min(axis=0), tprs.max(axis=0),
                    color=color, alpha=0.12)

ax.plot([0, 1], [0, 1], 'k--', lw=1, label=f'Null  AUC={auc_null:.3f}')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('Infection prediction: movement vs embeddings vs fusion')
ax.legend(fontsize=9, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / 'movement_roc.png', dpi=150)
plt.close(fig)
print('Saved: figures/movement_roc.png')

# ── Figure 3: mean ± SEM speed over relative frame ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

for gname, color in [('uninfected', '#5b8be0'), ('infected', '#e05b5b')]:
    mat = np.array(traces[gname])   # (n_cells, TRACK_WINDOW)
    n   = len(mat)
    mean = mat.mean(axis=0)
    sem  = mat.std(axis=0) / np.sqrt(n)
    x    = np.arange(TRACK_WINDOW)
    ax.plot(x, mean, color=color, lw=2, label=f'{gname.capitalize()} (n={n})')
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)

ax.set_xlabel('Frame index relative to tracking start\n'
              '(infected: onset−10 → onset;  uninfected: extraction frame → +10)')
ax.set_ylabel('Mean speed (µm / frame)')
ax.set_title('Cell movement during tracked window: infected vs uninfected')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / 'movement_speed_trace.png', dpi=150)
plt.close(fig)
print('Saved: figures/movement_speed_trace.png')

# ── Final summary ──────────────────────────────────────────────────────────────
print('\n' + '='*55)
print('Summary')
print('='*55)
for _, row in summary.iterrows():
    std_str = f'±{row["AUC_std"]:.3f}' if pd.notna(row['AUC_std']) else ''
    print(f'  {row["model"]:<22} AUC = {row["AUC"]:.3f} {std_str}')
print('='*55)
print('\nDone.')

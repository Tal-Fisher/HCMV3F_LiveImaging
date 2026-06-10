#!/usr/bin/env python3
"""
06_bidir_retrack_size.py

Same cell selection and bidirectional re-tracking as 03_bf_bidir_retrack.py,
but plots BF spot radius (cell size, µm) instead of movement speed.

Infected cells  – bidir-tracked from GFP onset, GFP-position validated.
Non-infected    – quality-passing TrackMate BF tracks (≥30 frames, ≥90%
                  coverage, no GFP match at any tier), tracked forward.
GFP overlay     – radius from GFP segmentation (column "R") for the same
                  surviving infected cells.

Outputs:
  bf_size_bidir_retracked.png           -- BF size: infected vs non-infected
  bf_size_bidir_retracked_with_gfp.png  -- same + GFP segmentation radius
  bf_size_halves.png                    -- first-half vs second-half per cell
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
ALLSPOTS    = ROOT / 'CompleteImage' / 'A2_BrightField_allspots.csv'
BF_SPOTS    = ROOT / 'CompleteImage' / 'A2_BrightField_spots.csv'
GFP_SPOTS   = ROOT / 'CompleteImage' / 'A2_Merged_spots.csv'
TIMESERIES  = ROOT / 'cache' / 'python_export' / 'timeseries_data.csv'
GFP_MATCHES = ROOT / 'BrightFieldEmbedding' / 'bf_gfp_matches.csv'
OUT_DIR     = Path(__file__).resolve().parent

# ── Parameters (identical to 03_bf_bidir_retrack.py) ─────────────────────────
MAX_RADIUS       = 15.0
AMBIGUITY_RATIO  = 2.0
MIN_BACK_FRAMES  = 10
MIN_FWD_FRAMES   = 10
MIN_TOTAL_FRAMES = 80
CELL_RADIUS_UM   = 20.5
MIN_FRAC_WITHIN  = 0.90
SMOOTH_WIN       = 15

# ── Load allspots with Radius ──────────────────────────────────────────────────
print('Loading all BF detections (with Radius) …')
allspots = pd.read_csv(
    ALLSPOTS, skiprows=[0, 2, 3], header=0,
    usecols=['Frame', 'X', 'Y', 'Radius'],
    dtype={'Frame': int, 'X': float, 'Y': float, 'Radius': float},
)
allspots = allspots.dropna(subset=['X', 'Y', 'Radius'])
print(f'  {len(allspots):,} detections, {allspots["Frame"].nunique()} frames')

print('Building per-frame KD-trees …')
coords = {}
trees  = {}
radii  = {}
for f, grp in allspots.groupby('Frame'):
    xy = grp[['X', 'Y']].values
    coords[f] = xy
    radii[f]  = grp['Radius'].values
    trees[f]  = cKDTree(xy)
MAX_FRAME = max(coords)
MIN_FRAME = min(coords)
print('  Done.')

# ── Tracking helpers ───────────────────────────────────────────────────────────
def _step(x, y, frame):
    """Return (new_x, new_y, spot_idx) if link is unambiguous, else None."""
    if frame not in trees:
        return None
    k = min(2, len(coords[frame]))
    dists, idxs = trees[frame].query([x, y], k=k)
    dists = np.atleast_1d(dists)
    idxs  = np.atleast_1d(idxs)
    if dists[0] > MAX_RADIUS:
        return None
    if len(dists) > 1 and dists[1] < AMBIGUITY_RATIO * dists[0]:
        return None
    idx = int(idxs[0])
    return coords[frame][idx][0], coords[frame][idx][1], idx

def snap(frame, x, y):
    if frame not in trees:
        return None
    d, i = trees[frame].query([x, y], k=1)
    if d > MAX_RADIUS:
        return None
    i = int(i)
    return coords[frame][i][0], coords[frame][i][1], i

def track_backward(seed_frame, seed_x, seed_y):
    """Chronological list of (frame, x, y, radius) before seed_frame."""
    pos = []
    x, y = seed_x, seed_y
    for f in range(seed_frame - 1, MIN_FRAME - 1, -1):
        result = _step(x, y, f)
        if result is None:
            break
        x, y, idx = result
        pos.append((f, x, y, radii[f][idx]))
    pos.reverse()
    return pos

def track_forward(seed_frame, seed_x, seed_y):
    """List of (frame, x, y, radius) after seed_frame."""
    pos = []
    x, y = seed_x, seed_y
    for f in range(seed_frame + 1, MAX_FRAME + 1):
        result = _step(x, y, f)
        if result is None:
            break
        x, y, idx = result
        pos.append((f, x, y, radii[f][idx]))
    return pos

# ── Load GFP tracks ────────────────────────────────────────────────────────────
print('\nLoading GFP track positions and radii …')
gfp = pd.read_csv(GFP_SPOTS, low_memory=False,
                  usecols=['Track ID', 'Frame', 'X', 'Y', 'R'])
gfp = gfp.dropna(subset=['Track ID'])
gfp['Track ID'] = gfp['Track ID'].astype(int)
gfp['Track.ID'] = 'A2_' + gfp['Track ID'].astype(str)

gfp_pos_lookup = {
    tid: dict(zip(grp['Frame'], zip(grp['X'], grp['Y'])))
    for tid, grp in gfp.groupby('Track.ID')
}

# Load GFP-segmented cell area (px²) from timeseries; convert to equivalent
# radius in µm so it's on the same axis as BF radius.
# Area_cell is in px²; PIXEL_SCALE = 0.2871 µm/px → Area_µm² = Area × scale²
# Equivalent radius = sqrt(Area_µm² / π)
PIXEL_SCALE = 0.2871   # µm/px
ts = pd.read_csv(TIMESERIES, low_memory=False,
                 usecols=['Track.ID', 'Frame', 'Area_cell'])
ts = ts[ts['Track.ID'].str.startswith('A2_')].copy()
ts['equiv_radius'] = np.sqrt(ts['Area_cell'] * PIXEL_SCALE**2 / np.pi)
gfp_area_lookup = {
    tid: dict(zip(grp['Frame'], grp['equiv_radius']))
    for tid, grp in ts.groupby('Track.ID')
}
print(f'  GFP area data: {ts["Track.ID"].nunique()} A2 cells, '
      f'median equiv radius = {ts["equiv_radius"].median():.2f} µm')

onset_df = (gfp.sort_values('Frame')
               .groupby('Track.ID').first()
               .reset_index()[['Track.ID', 'Frame', 'X', 'Y']]
               .rename(columns={'Frame': 'onset_frame',
                                'X': 'onset_x', 'Y': 'onset_y'}))
print(f'  {len(onset_df)} GFP tracks')

# ── Bidirectional re-tracking for infected cells ───────────────────────────────
print('Bidirectional re-tracking of infected cells …')
inf_records = []   # list of DataFrames: Frame, radius, track_id
n_length_fail = n_gfp_fail = 0

for _, row in onset_df.iterrows():
    onset = int(row['onset_frame'])

    snapped = snap(onset, float(row['onset_x']), float(row['onset_y']))
    if snapped is None:
        continue
    sx, sy, s_idx = snapped

    back = track_backward(onset, sx, sy)
    fwd  = track_forward(onset, sx, sy)

    total = len(back) + 1 + len(fwd)
    if len(back) < MIN_BACK_FRAMES or len(fwd) < MIN_FWD_FRAMES or total < MIN_TOTAL_FRAMES:
        n_length_fail += 1
        continue

    # GFP-position validation
    seed_r  = radii[onset][s_idx]
    full_pos = back + [(onset, sx, sy, seed_r)] + fwd
    gfp_frames = gfp_pos_lookup.get(row['Track.ID'], {})
    dists = [
        np.sqrt((bx - gfp_frames[f][0])**2 + (by - gfp_frames[f][1])**2)
        for (f, bx, by, _) in full_pos if f in gfp_frames
    ]
    if len(dists) < 10 or np.mean(np.array(dists) <= CELL_RADIUS_UM) < MIN_FRAC_WITHIN:
        n_gfp_fail += 1
        continue

    df = pd.DataFrame(full_pos, columns=['Frame', 'x', 'y', 'radius'])
    df['track_id'] = row['Track.ID']
    inf_records.append(df)

inf_df = pd.concat(inf_records, ignore_index=True) if inf_records else pd.DataFrame()
n_inf  = len(inf_records)
print(f'  {n_inf} infected cells kept')
print(f'  (dropped {n_length_fail} for track length, {n_gfp_fail} for GFP drift)')

# ── Non-infected: TrackMate BF tracks ─────────────────────────────────────────
print('Loading non-infected BF tracks …')
bf = pd.read_csv(
    BF_SPOTS, skiprows=[0, 2, 3], header=0,
    usecols=['Track ID', 'Frame', 'X', 'Y', 'Radius'],
    dtype={'Track ID': float, 'Frame': int, 'X': float,
           'Y': float, 'Radius': float},
)
bf = bf.dropna(subset=['Track ID', 'Radius'])
bf['Track ID'] = bf['Track ID'].astype(int)

stats = bf.groupby('Track ID')['Frame'].agg(
    n_frames='count', frame_min='min', frame_max='max')
stats['frame_span'] = stats['frame_max'] - stats['frame_min'] + 1
stats['coverage']   = stats['n_frames'] / stats['frame_span']
good_ids = set(stats.index[(stats['n_frames'] >= 30) & (stats['coverage'] >= 0.90)])

matches = pd.read_csv(GFP_MATCHES)
all_matched_bf = set(matches['bf_track_id'].dropna().astype(int))
non_inf_ids = good_ids - all_matched_bf
print(f'  {len(non_inf_ids)} non-infected quality tracks')

bf_non = bf[bf['Track ID'].isin(non_inf_ids)].copy()
n_non  = len(non_inf_ids)

# ── Time-series helper ─────────────────────────────────────────────────────────
SMOOTH_WIN = 15

def time_series(df, val_col='radius'):
    ts = (df.groupby('Frame')[val_col]
          .agg(n='count', mean='mean', sem=lambda x: x.sem())
          .reset_index()
          .sort_values('Frame'))
    full_idx = pd.RangeIndex(ts['Frame'].min(), ts['Frame'].max() + 1)
    ts = (ts.set_index('Frame')
            .reindex(full_idx)
            .rolling(SMOOTH_WIN, center=True, min_periods=SMOOTH_WIN // 2)
            .mean()
            .reset_index()
            .rename(columns={'index': 'Frame'}))
    return ts

inf_ts = time_series(inf_df) if not inf_df.empty else pd.DataFrame()
non_ts = time_series(bf_non, val_col='Radius')

# ── GFP-segmented cell area (as equiv radius) — all productive A2 cells
# timeseries_data only contains productive cells (those that turn red).
# Restricting to the 24 bidir-tracked cells would leave too few with area data,
# so use all 458 productive cells for a stable GFP area trend estimate.
gfp_size_df = ts.rename(columns={'equiv_radius': 'radius'})[['Frame', 'radius', 'Track.ID']]
gfp_size_df = gfp_size_df.rename(columns={'Track.ID': 'track_id'})
gfp_ts = time_series(gfp_size_df) if not gfp_size_df.empty else pd.DataFrame()
n_gfp  = ts['Track.ID'].nunique()

# ── Per-track summary ─────────────────────────────────────────────────────────
print(f'\nPer-track mean radius:')
if not inf_df.empty:
    inf_mean_r = inf_df.groupby('track_id')['radius'].mean()
    print(f'  Infected BF    : {n_inf} cells  mean {inf_mean_r.mean():.2f} µm')
if not bf_non.empty:
    non_mean_r = bf_non.groupby('Track ID')['Radius'].mean()
    print(f'  Non-infected BF: {n_non} tracks mean {non_mean_r.mean():.2f} µm')
    stat, pval = mannwhitneyu(inf_mean_r, non_mean_r, alternative='two-sided')
    print(f'  Mann-Whitney U={stat:.0f}, p={pval:.4g}')

# ── Half-radius: first half vs second half per cell ───────────────────────────
def half_sizes(df, id_col='track_id', val_col='radius'):
    rows = []
    for tid, grp in df.groupby(id_col):
        frames = grp['Frame'].values
        mid    = (frames.min() + frames.max()) / 2
        h1 = grp.loc[grp['Frame'] <= mid, val_col].mean()
        h2 = grp.loc[grp['Frame'] >  mid, val_col].mean()
        if np.isfinite(h1) and np.isfinite(h2):
            rows.append({'id': tid, 'first': h1, 'second': h2})
    return pd.DataFrame(rows)

inf_h = half_sizes(inf_df)
non_h = half_sizes(bf_non, id_col='Track ID', val_col='Radius')

# ── Plot helper ────────────────────────────────────────────────────────────────
def plot_group(ax, ts, color, label, linestyle='-'):
    if ts.empty:
        return
    d = ts.dropna(subset=['mean'])
    ax.plot(d['Frame'], d['mean'], color=color, lw=1.5,
            label=label, linestyle=linestyle)
    ax.fill_between(d['Frame'], d['mean'] - d['sem'], d['mean'] + d['sem'],
                    color=color, alpha=0.2)

# ── Figure 1: BF size, infected vs non-infected ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_group(ax, non_ts, '#5b8be0', f'Non-infected BF (n={n_non})')
plot_group(ax, inf_ts, '#e05b5b', f'Infected BF (n={n_inf})')
ax.set_xlabel(f'Frame  ({SMOOTH_WIN}-frame rolling mean ± SEM)')
ax.set_ylabel('Mean BF spot radius (µm)')
ax.set_title('Cell size (BF radius): infected vs non-infected (A2)')
ax.legend(); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / 'bf_size_bidir_retracked.png', dpi=150)
plt.close(fig)
print('Saved: bf_size_bidir_retracked.png')

# ── Figure 2: same + GFP segmentation radius overlay ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_group(ax, non_ts, '#5b8be0', f'BF non-infected (n={n_non})')
plot_group(ax, inf_ts, '#e05b5b', f'BF infected (n={n_inf})')
plot_group(ax, gfp_ts, '#2ca02c',
           f'GFP area → equiv radius (n={n_gfp})', linestyle='--')
ax.set_xlabel(f'Frame  ({SMOOTH_WIN}-frame rolling mean ± SEM)')
ax.set_ylabel('Mean radius (µm)  |  GFP: √(Area·scale²/π)')
ax.set_title('Cell size: BF re-tracked radius + GFP equiv radius overlay (A2)')
ax.legend(); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / 'bf_size_bidir_retracked_with_gfp.png', dpi=150)
plt.close(fig)
print('Saved: bf_size_bidir_retracked_with_gfp.png')

# ── Figure 3: paired first-half vs second-half per cell ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharey=True)

for ax, halves, color, title in [
    (axes[0], non_h, '#5b8be0', f'Non-infected\n(n={len(non_h)})'),
    (axes[1], inf_h, '#e05b5b', f'Infected\n(n={len(inf_h)})'),
]:
    for _, row in halves.iterrows():
        ax.plot([0, 1], [row['first'], row['second']],
                color=color, alpha=0.25, lw=0.8)
    for xi, col in [(0, 'first'), (1, 'second')]:
        m = halves[col].mean()
        s = halves[col].sem()
        ax.errorbar(xi, m, yerr=s, fmt='o', color=color,
                    markersize=8, capsize=4, lw=2, zorder=5)
    ax.plot([0, 1], [halves['first'].mean(), halves['second'].mean()],
            color=color, lw=2.5, zorder=4)

    _, pval = wilcoxon(halves['first'], halves['second'])
    p_str = f'p={pval:.2e}' if pval < 0.001 else f'p={pval:.3f}'
    y_top = halves[['first', 'second']].max().max() * 1.07
    ax.plot([0, 0, 1, 1], [y_top*0.94, y_top, y_top, y_top*0.94], lw=1, color='black')
    ax.text(0.5, y_top * 1.01, p_str, ha='center', va='bottom', fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['First half', 'Second half'])
    ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('Mean BF spot radius (µm)')
fig.suptitle('Within-cell size change: first half vs second half of track (A2)',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / 'bf_size_halves.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: bf_size_halves.png')

print('\nDone.')

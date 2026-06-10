#!/usr/bin/env python3
"""
05_onset_aligned_size.py

Plot BF-detected cell radius aligned to GFP onset (t=0).

Same cell selection and tracking as 04_onset_aligned_speed.py.
Instead of computing speed, reads the TrackMate-detected radius (µm)
of the BF spot at each tracked frame.

Outputs (saved to this directory):
  bf_size_onset_aligned.png          -- onset-aligned radius trace
  bf_size_onset_aligned_halves.png   -- pre-onset vs post-onset per cell
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
ROOT       = Path(__file__).resolve().parents[1]
ALLSPOTS   = ROOT / 'CompleteImage' / 'A2_BrightField_allspots.csv'
BF_SPOTS   = ROOT / 'CompleteImage' / 'A2_BrightField_spots.csv'
GFP_SPOTS  = ROOT / 'CompleteImage' / 'A2_Merged_spots.csv'
GFP_MATCHES = ROOT / 'BrightFieldEmbedding' / 'bf_gfp_matches.csv'
OUT_DIR    = Path(__file__).resolve().parent

# ── Parameters (identical to 04_onset_aligned_speed.py) ──────────────────────
MAX_RADIUS      = 15.0
AMBIGUITY_RATIO = 2.0
MIN_BACK_FRAMES = 10
MIN_FWD_FRAMES  = 10
MAX_BACK        = 100
MAX_FWD         = 150
CELL_RADIUS_UM  = 20.5
MIN_FRAC_WITHIN = 0.90

PLOT_MIN = -40
PLOT_MAX = 100

# ── Load allspots WITH Radius and build per-frame structures ──────────────────
print('Loading all BF detections (with Radius) …')
allspots = pd.read_csv(
    ALLSPOTS, skiprows=[0, 2, 3], header=0,
    usecols=['Frame', 'X', 'Y', 'Radius'],
    dtype={'Frame': int, 'X': float, 'Y': float, 'Radius': float},
)
allspots = allspots.dropna(subset=['X', 'Y', 'Radius'])
print(f'  {len(allspots):,} detections, {allspots["Frame"].nunique()} frames')
print(f'  Radius: median={allspots["Radius"].median():.2f} µm  '
      f'mean={allspots["Radius"].mean():.2f} µm')

print('Building per-frame KD-trees …')
coords = {}
trees  = {}
radii  = {}   # frame -> (N,) array of radii, parallel to coords[frame]
for f, grp in allspots.groupby('Frame'):
    xy = grp[['X', 'Y']].values
    r  = grp['Radius'].values
    coords[f] = xy
    radii[f]  = r
    trees[f]  = cKDTree(xy)
MAX_FRAME = max(coords)
MIN_FRAME = min(coords)
print('  Done.')

# ── Tracking helpers (same as 04) ──────────────────────────────────────────────
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
    idx = idxs[0]
    return coords[frame][idx][0], coords[frame][idx][1], int(idx)

def snap(frame, x, y):
    """Return (x, y, idx) of nearest BF spot at frame, or None if > MAX_RADIUS."""
    if frame not in trees:
        return None
    d, i = trees[frame].query([x, y], k=1)
    if d > MAX_RADIUS:
        return None
    return coords[frame][i][0], coords[frame][i][1], int(i)

def track_backward(seed_frame, seed_x, seed_y, max_frames):
    """Return chronological list of (frame, x, y, radius)."""
    pos = []
    x, y = seed_x, seed_y
    for f in range(seed_frame - 1,
                   max(MIN_FRAME - 1, seed_frame - max_frames - 1), -1):
        result = _step(x, y, f)
        if result is None:
            break
        x, y, idx = result
        pos.append((f, x, y, radii[f][idx]))
    pos.reverse()
    return pos

def track_forward(seed_frame, seed_x, seed_y, max_frames):
    """Return list of (frame, x, y, radius) after seed_frame."""
    pos = []
    x, y = seed_x, seed_y
    for f in range(seed_frame + 1,
                   min(MAX_FRAME + 1, seed_frame + max_frames + 1)):
        result = _step(x, y, f)
        if result is None:
            break
        x, y, idx = result
        pos.append((f, x, y, radii[f][idx]))
    return pos

# ── Load GFP tracks ────────────────────────────────────────────────────────────
print('\nLoading GFP track positions …')
gfp = pd.read_csv(GFP_SPOTS, low_memory=False,
                  usecols=['Track ID', 'Frame', 'X', 'Y'])
gfp = gfp.dropna(subset=['Track ID'])
gfp['Track ID'] = gfp['Track ID'].astype(int)
gfp['Track.ID'] = 'A2_' + gfp['Track ID'].astype(str)

gfp_pos_lookup = {
    tid: dict(zip(grp['Frame'], zip(grp['X'], grp['Y'])))
    for tid, grp in gfp.groupby('Track.ID')
}
onset_df = (gfp.sort_values('Frame')
               .groupby('Track.ID').first()
               .reset_index()[['Track.ID', 'Frame', 'X', 'Y']]
               .rename(columns={'Frame': 'onset_frame',
                                'X': 'onset_x', 'Y': 'onset_y'}))
print(f'  {len(onset_df)} GFP tracks')

# ── Bidirectional re-tracking for infected cells ───────────────────────────────
print('Bidirectional re-tracking of infected cells …')
# Each entry: list of (t_rel, radius)
inf_rel_sizes = []
n_snap_fail = n_length_fail = n_gfp_fail = 0

for _, row in onset_df.iterrows():
    onset = int(row['onset_frame'])

    snapped = snap(onset, float(row['onset_x']), float(row['onset_y']))
    if snapped is None:
        n_snap_fail += 1
        continue
    sx, sy, s_idx = snapped

    back = track_backward(onset, sx, sy, MAX_BACK)
    fwd  = track_forward(onset, sx, sy, MAX_FWD)

    if len(back) < MIN_BACK_FRAMES or len(fwd) < MIN_FWD_FRAMES:
        n_length_fail += 1
        continue

    # GFP-position validation
    seed_r = radii[onset][s_idx]
    full_pos = back + [(onset, sx, sy, seed_r)] + fwd
    gfp_frames = gfp_pos_lookup.get(row['Track.ID'], {})
    dists = [
        np.sqrt((bx - gfp_frames[f][0])**2 + (by - gfp_frames[f][1])**2)
        for (f, bx, by, _) in full_pos if f in gfp_frames
    ]
    if len(dists) < 10 or np.mean(np.array(dists) <= CELL_RADIUS_UM) < MIN_FRAC_WITHIN:
        n_gfp_fail += 1
        continue

    rel_sizes = [(f - onset, r) for f, _, _, r in full_pos]
    inf_rel_sizes.append(rel_sizes)

n_inf = len(inf_rel_sizes)
print(f'  Kept: {n_inf} infected cells')
print(f'  Dropped: {n_snap_fail} snap-miss, {n_length_fail} too short, '
      f'{n_gfp_fail} GFP-drift')

# ── Per-relative-frame stats for infected cells ────────────────────────────────
all_obs = [(t, r) for cell in inf_rel_sizes for t, r in cell]
obs_df  = pd.DataFrame(all_obs, columns=['t_rel', 'radius'])

inf_ts = (obs_df
          .groupby('t_rel')['radius']
          .agg(n='count', mean='mean', sem=lambda x: x.sem())
          .reset_index()
          .sort_values('t_rel'))
inf_ts = inf_ts[(inf_ts['t_rel'] >= PLOT_MIN) &
                (inf_ts['t_rel'] <= PLOT_MAX) &
                (inf_ts['n'] >= 5)]

# ── Non-infected reference ─────────────────────────────────────────────────────
print('\nBuilding non-infected reference …')
bf = pd.read_csv(
    BF_SPOTS, skiprows=[0, 2, 3], header=0,
    usecols=['Track ID', 'Frame', 'X', 'Y', 'Radius'],
    dtype={'Track ID': float, 'Frame': int, 'X': float, 'Y': float, 'Radius': float},
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
non_radii = bf_non['Radius'].dropna()
non_mean = float(non_radii.mean())
non_sem  = float(non_radii.sem())
n_non    = len(non_inf_ids)
print(f'  Non-infected radius: mean={non_mean:.2f} ± {non_sem:.3f} µm')

# ── Pre-onset vs post-onset per infected cell ──────────────────────────────────
pre_means  = []
post_means = []
for cell in inf_rel_sizes:
    pre  = [r for t, r in cell if t < 0]
    post = [r for t, r in cell if t >= 0]
    if pre and post:
        pre_means.append(np.mean(pre))
        post_means.append(np.mean(post))

pre_means  = np.array(pre_means)
post_means = np.array(post_means)
_, p_wilcoxon = wilcoxon(pre_means, post_means)
_, p_mw = mannwhitneyu(pre_means, post_means, alternative='two-sided')
print(f'\nPre-onset vs post-onset radius per cell (n={len(pre_means)}):')
print(f'  pre  mean={pre_means.mean():.3f}  median={np.median(pre_means):.3f} µm')
print(f'  post mean={post_means.mean():.3f}  median={np.median(post_means):.3f} µm')
print(f'  Wilcoxon p={p_wilcoxon:.4g}  Mann-Whitney p={p_mw:.4g}')

# ── Figure 1: onset-aligned radius trace ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))

# Non-infected reference band
ax.axhspan(non_mean - non_sem, non_mean + non_sem,
           color='#5b8be0', alpha=0.2, zorder=1)
ax.axhline(non_mean, color='#5b8be0', lw=1.5, linestyle='--',
           label=f'Non-infected mean ± SEM (n={n_non})', zorder=2)

# Infected onset-aligned trace
d = inf_ts
ax.plot(d['t_rel'], d['mean'], color='#e05b5b', lw=2,
        label=f'Infected (n={n_inf})', zorder=4)
ax.fill_between(d['t_rel'],
                d['mean'] - d['sem'],
                d['mean'] + d['sem'],
                color='#e05b5b', alpha=0.25, zorder=3)

# Onset line
ax.axvline(0, color='black', lw=1.2, linestyle=':', zorder=5)
y_label = inf_ts['mean'].max() * 1.02
ax.text(1, y_label, 'GFP onset', rotation=90, va='bottom', ha='left',
        fontsize=8, color='black', zorder=6)

# x-axis ticks with hour labels
frame_ticks = np.arange(PLOT_MIN, PLOT_MAX + 1, 20)
hour_labels  = [f'{t}\n({t*15/60:+.0f}h)' for t in frame_ticks]
ax.set_xticks(frame_ticks)
ax.set_xticklabels(hour_labels, fontsize=7)

ax.set_xlabel('Frame relative to GFP onset  (frame 0 = GFP onset)')
ax.set_ylabel('Mean BF spot radius (µm)')
ax.set_title('Cell size (BF radius) aligned to GFP onset: infected vs non-infected (A2)')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
out1 = OUT_DIR / 'bf_size_onset_aligned.png'
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f'\nSaved: {out1.name}')

# ── Figure 2: paired pre vs post per cell ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 5))
rng = np.random.default_rng(42)

for xi, (vals, label) in enumerate([(pre_means, 'Pre-onset'), (post_means, 'Post-onset')]):
    parts = ax.violinplot(vals, positions=[xi], widths=0.55,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#e05b5b'); pc.set_alpha(0.45)
    parts['cmedians'].set_color('black')
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(xi + jitter, vals, s=6, alpha=0.4, color='#e05b5b', zorder=3)

ax.plot([0, 1], [pre_means.mean(), post_means.mean()],
        color='black', lw=2, zorder=5)

y_top = max(pre_means.max(), post_means.max()) * 1.07
ax.plot([0, 0, 1, 1], [y_top*0.93, y_top, y_top, y_top*0.93], lw=1, color='black')
p_str = f'p={p_wilcoxon:.2e}' if p_wilcoxon < 0.001 else f'p={p_wilcoxon:.3f}'
ax.text(0.5, y_top * 1.01, f'Wilcoxon {p_str}', ha='center', va='bottom', fontsize=9)

ax.set_xticks([0, 1])
ax.set_xticklabels([f'Pre-onset\n(n={len(pre_means)})',
                    f'Post-onset\n(n={len(post_means)})'])
ax.set_ylabel('Mean BF spot radius (µm)')
ax.set_title('Infected cells: size before vs after GFP onset')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
out2 = OUT_DIR / 'bf_size_onset_aligned_halves.png'
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f'Saved: {out2.name}')

print('\nDone.')

"""
03_bf_bidir_retrack.py

For every GFP-positive cell in the A2 movie:
  1. Snap to the nearest BF detection at the GFP onset frame.
  2. Track BACKWARD (strict NN) frame-by-frame toward frame 0.
  3. Track FORWARD (strict NN) frame-by-frame toward the last frame.
  4. Keep the cell only if it survives ≥ MIN_BACK_FRAMES backward
     AND ≥ MIN_FWD_FRAMES forward without any ambiguous link.
  5. Show the full confirmed trajectory from wherever the backward trace
     reaches (ideally frame 0) through to where the forward trace ends.

Linking criteria (same for both directions):
  MAX_RADIUS      – nearest BF detection must be within this distance (µm)
  AMBIGUITY_RATIO – 2nd-nearest must be >= ratio × nearest; else stop
  MIN_BACK_FRAMES – minimum backward frames required to keep the cell
  MIN_FWD_FRAMES  – minimum forward frames required to keep the cell

Non-infected cells are re-tracked forward-only from their BF track's
first frame (same strict criteria) as a reference group.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
ALLSPOTS   = ROOT / "CompleteImage" / "A2_BrightField_allspots.csv"
BF_SPOTS   = ROOT / "CompleteImage" / "A2_BrightField_spots.csv"
GFP_SPOTS  = ROOT / "CompleteImage" / "A2_Merged_spots.csv"
GFP_MATCHES = ROOT / "BrightFieldEmbedding" / "bf_gfp_matches.csv"
OUT_DIR    = Path(__file__).resolve().parent

# ── parameters ─────────────────────────────────────────────────────────────
MAX_RADIUS       = 15.0   # µm
AMBIGUITY_RATIO  = 2.0
MIN_BACK_FRAMES  = 10     # min frames tracked backward from onset
MIN_FWD_FRAMES   = 10     # min frames tracked forward from onset
MIN_TOTAL_FRAMES = 80     # min total track length (back + 1 + fwd)

# ── load BF detections and build per-frame KD-trees ────────────────────────
print("Loading all BF detections …")
allspots = pd.read_csv(
    ALLSPOTS, skiprows=[0, 2, 3], header=0,
    usecols=["Frame", "X", "Y"],
    dtype={"Frame": int, "X": float, "Y": float},
)
print(f"  {len(allspots):,} detections, {allspots['Frame'].nunique()} frames")

print("Building per-frame KD-trees …")
coords = {}
trees  = {}
for f, grp in allspots.groupby("Frame"):
    xy = grp[["X", "Y"]].values
    coords[f] = xy
    trees[f]  = cKDTree(xy)
MAX_FRAME = max(coords)
MIN_FRAME = min(coords)

# ── strict NN tracker helpers ───────────────────────────────────────────────
def _step(x, y, frame):
    """
    Return (new_x, new_y) if the step to `frame` is unambiguous,
    else return None.
    """
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
    return tuple(coords[frame][idxs[0]])

def snap(frame, x, y):
    """Snap (x, y) to the nearest BF detection at frame; None if >MAX_RADIUS."""
    if frame not in trees:
        return None
    d, i = trees[frame].query([x, y], k=1)
    return tuple(coords[frame][i]) if d <= MAX_RADIUS else None

def track_forward(seed_frame, seed_x, seed_y):
    """Track frame+1, frame+2, … Return list of (frame, x, y) after seed."""
    pos = []
    x, y = seed_x, seed_y
    for f in range(seed_frame + 1, MAX_FRAME + 1):
        step = _step(x, y, f)
        if step is None:
            break
        x, y = step
        pos.append((f, x, y))
    return pos

def track_backward(seed_frame, seed_x, seed_y):
    """Track frame-1, frame-2, … Return list of (frame, x, y) before seed,
    in chronological order (earliest first)."""
    pos = []
    x, y = seed_x, seed_y
    for f in range(seed_frame - 1, MIN_FRAME - 1, -1):
        step = _step(x, y, f)
        if step is None:
            break
        x, y = step
        pos.append((f, x, y))
    pos.reverse()
    return pos

def positions_to_speed(positions):
    """(frame, x, y) list → DataFrame(Frame, speed µm/frame)."""
    if len(positions) < 2:
        return pd.DataFrame(columns=["Frame", "speed"])
    arr    = np.array(positions)
    dx, dy = np.diff(arr[:, 1]), np.diff(arr[:, 2])
    return pd.DataFrame({"Frame": arr[1:, 0].astype(int),
                         "speed": np.sqrt(dx**2 + dy**2)})

# ── GFP positions: onset + per-frame lookup for validation ─────────────────
print("\nLoading GFP track positions …")
gfp = pd.read_csv(GFP_SPOTS, low_memory=False,
                  usecols=["Track ID", "Frame", "X", "Y"])
gfp = gfp.dropna(subset=["Track ID"])
gfp["Track ID"] = gfp["Track ID"].astype(int)
gfp["Track.ID"] = "A2_" + gfp["Track ID"].astype(str)

# per-track GFP position lookup: {track_id: {frame: (x, y)}}
gfp_pos_lookup = {
    tid: dict(zip(grp["Frame"], zip(grp["X"], grp["Y"])))
    for tid, grp in gfp.groupby("Track.ID")
}

# first frame of each track = onset (per project segmentation convention)
onset_df = (gfp.sort_values("Frame")
               .groupby("Track.ID")
               .first()
               .reset_index()[["Track.ID", "Frame", "X", "Y"]]
               .rename(columns={"Frame": "onset_frame",
                                "X": "onset_x", "Y": "onset_y"}))
print(f"  {len(onset_df)} GFP tracks")

CELL_RADIUS      = 20.5   # µm — typical BF cell radius
MIN_FRAC_WITHIN  = 0.90   # fraction of shared frames that must stay within 1 cell radius

# ── bidirectional re-tracking for all GFP cells ─────────────────────────────
print("Bidirectional re-tracking of GFP-positive cells …")
inf_records = []
n_length_fail = 0
n_gfp_fail    = 0

for _, row in onset_df.iterrows():
    onset = int(row["onset_frame"])

    # snap GFP onset position to nearest BF detection
    bf_seed = snap(onset, float(row["onset_x"]), float(row["onset_y"]))
    if bf_seed is None:
        continue
    sx, sy = bf_seed

    back = track_backward(onset, sx, sy)
    fwd  = track_forward(onset, sx, sy)

    total = len(back) + 1 + len(fwd)
    if len(back) < MIN_BACK_FRAMES or len(fwd) < MIN_FWD_FRAMES or total < MIN_TOTAL_FRAMES:
        n_length_fail += 1
        continue

    # GFP-position validation: BF track must stay close to GFP centroid
    full_pos = back + [(onset, sx, sy)] + fwd
    gfp_frames = gfp_pos_lookup.get(row["Track.ID"], {})
    dists = [
        np.sqrt((bx - gfp_frames[f][0])**2 + (by - gfp_frames[f][1])**2)
        for (f, bx, by) in full_pos if f in gfp_frames
    ]
    if len(dists) < 10 or np.mean(np.array(dists) <= CELL_RADIUS) < MIN_FRAC_WITHIN:
        n_gfp_fail += 1
        continue

    spd = positions_to_speed(full_pos)
    spd["track_id"] = row["Track.ID"]
    spd["group"]    = "infected"
    inf_records.append(spd)

inf_df = pd.concat(inf_records, ignore_index=True) if inf_records else pd.DataFrame()
n_inf  = len(inf_records)
print(f"  {n_inf} infected cells kept after length + GFP-validation filter")
print(f"  (dropped {n_length_fail} for track length, {n_gfp_fail} for GFP drift)")

# ── non-infected: forward-only from BF track start ──────────────────────────
print("Re-tracking non-infected cells …")
bf = pd.read_csv(
    BF_SPOTS, skiprows=[0, 2, 3], header=0,
    usecols=["Track ID", "Frame", "X", "Y"],
    dtype={"Track ID": float, "Frame": int, "X": float, "Y": float},
)
bf = bf.dropna(subset=["Track ID"])
bf["Track ID"] = bf["Track ID"].astype(int)

# quality filter: ≥30 frames, ≥90% coverage
stats = bf.groupby("Track ID")["Frame"].agg(
    n_frames="count", frame_min="min", frame_max="max")
stats["frame_span"] = stats["frame_max"] - stats["frame_min"] + 1
stats["coverage"]   = stats["n_frames"] / stats["frame_span"]
good_ids = set(stats.index[(stats["n_frames"] >= 30) & (stats["coverage"] >= 0.90)])

# exclude any BF track that was matched to a GFP cell (any tier)
matches = pd.read_csv(GFP_MATCHES)
all_matched_bf = set(matches["bf_track_id"].dropna().astype(int))
non_inf_ids = good_ids - all_matched_bf

first_pos = (bf[bf["Track ID"].isin(non_inf_ids)]
             .sort_values("Frame")
             .groupby("Track ID").first()
             .reset_index()[["Track ID", "Frame", "X", "Y"]])

non_records = []
for _, row in first_pos.iterrows():
    fwd = track_forward(int(row["Frame"]), float(row["X"]), float(row["Y"]))
    if 1 + len(fwd) < MIN_TOTAL_FRAMES:
        continue
    full_pos = [(int(row["Frame"]), float(row["X"]), float(row["Y"]))] + fwd
    spd = positions_to_speed(full_pos)
    spd["track_id"] = int(row["Track ID"])
    spd["group"]    = "non_infected"
    non_records.append(spd)

non_df = pd.concat(non_records, ignore_index=True) if non_records else pd.DataFrame()
n_non  = len(non_records)
print(f"  {n_non} non-infected cells kept")

# ── per-track summary & Mann-Whitney ───────────────────────────────────────
print("\nPer-track mean speed summary:")
inf_mean = non_mean = None
if not inf_df.empty:
    inf_mean = inf_df[inf_df["speed"] > 0].groupby("track_id")["speed"].mean()
    print(f"  Infected    : {n_inf} cells, mean {inf_mean.mean():.3f} µm/frame")
if not non_df.empty:
    non_mean = non_df[non_df["speed"] > 0].groupby("track_id")["speed"].mean()
    print(f"  Non-infected: {n_non} cells, mean {non_mean.mean():.3f} µm/frame")
if inf_mean is not None and non_mean is not None:
    stat, pval = mannwhitneyu(inf_mean, non_mean, alternative="two-sided")
    print(f"  Mann-Whitney U={stat:.0f}, p={pval:.4g}")

# ── per-frame time series ───────────────────────────────────────────────────
SMOOTH_WIN = 15   # rolling-average window in frames

def time_series(df):
    ts = (df[df["speed"] > 0]
          .groupby("Frame")["speed"]
          .agg(n="count", mean="mean", sem=lambda x: x.sem())
          .reset_index()
          .sort_values("Frame"))
    # rolling smooth on a complete frame index so gaps don't shift the window
    full_idx = pd.RangeIndex(ts["Frame"].min(), ts["Frame"].max() + 1)
    ts = (ts.set_index("Frame")
            .reindex(full_idx)
            .rolling(SMOOTH_WIN, center=True, min_periods=SMOOTH_WIN // 2)
            .mean()
            .reset_index()
            .rename(columns={"index": "Frame"}))
    return ts

inf_ts = time_series(inf_df) if not inf_df.empty else pd.DataFrame()
non_ts = time_series(non_df) if not non_df.empty else pd.DataFrame()

# ── GFP segmentation speed for the same 87 surviving infected cells ─────────
surviving_gfp_ids = set(inf_df["track_id"].unique()) if not inf_df.empty else set()
gfp_speed = pd.read_csv(ROOT / "cache" / "python_export" / "extra_features.csv",
                        usecols=["Track.ID", "Frame", "speed_px_per_frame"])
gfp_speed = gfp_speed[gfp_speed["Track.ID"].isin(surviving_gfp_ids) &
                      (gfp_speed["speed_px_per_frame"] > 0)].copy()
gfp_speed = gfp_speed.rename(columns={"Track.ID": "track_id",
                                       "speed_px_per_frame": "speed"})
gfp_ts = time_series(gfp_speed) if not gfp_speed.empty else pd.DataFrame()
n_gfp  = gfp_speed["track_id"].nunique()

# ── helper to plot one group ────────────────────────────────────────────────
def plot_group(ax, ts, color, label, linestyle="-"):
    if ts.empty:
        return
    d = ts.dropna(subset=["mean"])
    ax.plot(d["Frame"], d["mean"], color=color, lw=1.5,
            label=label, linestyle=linestyle)
    ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                    color=color, alpha=0.2)

# ── plot 1: BF only (no GFP overlay) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_group(ax, non_ts, "#5b8be0", f"Non-infected (n={n_non})")
plot_group(ax, inf_ts, "#e05b5b", f"Infected (n={n_inf})")
ax.set_xlabel(f"Frame  ({SMOOTH_WIN}-frame rolling mean ± SEM)")
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title("BF re-tracked speed: infected vs non-infected (A2)")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_bidir_retracked.png", dpi=150)
plt.close(fig)
print("Saved: bf_speed_bidir_retracked.png")

# ── plot 2: same + GFP segmentation overlay ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_group(ax, non_ts, "#5b8be0", f"BF non-infected (n={n_non})")
plot_group(ax, inf_ts, "#e05b5b", f"BF infected (n={n_inf})")
plot_group(ax, gfp_ts, "#2ca02c", f"GFP segmentation (n={n_gfp})", linestyle="--")
ax.set_xlabel(f"Frame  ({SMOOTH_WIN}-frame rolling mean ± SEM)")
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title("BF re-tracked speed + GFP segmentation overlay (A2)")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_bidir_retracked_with_gfp.png", dpi=150)
plt.close(fig)
print("Saved: bf_speed_bidir_retracked_with_gfp.png")

# ── plot 3: paired slope — first-half vs second-half speed per cell ──────────
from scipy.stats import wilcoxon

def half_speeds(df, id_col="track_id"):
    """Per-cell first-half / second-half mean speed from a speed DataFrame."""
    rows = []
    for tid, grp in df[df["speed"] > 0].groupby(id_col):
        frames = grp["Frame"].values
        mid    = (frames.min() + frames.max()) / 2
        h1 = grp.loc[grp["Frame"] <= mid, "speed"].mean()
        h2 = grp.loc[grp["Frame"] >  mid, "speed"].mean()
        if np.isfinite(h1) and np.isfinite(h2):
            rows.append({"id": tid, "first": h1, "second": h2})
    return pd.DataFrame(rows)

inf_h  = half_speeds(inf_df)
non_h  = half_speeds(non_df)

fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharey=True)

for ax, halves, color, title in [
    (axes[0], non_h, "#5b8be0", f"Non-infected\n(n={len(non_h)})"),
    (axes[1], inf_h, "#e05b5b", f"Infected\n(n={len(inf_h)})"),
]:
    # individual cell lines
    for _, row in halves.iterrows():
        ax.plot([0, 1], [row["first"], row["second"]],
                color=color, alpha=0.25, lw=0.8)

    # group means ± SEM
    for xi, col in [(0, "first"), (1, "second")]:
        m = halves[col].mean()
        s = halves[col].sem()
        ax.errorbar(xi, m, yerr=s, fmt="o", color=color,
                    markersize=8, capsize=4, lw=2, zorder=5)

    # connect group means
    ax.plot([0, 1], [halves["first"].mean(), halves["second"].mean()],
            color=color, lw=2.5, zorder=4)

    # Wilcoxon p-value
    _, pval = wilcoxon(halves["first"], halves["second"])
    p_str = f"p={pval:.2e}" if pval < 0.001 else f"p={pval:.3f}"
    y_top = max(halves[["first","second"]].max()) * 1.07
    ax.plot([0, 0, 1, 1],
            [y_top*0.94, y_top, y_top, y_top*0.94], lw=1, color="black")
    ax.text(0.5, y_top*1.01, p_str, ha="center", va="bottom", fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["First half", "Second half"])
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Mean speed (µm / frame)")
fig.suptitle("Within-cell speed decline: first half vs second half of track (A2)",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_halves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: bf_speed_halves.png")

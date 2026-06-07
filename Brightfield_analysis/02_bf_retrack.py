"""
02_bf_retrack.py

Re-tracks cells directly from raw BF spot detections using strict
frame-to-frame nearest-neighbour (NN) linking. Stops tracking a cell
the moment any link is ambiguous or out of range, and omits the cell
if the resulting confident track is too short.

Infected cells  – seeded at the GFP onset position (bf_gfp_matches.csv).
Non-infected    – seeded at the first-frame position of each TrackMate
                  BF track that has no confident GFP match.

Linking criteria (applied at every frame transition):
  MAX_RADIUS      – nearest detection must be within this distance (µm)
  AMBIGUITY_RATIO – 2nd-nearest must be >= ratio × nearest distance;
                    if not, the link is ambiguous → stop tracking
  MIN_TRACK_FRAMES – minimum consecutive confident frames to keep a cell
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
ALLSPOTS   = ROOT / "CompleteImage" / "A2_BrightField_allspots.csv"
BF_SPOTS   = ROOT / "CompleteImage" / "A2_BrightField_spots.csv"
GFP_MATCHES = ROOT / "BrightFieldEmbedding" / "bf_gfp_matches.csv"
GFP_SPEED  = ROOT / "cache" / "python_export" / "extra_features.csv"
OUT_DIR    = Path(__file__).resolve().parent

# ── parameters ─────────────────────────────────────────────────────────────
MAX_RADIUS       = 15.0   # µm — max allowed single-frame displacement
AMBIGUITY_RATIO  = 2.0    # 2nd-nearest must be >= 2× the nearest distance
MIN_TRACK_FRAMES = 20     # minimum confident frames required to keep a cell

# ── load all BF detections and build per-frame KD-trees ────────────────────
print("Loading all BF detections …")
allspots = pd.read_csv(
    ALLSPOTS,
    skiprows=[0, 2, 3], header=0,
    usecols=["Frame", "X", "Y"],
    dtype={"Frame": int, "X": float, "Y": float},
)
print(f"  {len(allspots):,} detections, {allspots['Frame'].nunique()} frames")

print("Building per-frame KD-trees …")
frames_sorted = sorted(allspots["Frame"].unique())
trees   = {}   # frame → cKDTree
coords  = {}   # frame → (N,2) array
for f, grp in allspots.groupby("Frame"):
    xy = grp[["X", "Y"]].values
    coords[f] = xy
    trees[f]  = cKDTree(xy)
MAX_FRAME = max(frames_sorted)

# ── strict NN tracker ───────────────────────────────────────────────────────
def track_from(start_frame, start_x, start_y):
    """
    Track forward from (start_frame, start_x, start_y).
    Returns list of (frame, x, y); may be empty if start frame is missing.
    Stops at the first ambiguous or out-of-range link.
    """
    positions = []
    if start_frame not in trees:
        return positions

    # Snap to nearest detection at start frame (tolerance = MAX_RADIUS)
    d0, i0 = trees[start_frame].query([start_x, start_y], k=1)
    if d0 > MAX_RADIUS:
        return positions
    x, y = coords[start_frame][i0]
    positions.append((start_frame, x, y))

    for frame in range(start_frame + 1, MAX_FRAME + 1):
        if frame not in trees:
            break
        k = min(2, len(coords[frame]))
        result = trees[frame].query([x, y], k=k)
        dists = np.atleast_1d(result[0])
        idxs  = np.atleast_1d(result[1])

        d1 = dists[0]
        # Condition 1: nearest within search radius
        if d1 > MAX_RADIUS:
            break
        # Condition 2: unambiguous (second candidate is far enough away)
        if len(dists) > 1 and dists[1] < AMBIGUITY_RATIO * d1:
            break

        x, y = coords[frame][idxs[0]]
        positions.append((frame, x, y))

    return positions

# ── speed from a position list ──────────────────────────────────────────────
def positions_to_speed(positions):
    """Return DataFrame with columns Frame, speed (µm/frame)."""
    if len(positions) < 2:
        return pd.DataFrame(columns=["Frame", "speed"])
    arr = np.array(positions)   # (N, 3): frame, x, y
    dx  = np.diff(arr[:, 1])
    dy  = np.diff(arr[:, 2])
    spd = np.sqrt(dx**2 + dy**2)
    frames = arr[1:, 0].astype(int)
    return pd.DataFrame({"Frame": frames, "speed": spd})

# ── load BF tracks (needed for both groups) ────────────────────────────────
bf = pd.read_csv(
    BF_SPOTS,
    skiprows=[0, 2, 3], header=0,
    usecols=["Track ID", "Frame", "X", "Y"],
    dtype={"Track ID": float, "Frame": int, "X": float, "Y": float},
)
bf = bf.dropna(subset=["Track ID"])
bf["Track ID"] = bf["Track ID"].astype(int)

# ── load GFP matches and seed infected cells from BF track start ────────────
# Seeding from the BF track's first detected frame puts both groups in
# absolute movie time so they can be compared from frame 0.
print("\nRe-tracking infected cells from BF track start (absolute time) …")
matches = pd.read_csv(GFP_MATCHES)
matches["is_ambiguous"] = matches["is_ambiguous"].map(
    {"True": True, "False": False, True: True, False: False})

conf = matches[
    (matches["match_tier"] == "confident") &
    (matches["is_ambiguous"] == False)
].copy()
conf = conf.dropna(subset=["bf_track_id"])
conf["bf_track_id"] = conf["bf_track_id"].astype(int)
conf["gfp_track_id_str"] = "A2_" + conf["gfp_track_id"].astype(int).astype(str)

# First-frame BF position for each matched infected track
bf_first = (bf[bf["Track ID"].isin(conf["bf_track_id"])]
            .sort_values("Frame")
            .groupby("Track ID")
            .first()
            .reset_index()[["Track ID", "Frame", "X", "Y"]])
bf_first_map = bf_first.set_index("Track ID")[["Frame", "X", "Y"]].to_dict("index")

inf_records = []
for _, row in conf.iterrows():
    tid = int(row["bf_track_id"])
    if tid not in bf_first_map:
        continue
    seed = bf_first_map[tid]
    pos  = track_from(int(seed["Frame"]), float(seed["X"]), float(seed["Y"]))
    if len(pos) < MIN_TRACK_FRAMES:
        continue
    spd_df = positions_to_speed(pos)
    spd_df["gfp_track_id"] = row["gfp_track_id_str"]
    spd_df["group"]         = "infected"
    inf_records.append(spd_df)

inf_df = pd.concat(inf_records, ignore_index=True) if inf_records else pd.DataFrame()
n_inf  = len(inf_records)
print(f"  {n_inf} infected cells with ≥{MIN_TRACK_FRAMES} confident frames")

# ── seed non-infected cells ─────────────────────────────────────────────────
print("Re-tracking non-infected cells from BF track start …")

# Quality filter matching the main script (≥30 frames, ≥90% coverage)
stats = bf.groupby("Track ID")["Frame"].agg(
    n_frames="count", frame_min="min", frame_max="max")
stats["frame_span"] = stats["frame_max"] - stats["frame_min"] + 1
stats["coverage"]   = stats["n_frames"] / stats["frame_span"]
good_ids = stats.index[(stats["n_frames"] >= 30) & (stats["coverage"] >= 0.90)]

# Exclude BF tracks matched to any GFP cell (all confidence tiers)
all_matched_bf = set(
    matches["bf_track_id"].dropna().astype(int)
)
non_inf_ids = [t for t in good_ids if t not in all_matched_bf]

# Seed from first-frame position of each non-infected track
first_pos = (bf[bf["Track ID"].isin(non_inf_ids)]
             .sort_values("Frame")
             .groupby("Track ID")
             .first()
             .reset_index()[["Track ID", "Frame", "X", "Y"]])

non_records = []
for _, row in first_pos.iterrows():
    pos = track_from(int(row["Frame"]), float(row["X"]), float(row["Y"]))
    if len(pos) < MIN_TRACK_FRAMES:
        continue
    spd_df = positions_to_speed(pos)
    spd_df["track_id"] = int(row["Track ID"])
    spd_df["group"]    = "non_infected"
    non_records.append(spd_df)

non_df = pd.concat(non_records, ignore_index=True) if non_records else pd.DataFrame()
n_non  = len(non_records)
print(f"  {n_non} non-infected cells with ≥{MIN_TRACK_FRAMES} confident frames")

# ── per-frame time series ───────────────────────────────────────────────────
def time_series(df, speed_col="speed"):
    return (df[df[speed_col] > 0]
            .groupby("Frame")[speed_col]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index())

inf_ts = time_series(inf_df) if not inf_df.empty else pd.DataFrame()
non_ts = time_series(non_df) if not non_df.empty else pd.DataFrame()

# ── per-track summary & stats ───────────────────────────────────────────────
print("\nPer-track mean speed summary:")
if not inf_df.empty:
    inf_mean = inf_df[inf_df["speed"] > 0].groupby("gfp_track_id")["speed"].mean()
    print(f"  Re-tracked infected  : {n_inf} cells, mean {inf_mean.mean():.3f} µm/frame")
if not non_df.empty:
    non_mean = non_df[non_df["speed"] > 0].groupby("track_id")["speed"].mean()
    print(f"  Re-tracked non-inf.  : {n_non} cells, mean {non_mean.mean():.3f} µm/frame")

if not inf_df.empty and not non_df.empty:
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(inf_mean, non_mean, alternative="two-sided")
    print(f"  Mann-Whitney U: U={stat:.0f}, p={pval:.4g}")

# ── plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))

if not non_ts.empty:
    d = non_ts.sort_values("Frame")
    ax.plot(d["Frame"], d["mean"], color="#5b8be0", lw=1.5,
            label=f"Non-infected (n={n_non})")
    ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                    color="#5b8be0", alpha=0.2)

if not inf_ts.empty:
    d = inf_ts.sort_values("Frame")
    ax.plot(d["Frame"], d["mean"], color="#e05b5b", lw=1.5,
            label=f"Infected (n={n_inf})")
    ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                    color="#e05b5b", alpha=0.2)

ax.set_xlabel("Frame")
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title(f"BF re-tracked movement speed: infected vs non-infected (A2)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out_path = OUT_DIR / "bf_speed_retracked.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {out_path}")

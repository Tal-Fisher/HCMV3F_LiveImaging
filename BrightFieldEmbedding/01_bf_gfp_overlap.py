#!/usr/bin/env python3
"""
01_bf_gfp_overlap.py

Validate spatial overlap between brightfield (BF) TrackMate tracks and
GFP-channel TrackMate tracks.

NOTE: Uses A2_BrightField_allspots.csv (not _spots.csv) because the
_spots.csv only includes tracks passing TrackMate's quality/duration
filter, which excludes many short-lived BF tracks that correspond to
GFP-positive cells at their onset frame.

Normalisation uses the BF allspot radius (not the GFP onset radius),
because the GFP signal at onset is often faint and the detection
circle is artificially small.

Matching logic:
  1. At each GFP cell's onset frame, find the nearest BF allspot centroid.
  2. Classify by dist / r_bf:
       confident  < 0.5
       plausible  < 1.0
       marginal   < 1.5
       unmatched  >= 1.5
  3. For confident unambiguous matches (with a TRACK_ID in allspots),
     verify the pairing holds at onset+10, onset+25, onset+50 frames.
  4. For consistent matches, check whether the BF track reaches frame 0.

Outputs: bf_gfp_matches.csv, summary.txt, figures/01-04_*.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path("/home/labs/ginossar/talfis/LiveImaging")
COMPLETE = ROOT / "CompleteImage"
OUT_DIR  = ROOT / "BrightFieldEmbedding"
FIG_DIR  = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── thresholds ─────────────────────────────────────────────────────────────────
STRICT_THR   = 0.5
MODERATE_THR = 1.0
LOOSE_THR    = 1.5
TEMPORAL_THR = 1.0
CHECK_OFFSETS = [10, 25, 50]   # frames after onset
MIN_CHECKS    = 2              # minimum valid checks for temporal decision

# ── 1. GFP onset positions ─────────────────────────────────────────────────────
print("Loading GFP onset data ...")
onset = pd.read_csv(COMPLETE / "A2_gfp_onset.csv")
onset = onset.rename(columns={"track_id": "gfp_track_id",
                               "gfp_onset_frame": "onset_frame"})
print(f"  GFP cells: {len(onset)}")

# ── 2. GFP merged spots (radius + full trajectory for temporal checks) ─────────
print("Loading GFP merged spots ...")
gfp_raw = pd.read_csv(COMPLETE / "A2_Merged_spots.csv", low_memory=False)
gfp_raw.columns = gfp_raw.columns.str.strip()
gfp_raw = gfp_raw.rename(columns={
    "Track ID": "gfp_track_id", "Frame": "frame",
    "X": "x", "Y": "y", "R": "r"
})
for col in ["gfp_track_id", "frame", "x", "y", "r"]:
    gfp_raw[col] = pd.to_numeric(gfp_raw[col], errors="coerce")
gfp_raw = gfp_raw.dropna(subset=["gfp_track_id", "frame", "x", "y", "r"])
gfp_raw["gfp_track_id"] = gfp_raw["gfp_track_id"].astype(int)
gfp_raw["frame"] = gfp_raw["frame"].astype(int)
print(f"  GFP spots loaded: {len(gfp_raw)}")

# Add radius from merged spots to onset table
onset = onset.merge(
    gfp_raw.rename(columns={"frame": "onset_frame"})[
        ["gfp_track_id", "onset_frame", "r"]
    ],
    on=["gfp_track_id", "onset_frame"], how="left"
)
n_missing_r = onset["r"].isna().sum()
if n_missing_r:
    median_r = onset["r"].median()
    print(f"  WARNING: {n_missing_r} cells missing radius; filling with median {median_r:.2f} µm")
    onset["r"] = onset["r"].fillna(median_r)

# Build GFP lookup: (gfp_track_id, frame) → {"x", "y", "r"}
gfp_lookup = (
    gfp_raw.set_index(["gfp_track_id", "frame"])[["x", "y", "r"]]
    .to_dict("index")
)

# ── 3. BF allspots (all detections, not just quality-filtered tracks) ──────────
# _spots.csv only contains tracks passing TrackMate's filter — many BF cells
# corresponding to GFP onset events are in short tracks that get filtered out.
print("Loading BF allspots (large file, ~30s) ...")
chunks = []
for chunk in pd.read_csv(
        COMPLETE / "A2_BrightField_allspots.csv",
        usecols=["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y", "RADIUS"],
        low_memory=False, chunksize=200_000):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunks.append(chunk.dropna(subset=["FRAME", "POSITION_X", "POSITION_Y"]))
bf_raw = pd.concat(chunks, ignore_index=True)
bf_raw = bf_raw.rename(columns={
    "TRACK_ID": "bf_track_id", "FRAME": "frame",
    "POSITION_X": "x", "POSITION_Y": "y", "RADIUS": "r"
})
bf_raw["frame"] = bf_raw["frame"].astype(int)
# bf_track_id may be NaN for untracked spots — keep as float for now
print(f"  BF allspots loaded: {len(bf_raw)}")

# Build BF lookup for temporal checks: only for spots that have a track_id
bf_tracked = bf_raw.dropna(subset=["bf_track_id"]).copy()
bf_tracked["bf_track_id"] = bf_tracked["bf_track_id"].astype(int)
bf_lookup = (
    bf_tracked.set_index(["bf_track_id", "frame"])[["x", "y"]]
    .to_dict("index")
)

# Build per-frame KDTree for BF (all spots including untracked)
print("Building per-frame BF spatial index ...")
bf_frames = {}
for frame, grp in bf_raw.groupby("frame"):
    xy  = grp[["x", "y"]].values
    ids = grp["bf_track_id"].values   # may contain NaN
    rs  = grp["r"].fillna(grp["r"].median()).values
    bf_frames[frame] = (KDTree(xy), ids, rs, xy)

# ── 4. Nearest-neighbour matching at onset frame ───────────────────────────────
print("Matching GFP cells to BF cells at onset ...")
records = []
for _, row in onset.iterrows():
    gfp_id     = int(row["gfp_track_id"])
    onset_f    = int(row["onset_frame"])
    gfp_x      = float(row["x_at_onset"])
    gfp_y      = float(row["y_at_onset"])
    gfp_r      = float(row["r"])
    base = dict(gfp_track_id=gfp_id, onset_frame=onset_f,
                gfp_x=gfp_x, gfp_y=gfp_y, gfp_r=gfp_r)

    if onset_f not in bf_frames:
        records.append({**base, "bf_track_id": np.nan, "bf_x": np.nan,
                        "bf_y": np.nan, "bf_r": np.nan,
                        "dist_um_onset": np.nan, "dist_norm_onset": np.nan,
                        "match_tier": "no_bf_at_frame"})
        continue

    tree, ids, rs, xy = bf_frames[onset_f]
    dist, idx = tree.query([gfp_x, gfp_y])
    # Use BF radius for normalisation — more reliable than GFP onset radius
    # (GFP signal is faint at onset, making its detection circle artificially small)
    bf_r   = float(rs[idx])
    dist_n = dist / bf_r if bf_r > 0 else np.nan
    tier   = ("confident" if dist_n < STRICT_THR else
              "plausible"  if dist_n < MODERATE_THR else
              "marginal"   if dist_n < LOOSE_THR else "unmatched")
    bf_id_raw = ids[idx]
    bf_id = int(bf_id_raw) if not np.isnan(bf_id_raw) else np.nan

    records.append({**base,
                    "bf_track_id": bf_id,
                    "bf_x": float(xy[idx, 0]), "bf_y": float(xy[idx, 1]),
                    "bf_r": bf_r,
                    "dist_um_onset": dist, "dist_norm_onset": dist_n,
                    "match_tier": tier})

matches = pd.DataFrame(records)

# Flag ambiguous: two or more GFP cells → same BF track (at "confident" level)
# Only meaningful for spots that have a track_id (not untracked allspots)
conf_with_track = matches[(matches["match_tier"] == "confident") & matches["bf_track_id"].notna()]
bf_usage = conf_with_track.groupby("bf_track_id")["gfp_track_id"].count()
ambiguous_bf = set(bf_usage[bf_usage > 1].index)
matches["is_ambiguous"] = matches["bf_track_id"].isin(ambiguous_bf)

# strict = confident + unambiguous; temporal checks also require a track_id
strict_mask = (matches["match_tier"] == "confident") & (~matches["is_ambiguous"])
strict_trackable = strict_mask & matches["bf_track_id"].notna()

# ── 5. Temporal consistency ────────────────────────────────────────────────────
print("Checking temporal consistency for strict matches ...")
for i in range(1, 4):
    matches[f"dist_norm_check{i}"] = np.nan

for idx, row in matches[strict_mask].iterrows():
    gfp_id  = int(row["gfp_track_id"])
    onset_f = int(row["onset_frame"])
    # Skip temporal check if matched allspot has no track_id
    if pd.isna(row["bf_track_id"]):
        continue
    bf_id = int(row["bf_track_id"])

    for i, offset in enumerate(CHECK_OFFSETS, 1):
        check_f  = onset_f + offset
        gfp_key  = (gfp_id, check_f)
        bf_key   = (bf_id, check_f)
        if gfp_key not in gfp_lookup or bf_key not in bf_lookup:
            continue
        gx = gfp_lookup[gfp_key]["x"]
        gy = gfp_lookup[gfp_key]["y"]
        gr = gfp_lookup[gfp_key]["r"]
        bx = bf_lookup[bf_key]["x"]
        by = bf_lookup[bf_key]["y"]
        d  = np.sqrt((gx - bx) ** 2 + (gy - by) ** 2)
        matches.loc[idx, f"dist_norm_check{i}"] = d / gr if gr > 0 else np.nan

def _is_consistent(row):
    vals = [row[f"dist_norm_check{i}"] for i in range(1, 4)]
    valid = [v for v in vals if not np.isnan(v)]
    if len(valid) < MIN_CHECKS:
        return False
    return all(v < TEMPORAL_THR for v in valid)

matches["temporally_consistent"] = False
matches.loc[strict_mask, "temporally_consistent"] = (
    matches[strict_mask].apply(_is_consistent, axis=1)
)

# ── 6. BF back-trace to frame 0 ───────────────────────────────────────────────
# Use the tracked subset of allspots to find the earliest frame per BF track
bf_earliest = bf_tracked.groupby("bf_track_id")["frame"].min()
matches["bf_earliest_frame"] = matches["bf_track_id"].map(bf_earliest)

# ── Save matches CSV ───────────────────────────────────────────────────────────
out_csv = OUT_DIR / "bf_gfp_matches.csv"
matches.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

# ── Summary ────────────────────────────────────────────────────────────────────
n_total       = len(matches)
n_confident   = (matches["match_tier"] == "confident").sum()
n_strict      = strict_mask.sum()
n_strict_trk  = strict_trackable.sum()
n_moderate    = (matches["match_tier"] == "plausible").sum()
n_marginal    = (matches["match_tier"] == "marginal").sum()
n_unmatched   = (matches["match_tier"].isin(["unmatched", "no_bf_at_frame"])).sum()
n_ambiguous   = matches["is_ambiguous"].sum()
n_no_track    = strict_mask.sum() - strict_trackable.sum()
n_temporal    = matches["temporally_consistent"].sum()

consist_mask = strict_mask & matches["temporally_consistent"]
n_frame0    = (matches.loc[consist_mask, "bf_earliest_frame"] == 0).sum()
n_frame5    = (matches.loc[consist_mask, "bf_earliest_frame"] <= 5).sum()
n_consist   = consist_mask.sum()

summary_lines = [
    f"GFP cells total:                  {n_total}",
    f"Confident matches (onset):        {n_confident} ({100*n_confident/n_total:.1f}%)",
    f"  of which unambiguous (strict):  {n_strict} ({100*n_strict/n_total:.1f}%)",
    f"    with BF track_id:             {n_strict_trk} (temporal checks possible)",
    f"    untracked allspot only:       {n_no_track} (no temporal check)",
    f"  of which temporally consistent: {n_temporal} ({100*n_temporal/max(n_strict_trk,1):.1f}% of trackable strict)",
    f"Plausible matches:                {n_moderate} ({100*n_moderate/n_total:.1f}%)",
    f"Marginal matches:                 {n_marginal} ({100*n_marginal/n_total:.1f}%)",
    f"Unmatched:                        {n_unmatched} ({100*n_unmatched/n_total:.1f}%)",
    f"Ambiguous (confident tier):       {n_ambiguous}",
    "",
    f"Of {n_consist} strict+consistent matches:",
    f"  BF track starts at frame 0:     {n_frame0} ({100*n_frame0/max(n_consist,1):.1f}%)",
    f"  BF track starts at frame ≤5:    {n_frame5} ({100*n_frame5/max(n_consist,1):.1f}%)",
]
for line in summary_lines:
    print(line)

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write("\n".join(summary_lines) + "\n")

# ── Figure 1: Distance histogram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
valid_d = matches["dist_norm_onset"].dropna()
ax.hist(valid_d, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
for thr, col, lbl in [(STRICT_THR, "green", "strict 0.5"),
                       (MODERATE_THR, "orange", "moderate 1.0"),
                       (LOOSE_THR, "red", "loose 1.5")]:
    ax.axvline(thr, color=col, linewidth=1.5, linestyle="--", label=lbl)
    pct = (valid_d < thr).mean() * 100
    ax.text(thr + 0.03, ax.get_ylim()[1] * 0.95, f"{pct:.0f}%", color=col,
            fontsize=8, va="top")
ax.set_xlabel("Distance (normalised by BF radius)")
ax.set_ylabel("Count")
ax.set_title("BF–GFP centroid distance at GFP onset frame (using allspots)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_distance_histogram.png", dpi=150)
plt.close(fig)
print("Saved 01_distance_histogram.png")

# ── Figure 2: Centroid scatter ─────────────────────────────────────────────────
tier_colors = {"confident": "green", "plausible": "gold",
               "marginal": "orange", "unmatched": "red",
               "no_bf_at_frame": "gray"}
fig, ax = plt.subplots(figsize=(8, 8))
for tier, col in tier_colors.items():
    sub = matches[matches["match_tier"] == tier]
    ax.scatter(sub["gfp_x"], sub["gfp_y"], c=col, s=10, alpha=0.6,
               label=f"{tier} (n={len(sub)})", zorder=2)

# Lines from GFP → BF centroid for strict matches
strict_rows = matches[strict_mask]
for _, row in strict_rows.iterrows():
    ax.plot([row["gfp_x"], row["bf_x"]], [row["gfp_y"], row["bf_y"]],
            color="green", alpha=0.2, linewidth=0.5, zorder=1)

ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.set_title("GFP cell positions at onset — coloured by BF match tier")
ax.legend(fontsize=8, markerscale=1.5)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_centroid_scatter.png", dpi=150)
plt.close(fig)
print("Saved 02_centroid_scatter.png")

# ── Figure 3: Temporal consistency ────────────────────────────────────────────
offsets = [0] + CHECK_OFFSETS
check_cols = ["dist_norm_onset", "dist_norm_check1", "dist_norm_check2", "dist_norm_check3"]

fig, ax = plt.subplots(figsize=(7, 5))
strict_rows = matches[strict_mask].copy()
for _, row in strict_rows.iterrows():
    vals = [row[c] for c in check_cols]
    color = "green" if row["temporally_consistent"] else "tomato"
    ax.plot(offsets, vals, color=color, alpha=0.15, linewidth=0.8)

# Medians
for label, mask_tc, col in [("consistent", strict_rows["temporally_consistent"] == True, "darkgreen"),
                              ("inconsistent", strict_rows["temporally_consistent"] == False, "darkred")]:
    sub = strict_rows[mask_tc]
    if len(sub) == 0:
        continue
    medians = [sub[c].median() for c in check_cols]
    ax.plot(offsets, medians, color=col, linewidth=2.5, label=f"{label} median (n={len(sub)})")

ax.axhline(TEMPORAL_THR, color="gray", linestyle="--", linewidth=1, label="threshold 1.0")
ax.set_xticks(offsets)
ax.set_xticklabels([f"onset\n(+0)", f"+{CHECK_OFFSETS[0]}", f"+{CHECK_OFFSETS[1]}", f"+{CHECK_OFFSETS[2]}"])
ax.set_xlabel("Frames after onset")
ax.set_ylabel("Distance (normalised by GFP radius)")
ax.set_title("BF–GFP distance over time (strict matches)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_temporal_consistency.png", dpi=150)
plt.close(fig)
print("Saved 03_temporal_consistency.png")

# ── Figure 4: BF back-trace bar ───────────────────────────────────────────────
if n_consist > 0:
    earliest = matches.loc[consist_mask, "bf_earliest_frame"]
    bins = {"frame 0": (earliest == 0).sum(),
            "frame 1–5": ((earliest > 0) & (earliest <= 5)).sum(),
            "frame 6–10": ((earliest > 5) & (earliest <= 10)).sum(),
            "frame >10": (earliest > 10).sum()}
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ca02c", "#98df8a", "#ffbb78", "#d62728"]
    bars = ax.bar(list(bins.keys()), list(bins.values()), color=colors)
    for bar, val in zip(bars, bins.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val}\n({100*val/n_consist:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Earliest BF frame for strict+consistent matches (n={n_consist})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_backtrace_bar.png", dpi=150)
    plt.close(fig)
    print("Saved 04_backtrace_bar.png")
else:
    print("Skipped 04_backtrace_bar.png (no consistent matches)")

print("Done.")

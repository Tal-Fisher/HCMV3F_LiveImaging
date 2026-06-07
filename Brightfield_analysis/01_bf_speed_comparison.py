"""
Compare movement speed of infected (GFP+) vs non-infected cells in A2 movie.

Uses brightfield TrackMate tracks for all cells, then labels infected cells
via the spatial BF-GFP matching table. Only high-quality tracks are kept.

Quality criteria:
  - MIN_FRAMES: minimum number of detected frames per track
  - MIN_COVERAGE: minimum fraction of frames filled (identity retention proxy)
    coverage = n_detected_frames / (last_frame - first_frame + 1)

Infection labeling:
  - Infected: BF track matched to a GFP track with match_tier == "confident"
    and is_ambiguous == False
  - Non-infected: all other quality-passing BF tracks
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
BF_SPOTS   = ROOT / "CompleteImage" / "A2_BrightField_spots.csv"
GFP_MATCHES = ROOT / "BrightFieldEmbedding" / "bf_gfp_matches.csv"
OUT_DIR    = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

# ── quality thresholds ─────────────────────────────────────────────────────
MIN_FRAMES   = 30    # minimum detected frames per track
MIN_COVERAGE = 0.90  # minimum frame coverage (no large gaps)

# ── load BF spots ──────────────────────────────────────────────────────────
# TrackMate CSV has 3 header lines: ALL_CAPS names (0), human-readable (1), units (2)
print("Loading BF spots …")
bf = pd.read_csv(
    BF_SPOTS,
    skiprows=[0, 2, 3],  # skip ALL_CAPS (0), duplicate header (2), units (3); row 1 → header
    header=0,
    usecols=["Track ID", "Frame", "X", "Y"],
    dtype={"Track ID": float, "Frame": int, "X": float, "Y": float},
)
bf = bf.dropna(subset=["Track ID"])
bf["Track ID"] = bf["Track ID"].astype(int)
print(f"  {len(bf):,} spot rows, {bf['Track ID'].nunique():,} unique tracks")

# ── per-track quality metrics ──────────────────────────────────────────────
stats = bf.groupby("Track ID")["Frame"].agg(
    n_frames="count",
    frame_min="min",
    frame_max="max",
)
stats["frame_span"] = stats["frame_max"] - stats["frame_min"] + 1
stats["coverage"]   = stats["n_frames"] / stats["frame_span"]

# ── quality filter ─────────────────────────────────────────────────────────
good = stats[(stats["n_frames"] >= MIN_FRAMES) & (stats["coverage"] >= MIN_COVERAGE)]
print(f"  {len(good):,} tracks pass quality filter "
      f"(≥{MIN_FRAMES} frames, ≥{MIN_COVERAGE:.0%} coverage)")

bf = bf[bf["Track ID"].isin(good.index)].copy()

# ── speed calculation ──────────────────────────────────────────────────────
bf.sort_values(["Track ID", "Frame"], inplace=True)
dx = bf.groupby("Track ID")["X"].diff()
dy = bf.groupby("Track ID")["Y"].diff()
bf["speed_um_per_frame"] = np.sqrt(dx**2 + dy**2)
bf["speed_um_per_frame"] = bf["speed_um_per_frame"].fillna(0.0)  # first frame of each track

# ── infection labeling ─────────────────────────────────────────────────────
matches = pd.read_csv(GFP_MATCHES)
matches["is_ambiguous"] = matches["is_ambiguous"].map({"True": True, "False": False, True: True, False: False})
confident_mask = (matches["match_tier"] == "confident") & (matches["is_ambiguous"] == False)
infected_bf_ids = set(matches.loc[confident_mask, "bf_track_id"].dropna().astype(int))
print(f"  {len(infected_bf_ids)} confident infected BF track IDs from GFP matching")

bf["group"] = np.where(bf["Track ID"].isin(infected_bf_ids), "infected", "non_infected")

# ── per-track mean speed (exclude first-frame zeros) ──────────────────────
per_track = (
    bf[bf["speed_um_per_frame"] > 0]
    .groupby("Track ID")
    .agg(
        mean_speed=("speed_um_per_frame", "mean"),
        group=("group", "first"),
    )
    .reset_index()
)

n_inf  = (per_track["group"] == "infected").sum()
n_non  = (per_track["group"] == "non_infected").sum()
print(f"\nPer-track summary:")
print(f"  Infected    : {n_inf} tracks  mean speed = "
      f"{per_track.loc[per_track['group']=='infected','mean_speed'].mean():.3f} µm/frame")
print(f"  Non-infected: {n_non} tracks  mean speed = "
      f"{per_track.loc[per_track['group']=='non_infected','mean_speed'].mean():.3f} µm/frame")

# ── Mann-Whitney U test ────────────────────────────────────────────────────
s_inf = per_track.loc[per_track["group"] == "infected",     "mean_speed"]
s_non = per_track.loc[per_track["group"] == "non_infected", "mean_speed"]
stat, pval = mannwhitneyu(s_inf, s_non, alternative="two-sided")
print(f"\nMann-Whitney U: U={stat:.0f}, p={pval:.4g}")

# ── save per-track table ───────────────────────────────────────────────────
out_csv = OUT_DIR / "bf_track_speeds.csv"
per_track.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── plot 1: violin + strip plot ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
colors = {"infected": "#e05b5b", "non_infected": "#5b8be0"}
order  = ["non_infected", "infected"]
labels = {"non_infected": f"Non-infected\n(n={n_non})",
          "infected":     f"Infected\n(n={n_inf})"}

for i, grp in enumerate(order):
    vals = per_track.loc[per_track["group"] == grp, "mean_speed"]
    parts = ax.violinplot(vals, positions=[i], widths=0.6, showmedians=True,
                          showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(colors[grp])
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("black")
    ax.scatter(
        np.random.default_rng(42).uniform(i - 0.15, i + 0.15, size=len(vals)),
        vals, s=8, alpha=0.4, color=colors[grp], zorder=3,
    )

# p-value annotation
y_top = per_track["mean_speed"].max() * 1.08
ax.plot([0, 0, 1, 1], [y_top * 0.93, y_top, y_top, y_top * 0.93], lw=1, color="black")
p_text = f"p={pval:.3g}" if pval >= 0.001 else f"p<0.001"
ax.text(0.5, y_top * 1.01, p_text, ha="center", va="bottom", fontsize=10)

ax.set_xticks([0, 1])
ax.set_xticklabels([labels[g] for g in order])
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title("Cell movement speed: infected vs non-infected (A2)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_comparison.png", dpi=150)
plt.close(fig)
print("Saved: bf_speed_comparison.png")

# ── plot 2: mean ± SEM speed over time ────────────────────────────────────
# bf already has group labels; exclude first-frame zeros
frame_data = bf[bf["speed_um_per_frame"] > 0].copy()

time_stats = (
    frame_data.groupby(["Frame", "group"])["speed_um_per_frame"]
    .agg(mean="mean", sem=lambda x: x.sem())
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 4))
for grp, label, color in [
    ("non_infected", f"Non-infected (n={n_non})", "#5b8be0"),
    ("infected",     f"Infected (n={n_inf})",     "#e05b5b"),
]:
    d = time_stats[time_stats["group"] == grp].sort_values("Frame")
    ax.plot(d["Frame"], d["mean"], color=color, label=label, lw=1.5)
    ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                    color=color, alpha=0.2)

ax.set_xlabel("Frame")
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title("Movement speed over time: infected vs non-infected (A2)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_over_time.png", dpi=150)
plt.close(fig)
print("Saved: bf_speed_over_time.png")

# ── plot 3: BF lines + GFP speed for the same 255 matched cells ────────────
# Both lines are trimmed to each cell's GFP onset so they cover the same
# temporal window per cell (frames before GFP onset are excluded from BF too).
quality_infected_bf = set(per_track.loc[per_track["group"] == "infected", "Track ID"])
matches["temporally_consistent"] = matches["temporally_consistent"].map(
    {"True": True, "False": False, True: True, False: False}
)
matched = matches[
    (matches["match_tier"] == "confident") &
    (matches["is_ambiguous"] == False) &
    (matches["temporally_consistent"] == True) &
    (matches["bf_track_id"].dropna().astype(int).isin(quality_infected_bf))
].copy()
matched["bf_track_id"] = matched["bf_track_id"].astype(int)

# onset_frame per BF track
bf_onset = matched.set_index("bf_track_id")["onset_frame"].to_dict()

# GFP track IDs for those cells
matched_gfp_ids = set("A2_" + matched["gfp_track_id"].astype(int).astype(str))

# Trim BF infected data: only frames >= GFP onset for each cell
bf_inf_post = bf[bf["Track ID"].isin(quality_infected_bf)].copy()
bf_inf_post = bf_inf_post[
    bf_inf_post.apply(lambda r: r["Frame"] >= bf_onset.get(r["Track ID"], 0), axis=1)
]
bf_inf_post = bf_inf_post[bf_inf_post["speed_um_per_frame"] > 0]

bf_inf_time = (
    bf_inf_post.groupby("Frame")["speed_um_per_frame"]
    .agg(mean="mean", sem=lambda x: x.sem())
    .reset_index()
)

# GFP speed for matched cells (already starts at onset)
GFP_SPEED = ROOT / "cache" / "python_export" / "extra_features.csv"
gfp = pd.read_csv(GFP_SPEED, usecols=["Track.ID", "Frame", "speed_px_per_frame"])
gfp = gfp[gfp["Track.ID"].isin(matched_gfp_ids)].copy()
gfp = gfp[gfp["speed_px_per_frame"] > 0]

gfp_time = (
    gfp.groupby("Frame")["speed_px_per_frame"]
    .agg(mean="mean", sem=lambda x: x.sem())
    .reset_index()
)
n_gfp = gfp["Track.ID"].nunique()
print(f"GFP + BF infected lines: {n_gfp} matched cells, aligned to GFP onset")

fig, ax = plt.subplots(figsize=(9, 4))

# Non-infected BF (full movie, unchanged)
d = time_stats[time_stats["group"] == "non_infected"].sort_values("Frame")
ax.plot(d["Frame"], d["mean"], color="#5b8be0",
        label=f"BF non-infected (n={n_non})", lw=1.5)
ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                color="#5b8be0", alpha=0.2)

# BF infected — post-onset only
d = bf_inf_time.sort_values("Frame")
ax.plot(d["Frame"], d["mean"], color="#e05b5b",
        label=f"BF infected, post-onset (n={n_inf})", lw=1.5)
ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                color="#e05b5b", alpha=0.2)

# GFP segmentation — same cells, same window
d = gfp_time.sort_values("Frame")
ax.plot(d["Frame"], d["mean"], color="#2ca02c",
        label=f"GFP segmentation, post-onset (n={n_gfp})", lw=1.5, linestyle="--")
ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                color="#2ca02c", alpha=0.15)

ax.set_xlabel("Frame")
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title("Movement speed over time: BF tracks + GFP segmentation, post-onset (A2)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_over_time_with_gfp.png", dpi=150)
plt.close(fig)
print("Saved: bf_speed_over_time_with_gfp.png")

# ── plot 4: faithful BF tracks (max BF–GFP distance ≤ threshold) ──────────
MAX_DIST_THR = 50   # µm — ~2.5 cell radii; tracks that stay within this are
                    # considered to reliably follow the same cell as GFP

# All confident unambiguous matches for quality-passing infected BF tracks
all_matched = matches[
    (matches["match_tier"] == "confident") &
    (matches["is_ambiguous"] == False) &
    (matches["bf_track_id"].dropna().astype(int).isin(quality_infected_bf))
].copy()
all_matched["bf_track_id"] = all_matched["bf_track_id"].astype(int)
all_matched["gfp_track_id_str"] = "A2_" + all_matched["gfp_track_id"].astype(int).astype(str)

# Load GFP centroid positions (needed for distance computation)
gfp_pos = pd.read_csv(
    ROOT / "CompleteImage" / "A2_Merged_spots.csv",
    low_memory=False, usecols=["Track ID", "Frame", "X", "Y"]
)
gfp_pos = gfp_pos.dropna(subset=["Track ID"])
gfp_pos["Track.ID"] = "A2_" + gfp_pos["Track ID"].astype(int).astype(str)
gfp_pos = gfp_pos[gfp_pos["Track.ID"].isin(all_matched["gfp_track_id_str"])]

# Per-cell: max BF–GFP distance post-onset
faithful_bf_ids = []
for _, row in all_matched.iterrows():
    onset  = int(row["onset_frame"])
    cell_bf  = bf[(bf["Track ID"] == row["bf_track_id"]) &
                  (bf["Frame"] >= onset)][["Frame", "X", "Y"]]
    cell_gfp = gfp_pos[(gfp_pos["Track.ID"] == row["gfp_track_id_str"]) &
                       (gfp_pos["Frame"] >= onset)][["Frame", "X", "Y"]]
    merged = cell_bf.merge(cell_gfp, on="Frame", suffixes=("_bf", "_gfp"))
    if len(merged) < 5:
        continue
    dist = np.sqrt((merged["X_bf"] - merged["X_gfp"])**2 +
                   (merged["Y_bf"] - merged["Y_gfp"])**2)
    if dist.max() <= MAX_DIST_THR:
        faithful_bf_ids.append(row["bf_track_id"])

faithful_bf_ids = set(faithful_bf_ids)
faithful_matched = all_matched[all_matched["bf_track_id"].isin(faithful_bf_ids)]
faithful_gfp_ids = set(faithful_matched["gfp_track_id_str"])
faithful_onset   = faithful_matched.set_index("bf_track_id")["onset_frame"].to_dict()

n_faithful = len(faithful_bf_ids)
print(f"Faithful BF tracks (max dist ≤{MAX_DIST_THR} µm): {n_faithful} cells")

# BF speed — faithful tracks, post-onset only
bf_faithful = bf[bf["Track ID"].isin(faithful_bf_ids)].copy()
bf_faithful = bf_faithful[
    bf_faithful.apply(lambda r: r["Frame"] >= faithful_onset.get(r["Track ID"], 0), axis=1)
]
bf_faithful = bf_faithful[bf_faithful["speed_um_per_frame"] > 0]
bf_faithful_time = (
    bf_faithful.groupby("Frame")["speed_um_per_frame"]
    .agg(mean="mean", sem=lambda x: x.sem())
    .reset_index()
)

# GFP speed — same faithful cells, post-onset
GFP_SPEED = ROOT / "cache" / "python_export" / "extra_features.csv"
gfp_faithful = pd.read_csv(GFP_SPEED, usecols=["Track.ID", "Frame", "speed_px_per_frame"])
gfp_faithful = gfp_faithful[gfp_faithful["Track.ID"].isin(faithful_gfp_ids)].copy()
gfp_faithful = gfp_faithful[gfp_faithful["speed_px_per_frame"] > 0]
gfp_faithful_time = (
    gfp_faithful.groupby("Frame")["speed_px_per_frame"]
    .agg(mean="mean", sem=lambda x: x.sem())
    .reset_index()
)
n_gfp_faithful = gfp_faithful["Track.ID"].nunique()

fig, ax = plt.subplots(figsize=(9, 4))

# BF non-infected background reference
d = time_stats[time_stats["group"] == "non_infected"].sort_values("Frame")
ax.plot(d["Frame"], d["mean"], color="#5b8be0",
        label=f"BF non-infected (n={n_non})", lw=1.5)
ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                color="#5b8be0", alpha=0.2)

# BF infected — faithful tracks only, post-onset
d = bf_faithful_time.sort_values("Frame")
ax.plot(d["Frame"], d["mean"], color="#e05b5b",
        label=f"BF infected, faithful (n={n_faithful})", lw=1.5)
ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                color="#e05b5b", alpha=0.2)

# GFP segmentation — same faithful cells
d = gfp_faithful_time.sort_values("Frame")
ax.plot(d["Frame"], d["mean"], color="#2ca02c",
        label=f"GFP segmentation (n={n_gfp_faithful})", lw=1.5, linestyle="--")
ax.fill_between(d["Frame"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                color="#2ca02c", alpha=0.15)

ax.set_xlabel("Frame")
ax.set_ylabel("Mean speed (µm / frame)")
ax.set_title(f"Movement speed: faithful BF tracks (max BF–GFP dist ≤{MAX_DIST_THR} µm) + GFP overlay (A2)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "bf_speed_faithful_tracks.png", dpi=150)
plt.close(fig)
print(f"Saved: bf_speed_faithful_tracks.png")
print("\nDone.")

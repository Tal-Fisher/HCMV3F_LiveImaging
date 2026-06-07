"""
23_rolling_avg_morphology_A2A3.py

Rolling averages of cell area, nucleus area, nucleus circularity, and cell speed
vs. frames since first detection, all productive A2+A3 cells pooled.

Mean ± SEM across cells. Min 10 cells per norm_frame bin shown.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS  = 10   # min cells per norm_frame to plot
WIN        = 5    # rolling window (frames) for smoothing

# ── load timeseries + categories ───────────────────────────────────────────────
ts  = pd.read_csv(EXPORT_DIR / "timeseries_data.csv", low_memory=False)
cat = pd.read_csv(EXPORT_DIR / "category_df.csv")
cat = cat[cat["productive"] == True].dropna(subset=["category"])

# assign norm_frame = 0, 1, 2, ... from first frame per track
ts = ts.sort_values(["Track.ID", "T_min"]).copy()
ts["norm_frame"] = ts.groupby("Track.ID")["Frame"].transform(lambda f: f - f.min())

# keep only productive categorised cells
ts = ts[ts["Track.ID"].isin(cat["Track.ID"])]
print(f"Timeseries: {ts['Track.ID'].nunique()} cells  |  rows: {len(ts)}")

# ── compute cell speed from raw spots ─────────────────────────────────────────
def load_spots(path, dataset_label):
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"Track ID": "track_id", "T (sec)": "t_sec",
                             "Frame": "frame", "X": "x", "Y": "y"})
    for col in ["x", "y", "t_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["track_id"].notna()].copy()
    df["track_id"] = df["track_id"].astype(int)
    df["dataset"]  = dataset_label
    return df[["track_id", "frame", "t_sec", "x", "y", "dataset"]]

a2 = load_spots(BASE / "CompleteImage" / "A2_Merged_spots.csv", "A2")
a3 = load_spots(BASE / "CompleteImage" / "A3_Merged_spots.csv", "A3")
spots = pd.concat([a2, a3], ignore_index=True)

# build track_id key matching timeseries Track.ID format (e.g. "A2_1000")
spots["Track.ID"] = spots["dataset"] + "_" + spots["track_id"].astype(str)
spots = spots[spots["Track.ID"].isin(cat["Track.ID"])]
spots = spots.sort_values(["Track.ID", "t_sec"]).copy()

# norm_frame from track start
spots["norm_frame"] = spots.groupby("Track.ID")["frame"].transform(lambda f: f - f.min())

# per-track speed: displacement between consecutive frames / dt (pixels/min)
spots["dt_min"]  = spots.groupby("Track.ID")["t_sec"].diff() / 60.0
spots["dx"]      = spots.groupby("Track.ID")["x"].diff()
spots["dy"]      = spots.groupby("Track.ID")["y"].diff()
spots["speed"]   = np.sqrt(spots["dx"]**2 + spots["dy"]**2) / spots["dt_min"]
# first frame of each track has no predecessor → NaN (correct)

speed_df = spots[["Track.ID", "norm_frame", "speed"]].copy()
print(f"Speed computed: {speed_df['speed'].notna().sum()} values across {speed_df['Track.ID'].nunique()} cells")

# ── merge speed into timeseries ────────────────────────────────────────────────
ts = ts.merge(speed_df, on=["Track.ID", "norm_frame"], how="left")

# ── rolling mean ± SEM helper ──────────────────────────────────────────────────
def pop_stats(df, feat, min_cells=MIN_CELLS):
    """Return DataFrame with norm_frame, mean, sem, n per norm_frame."""
    g = df.groupby("norm_frame")[feat].agg(
        mean=lambda x: x.mean(skipna=True),
        sem=lambda x: (x.std(skipna=True) / np.sqrt(x.notna().sum()))
                      if x.notna().sum() > 1 else np.nan,
        n=lambda x: x.notna().sum()
    ).reset_index()
    g = g[g["n"] >= min_cells]
    # light smoothing
    g["mean_sm"] = g["mean"].rolling(WIN, center=True, min_periods=1).mean()
    g["sem_sm"]  = g["sem"].rolling(WIN,  center=True, min_periods=1).mean()
    return g

# ── build stats (all cells pooled) ────────────────────────────────────────────
features = [
    ("Area_cell",  "Cell area (px²)"),
    ("Area_nuc",   "Nucleus area (px²)"),
    ("Circ_nuc",   "Nucleus circularity"),
    ("speed",      "Cell speed (px/min)"),
]

n_cells = ts["Track.ID"].nunique()
print(f"\nAll productive cells: {n_cells}")

stats = {feat: pop_stats(ts, feat) for feat, _ in features}

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"A2+A3: rolling mean ± SEM by frames since first detection\n"
    f"All productive cells  (n={n_cells}, min 10 cells/bin, 5-frame smoothing)",
    fontsize=13, fontweight="bold"
)

for ax, (feat, ylabel) in zip(axes.flat, features):
    d = stats[feat]
    ax.fill_between(d["norm_frame"],
                    d["mean_sm"] - d["sem_sm"],
                    d["mean_sm"] + d["sem_sm"],
                    color="steelblue", alpha=0.25)
    ax.plot(d["norm_frame"], d["mean_sm"], color="steelblue", lw=2)

    ax.set_xlabel("Frames from first detection", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(left=0)

plt.tight_layout()
out = FIG_DIR / "rolling_avg_morphology_A2A3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

"""
Rolling average of morphological + ratio features across all A2+A3 cells,
each cell aligned to its own track start (norm_frame 0 = first frame).
Population mean ± SEM per norm_frame, smoothed with a rolling window.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/timeseries_data.csv"
OUT   = "/home/labs/ginossar/talfis/LiveImaging/figures/combined/rolling_avg_morphology.png"
ROLL_WIN = 7   # rolling window for smoothing (frames)
MIN_N    = 10  # drop norm_frames with fewer cells than this

df = pd.read_csv(CACHE, low_memory=False)
df = df[df["dataset"].isin(["A2", "A3"])].copy()

# GFP/BFP ratio — guard against zero/negative BFP
df["gfp_bfp_ratio"] = np.where(
    df["Mean.ch1"] > 0,
    df["ch2_corrected"] / df["Mean.ch1"],
    np.nan
)

features = {
    "bf_ctrst":      ("Ctrst.ch4",    "BF contrast (Ctrst ch4)"),
    "solidity":      ("Solidity",      "Solidity"),
    "shape_idx":     ("Shape_index",   "Shape index"),
    "nuc_ratio":     ("nuc_ratio",     "Nuc/cell area ratio"),
    "gfp_bfp_ratio": ("gfp_bfp_ratio", "GFP/BFP ratio"),
}

# clip nuc_ratio to physically valid range (nucleus can't exceed cell area)
df["nuc_ratio"] = df["nuc_ratio"].where(df["nuc_ratio"] <= 1.0)

# normalise time: norm_frame = Frame - min(Frame) per cell
df["norm_frame"] = df.groupby("Track.ID")["Frame"].transform(lambda f: f - f.min())

def pop_stats(series_by_frame):
    """Per norm_frame: mean, SEM, n."""
    mean = series_by_frame.mean()
    sem  = series_by_frame.std() / np.sqrt(series_by_frame.count())
    n    = series_by_frame.count()
    return mean.rename("mean"), sem.rename("sem"), n.rename("n")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

n_cells = df["Track.ID"].nunique()

for i, (feat_key, (col, label)) in enumerate(features.items()):
    ax = axes[i]

    grp   = df.groupby("norm_frame")[col]
    mean, sem, n = pop_stats(grp)

    idx   = n[n >= MIN_N].index
    mean  = mean.loc[idx]
    sem   = sem.loc[idx]

    # rolling smooth
    mean_s = mean.rolling(ROLL_WIN, center=True, min_periods=1).mean()
    sem_s  = sem.rolling(ROLL_WIN,  center=True, min_periods=1).mean()

    ax.fill_between(idx, mean_s - sem_s, mean_s + sem_s, alpha=0.25, color="steelblue")
    ax.plot(idx, mean_s, color="steelblue", lw=2)

    ax.set_xlabel("Frames from track start", fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

# hide unused 6th panel
axes[5].set_visible(False)

fig.suptitle(
    f"Population trajectory (A2+A3, n={n_cells} cells)\n"
    f"Mean ± SEM, rolling window = {ROLL_WIN} frames, aligned to each cell's track start",
    fontsize=12, y=1.01
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")

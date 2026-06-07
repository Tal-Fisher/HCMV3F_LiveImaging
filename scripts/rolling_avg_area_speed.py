"""
Rolling average of area/circularity/speed features across all A2+A3 cells,
each cell aligned to its own track start (norm_frame 0 = first frame).
Population mean ± SEM per norm_frame, smoothed with a rolling window.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_TS    = "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/timeseries_data.csv"
CACHE_EXTRA = "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/extra_features.csv"
OUT         = "/home/labs/ginossar/talfis/LiveImaging/figures/combined/rolling_avg_area_speed.png"
ROLL_WIN    = 7
MIN_N       = 10

ts    = pd.read_csv(CACHE_TS,    low_memory=False)
extra = pd.read_csv(CACHE_EXTRA, low_memory=False)

ts = ts[ts["dataset"].isin(["A2", "A3"])].copy()
df = ts.merge(extra[["Track.ID", "Frame", "speed_px_per_frame"]], on=["Track.ID", "Frame"], how="left")

df["norm_frame"] = df.groupby("Track.ID")["Frame"].transform(lambda f: f - f.min())

features = {
    "Area_cell":         "Cell area (px²)",
    "Area_nuc":          "Nucleus area (px²)",
    "Circ_nuc":          "Nucleus circularity",
    "speed_px_per_frame": "Cell speed (px/frame)",
}

def pop_stats(grp):
    mean = grp.mean()
    sem  = grp.std() / np.sqrt(grp.count())
    n    = grp.count()
    return mean.rename("mean"), sem.rename("sem"), n.rename("n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

n_cells = df["Track.ID"].nunique()

for i, (col, label) in enumerate(features.items()):
    ax = axes[i]

    grp         = df.groupby("norm_frame")[col]
    mean, sem, n = pop_stats(grp)

    idx    = n[n >= MIN_N].index
    mean   = mean.loc[idx]
    sem    = sem.loc[idx]

    mean_s = mean.rolling(ROLL_WIN, center=True, min_periods=1).mean()
    sem_s  = sem.rolling(ROLL_WIN,  center=True, min_periods=1).mean()

    ax.fill_between(idx, mean_s - sem_s, mean_s + sem_s, alpha=0.25, color="steelblue")
    ax.plot(idx, mean_s, color="steelblue", lw=2)

    ax.set_xlabel("Frames from track start", fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"Population trajectory (A2+A3, n={n_cells} cells)\n"
    f"Mean ± SEM, rolling window = {ROLL_WIN} frames, aligned to each cell's track start",
    fontsize=12, y=1.01
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")

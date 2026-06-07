"""
Plot when each cell first appears in the movie (first segmented frame),
shown as horizontal lines spanning the full tracked duration.
Colored by group: early / medium / late / non-productive.
Shows ALL 861 cells from model_df (no filters).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE   = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT    = BASE / "Forecast"

EARLY_CUT = 911
LATE_CUT  = 2163

COLORS = {
    "early":          "#e67e22",
    "medium":         "#2980b9",
    "late":           "#27ae60",
    "non-productive": "#aaaaaa",
}

FRAME_INTERVAL = 15.0005   # minutes per frame (A2); close enough for A3 too
MAX_FRAME      = 288        # last frame in both datasets

ts = pd.read_csv(BASE / "cache/python_export/timeseries_data.csv", low_memory=False)
md = pd.read_csv(BASE / "cache/python_export/model_df.csv")[
    ["Track.ID", "delay_green_to_red", "track_start_min", "dataset"]
]

# cell extents from timeseries (809 cells)
ts_extents = (
    ts.groupby("Track.ID")
    .agg(first_frame=("Frame", "min"), last_frame=("Frame", "max"))
    .reset_index()
)

# merge all model_df cells (861) with ts extents — left join keeps all
cell_summary = md.merge(ts_extents, on="Track.ID", how="left")

# fill missing extents for cells absent from timeseries
missing = cell_summary["first_frame"].isna()
cell_summary.loc[missing, "first_frame"] = (
    cell_summary.loc[missing, "track_start_min"] / FRAME_INTERVAL
).round().astype(int)
cell_summary.loc[missing, "last_frame"] = MAX_FRAME

cell_summary["first_frame"] = cell_summary["first_frame"].astype(int)
cell_summary["last_frame"]  = cell_summary["last_frame"].astype(int)


def group(delay):
    if pd.isna(delay) or not np.isfinite(delay):
        return "non-productive"
    return "early" if delay <= EARLY_CUT else ("medium" if delay <= LATE_CUT else "late")


cell_summary["group"] = cell_summary["delay_green_to_red"].apply(group)

# sort: non-productive first (bottom), then early, medium, late; within each by first_frame
ORDER = ["non-productive", "early", "medium", "late"]
cell_summary["group_order"] = cell_summary["group"].map({g: i for i, g in enumerate(ORDER)})
cell_summary = cell_summary.sort_values(["group_order", "first_frame"]).reset_index(drop=True)
cell_summary["y"] = np.arange(len(cell_summary))

fig, ax = plt.subplots(figsize=(13, 8))

for _, row in cell_summary.iterrows():
    col = COLORS[row["group"]]
    ax.hlines(row["y"], row["first_frame"], row["last_frame"],
              colors=col, lw=0.6, alpha=0.7)
    ax.plot(row["first_frame"], row["y"], "o", color=col,
            ms=2.0, alpha=0.85, zorder=3)

# group boundary lines
for g in ORDER[:-1]:
    boundary = cell_summary[cell_summary["group"] == g]["y"].max()
    if not np.isnan(boundary):
        ax.axhline(boundary + 0.5, color="#333", lw=0.6, linestyle="--", alpha=0.4)

# group labels on right side
ax.set_xlim(cell_summary["first_frame"].min() - 5,
            cell_summary["last_frame"].max() + 20)
for g in ORDER:
    sub = cell_summary[cell_summary["group"] == g]
    if len(sub) == 0:
        continue
    mid_y = sub["y"].median()
    ax.text(ax.get_xlim()[1], mid_y, f" {g}\n (n={len(sub)})",
            va="center", fontsize=7.5, color=COLORS[g], fontweight="bold")

ax.set_xlabel("Movie frame", fontsize=10)
ax.set_ylabel("Cell (sorted by group, then first appearance)", fontsize=9)
ax.set_title(
    f"Cell tracking windows across the movie  (n={len(cell_summary)} total)\n"
    "Dot = first segmented frame  |  Line = full tracked duration",
    fontsize=11, fontweight="bold"
)
ax.set_ylim(-1, len(cell_summary))

legend_patches = [mpatches.Patch(color=COLORS[g], label=g) for g in ORDER]
ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = OUT / "cell_entry_frames.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out_path}")

for g in ORDER:
    n = (cell_summary["group"] == g).sum()
    sub = cell_summary[cell_summary["group"] == g]
    print(f"  {g:16s}: {n:3d} cells  "
          f"first_frame {sub['first_frame'].min()}-{sub['first_frame'].max()}")

"""
Figure 1 v5: Lines-only, productive cells only (reach mCherry), no dots, no smoothing.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/model_df.csv"
)

df["gfp_onset"] = df["abs_gfp_onset_min"]
df["bfp_onset"] = df["abs_gfp_onset_min"] + df["delay_green_to_blue"]
df["red_onset"] = df["abs_gfp_onset_min"] + df["delay_green_to_red"]
df["productive"] = np.isfinite(df["red_onset"])

# Track end times from raw spots
def load_track_ends(path, prefix):
    spots = pd.read_csv(path, low_memory=False)
    spots["t_min"] = spots["T (sec)"] / 60.0
    ends = spots.groupby("Track ID")["t_min"].max().reset_index()
    ends["Track.ID"] = prefix + ends["Track ID"].astype(int).astype(str)
    return ends[["Track.ID", "t_min"]].rename(columns={"t_min": "track_end_min"})

base = "/home/labs/ginossar/talfis/LiveImaging/CompleteImage/"
track_ends = pd.concat([
    load_track_ends(base + "A2_Merged_spots.csv", "A2_"),
    load_track_ends(base + "A3_Merged_spots.csv", "A3_"),
])
df = df.merge(track_ends, on="Track.ID", how="left")

# Productive cells only, sorted by GFP onset
df = df[df["productive"]].sort_values("gfp_onset").reset_index(drop=True)
n = len(df)

# ── figure ────────────────────────────────────────────────────────────────────
FONTSIZE = 52
DOT_SIZE = 30         # marker size in points
LINE_WIDTH = round(0.8 * 861 / n, 2)  # scale up to fill rows at 16×14

fig_w = 24   # 1.5× wider than v6
fig_h = 14

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

for i, row in df.iterrows():
    ax.plot([row["gfp_onset"],  row["bfp_onset"]], [i, i], color="#00BB00", linewidth=LINE_WIDTH, solid_capstyle="butt", zorder=1)
    ax.plot([row["bfp_onset"],  row["red_onset"]], [i, i], color="#1155EE", linewidth=LINE_WIDTH, solid_capstyle="butt", zorder=1)
    ax.plot([row["red_onset"],  row["track_end_min"]], [i, i], color="#EE0000", linewidth=LINE_WIDTH, solid_capstyle="butt", zorder=1)


# ── axes formatting ────────────────────────────────────────────────────────────
ax.set_xlabel("Time (minutes)", fontsize=FONTSIZE)
ax.set_ylabel("Cells", fontsize=FONTSIZE)
ax.tick_params(axis="both", labelsize=FONTSIZE - 4)

ax.set_ylim(-1, n)
ax.set_xlim(left=0)

# Minor grid on x
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.grid(axis="x", which="major", color="0.85", linewidth=0.5, zorder=0)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Y-axis: hide ticks (861 rows — individual labels are meaningless)
ax.set_yticks([])

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "fig1_onset_timeline_v7.pdf")
png_path = os.path.join(script_dir, "fig1_onset_timeline_v7.png")
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Saved: {png_path}")

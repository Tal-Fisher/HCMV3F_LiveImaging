"""
Cell entry figure with per-channel onset lines, old-figure style.
Shows ALL 861 cells from model_df (no filters).

Left panel  : productive cells, four separate stacked blocks
              (green / blue-GFP-rank / red / blue-BFP-rank),
              first three ordered by GFP onset, last ordered by BFP onset.
Right panel : non-productive cells, three blocks (green / blue-GFP-rank /
              blue-BFP-rank), first two ordered by GFP onset, last by BFP onset.

The BFP-onset-sorted block makes late-onset BFP events cluster at the bottom
and easy to identify visually.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT  = BASE / "figures/combined"

# ── load ────────────────────────────────────────────────────────────────────
ts = pd.read_csv(BASE / "cache/python_export/timeseries_data.csv", low_memory=False)
md = pd.read_csv(BASE / "cache/python_export/model_df.csv")
movie_dur = pd.read_csv(BASE / "cache/python_export/movie_durations.csv")
dur_map = dict(zip(movie_dur["dataset"], movie_dur["duration_min"]))

cell_extents = (
    ts.groupby("Track.ID")
    .agg(first_t=("T_min", "min"), last_t=("T_min", "max"))
    .reset_index()
)

# left join: keep all 861 model_df cells
df = md[["Track.ID", "abs_gfp_onset_min", "delay_green_to_blue",
         "delay_green_to_red", "dataset", "track_start_min"]].merge(
    cell_extents, on="Track.ID", how="left"
)

# fill extents for cells absent from timeseries
missing = df["first_t"].isna()
df.loc[missing, "first_t"] = df.loc[missing, "track_start_min"]
df.loc[missing, "last_t"]  = df.loc[missing, "dataset"].map(dur_map)

df["productive"]  = np.isfinite(df["delay_green_to_red"])
df["blue_onset"]  = df["abs_gfp_onset_min"] + df["delay_green_to_blue"]
df["red_onset"]   = df["abs_gfp_onset_min"] + df["delay_green_to_red"]
df["has_blue"]    = np.isfinite(df["delay_green_to_blue"])

# sort by GFP onset
prod    = df[df["productive"]].sort_values("abs_gfp_onset_min").reset_index(drop=True)
nonprod = df[~df["productive"]].sort_values("abs_gfp_onset_min").reset_index(drop=True)
print(f"Productive: {len(prod)}   Non-productive: {len(nonprod)}")

# BFP-sorted copies (for the extra block)
prod_bfp_sorted    = prod.sort_values("blue_onset").reset_index(drop=True)
nonprod_bfp_sorted = nonprod.sort_values("blue_onset").reset_index(drop=True)

# late bloomers count for label
LATE_THRESH = 150   # min
n_late_prod    = (prod["delay_green_to_blue"]    > LATE_THRESH).sum()
n_late_nonprod = (nonprod["delay_green_to_blue"] > LATE_THRESH).sum()
print(f"Late BFP (delay>{LATE_THRESH} min)  productive: {n_late_prod}  non-productive: {n_late_nonprod}")

# ── visual constants ────────────────────────────────────────────────────────
C_GREEN = "#27ae60"
C_BLUE  = "#2980b9"
C_RED   = "#e74c3c"
LW      = 0.55
ALPHA   = 0.75
DOT_MS  = 1.5
GAP     = 30          # blank rows between channel blocks

X_MIN = df["abs_gfp_onset_min"].min() - 50
X_MAX = df["last_t"].max() + 50


# ── drawing ──────────────────────────────────────────────────────────────────
def draw_panel(ax, cells, cells_bfp, has_red, gap=GAP):
    """
    cells     : GFP-onset-sorted DataFrame (reset_index, index=rank)
    cells_bfp : same cells but sorted by blue_onset (reset_index, index=bfp_rank)
    has_red   : True for productive panel
    """
    n  = len(cells)

    # y-offsets for each block
    g0  = 0
    b0  = n + gap
    r0  = 2 * n + 2 * gap          # only used if has_red
    bs0 = (3 if has_red else 2) * n + (2 if has_red else 1) * gap + gap  # BFP-sorted

    # ── GFP-rank blocks ──────────────────────────────────────────────────────
    for rank, row in cells.iterrows():
        # green
        yg = g0 + rank
        ax.hlines(yg, row["abs_gfp_onset_min"], row["last_t"],
                  colors=C_GREEN, lw=LW, alpha=ALPHA)
        ax.plot(row["abs_gfp_onset_min"], yg, "o",
                color=C_GREEN, ms=DOT_MS, alpha=0.9, zorder=3)

        # blue (GFP-rank)
        yb = b0 + rank
        if row["has_blue"]:
            ax.hlines(yb, row["blue_onset"], row["last_t"],
                      colors=C_BLUE, lw=LW, alpha=ALPHA)
            ax.plot(row["blue_onset"], yb, "o",
                    color=C_BLUE, ms=DOT_MS, alpha=0.9, zorder=3)

        # red
        if has_red:
            yr = r0 + rank
            ax.hlines(yr, row["red_onset"], row["last_t"],
                      colors=C_RED, lw=LW, alpha=ALPHA)
            ax.plot(row["red_onset"], yr, "o",
                    color=C_RED, ms=DOT_MS, alpha=0.9, zorder=3)

    # ── BFP-onset-sorted block ────────────────────────────────────────────────
    for bfp_rank, row in cells_bfp.iterrows():
        ybs = bs0 + bfp_rank
        if row["has_blue"]:
            ax.hlines(ybs, row["blue_onset"], row["last_t"],
                      colors=C_BLUE, lw=LW, alpha=ALPHA)
            ax.plot(row["blue_onset"], ybs, "o",
                    color=C_BLUE, ms=DOT_MS, alpha=0.9, zorder=3)

    # ── separator lines ────────────────────────────────────────────────────────
    n_gfp_blocks = 3 if has_red else 2          # GFP-rank blocks count
    separators = []
    for k in range(1, n_gfp_blocks):
        separators.append(k * n + (k - 1) * gap + gap / 2)
    separators.append(bs0 - gap / 2)            # before BFP-sorted block
    for sep in separators:
        ax.axhline(sep, color="#555", lw=0.7, linestyle="--", alpha=0.5)

    # ── axis limits ───────────────────────────────────────────────────────────
    total = bs0 + n
    ax.set_ylim(-5, total + 5)

    # ── block labels ──────────────────────────────────────────────────────────
    label_x = X_MAX - (X_MAX - X_MIN) * 0.01
    n_blue  = int(cells["has_blue"].sum())

    ax.text(label_x, g0 + n / 2, f"GFP\n(n={n})", color=C_GREEN,
            fontsize=7.5, fontweight="bold", va="center", ha="right")
    ax.text(label_x, b0 + n / 2, f"BFP\n(n={n_blue})\n[GFP order]",
            color=C_BLUE, fontsize=7.5, fontweight="bold", va="center", ha="right")
    if has_red:
        ax.text(label_x, r0 + n / 2, f"mCherry\n(n={n})",
                color=C_RED, fontsize=7.5, fontweight="bold", va="center", ha="right")
    ax.text(label_x, bs0 + n / 2,
            f"BFP\n(n={n_blue})\n[BFP order]",
            color=C_BLUE, fontsize=7.5, fontweight="bold", va="center", ha="right")


# ── figure ──────────────────────────────────────────────────────────────────
n_prod    = len(prod)
n_nonprod = len(nonprod)

# 4 blocks left (GFP, BFP-gfp, red, BFP-bfp), 3 blocks right (GFP, BFP-gfp, BFP-bfp)
y_range_l = 4 * n_prod    + 3 * GAP
y_range_r = 3 * n_nonprod + 2 * GAP

height = max(y_range_l, y_range_r) / 80   # ~80 cells per inch

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, max(height, 10)))

# LEFT panel
draw_panel(ax_l, prod, prod_bfp_sorted, has_red=True)
ax_l.set_xlim(X_MIN, X_MAX)
ax_l.set_xlabel("Time (min)", fontsize=10)
ax_l.set_ylabel("Cell rank within block", fontsize=9)
ax_l.set_title(
    f"Productive cells  (n={n_prod})\n"
    "Dot = channel onset  |  bottom block sorted by BFP onset",
    fontsize=10, fontweight="bold",
)

# RIGHT panel
draw_panel(ax_r, nonprod, nonprod_bfp_sorted, has_red=False)
ax_r.set_xlim(X_MIN, X_MAX)
ax_r.set_xlabel("Time (min)", fontsize=10)
ax_r.set_title(
    f"Non-productive cells  (n={n_nonprod})\n"
    "Dot = channel onset  |  bottom block sorted by BFP onset",
    fontsize=10, fontweight="bold",
)

for ax in (ax_l, ax_r):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.invert_yaxis()

fig.suptitle(
    f"Cell tracking windows — per-channel onset  (n={len(df)} total)\n"
    "Lines span from channel onset to end of tracking  |  "
    f"Late BFP (>{LATE_THRESH} min delay): {n_late_prod+n_late_nonprod} cells",
    fontsize=12, fontweight="bold",
)

plt.tight_layout()
out_path = OUT / "cell_entry_frames_channels.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out_path}")

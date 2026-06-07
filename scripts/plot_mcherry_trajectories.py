"""
plot_mcherry_trajectories.py

PDF with mCherry trajectories for 20 random early and 20 random medium+late cells.
X-axis: time from GFP onset (min). Y-axis: Mean mCherry intensity.
Red dashed horizontal line: red threshold (2.25).
Black dashed vertical line: red onset time.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RED_THRESH    = 2.25
CUT_EARLY_MED = 911
N_CELLS       = 20
SEED          = 42

# ── load data ──────────────────────────────────────────────────────────────────
ts  = pd.read_csv(EXPORT_DIR / "timeseries_data.csv", low_memory=False)
cat = pd.read_csv(EXPORT_DIR / "category_df.csv")

# keep only productive cells with a category
cat = cat[cat["productive"] == True].dropna(subset=["category"])

# align x-axis: time relative to GFP onset
ts["t_from_onset"] = ts["T_min"] - ts["abs_gfp_onset_min"]

# ── select cells ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

early_ids   = cat[cat["category"] == "early"]["Track.ID"].values
rest_ids    = cat[cat["category"].isin(["medium", "late"])]["Track.ID"].values

# only keep cells that have timeseries data
ts_ids      = set(ts["Track.ID"].unique())
early_ids   = [t for t in early_ids if t in ts_ids]
rest_ids    = [t for t in rest_ids  if t in ts_ids]

sel_early   = rng.choice(early_ids, size=min(N_CELLS, len(early_ids)),  replace=False)
sel_rest    = rng.choice(rest_ids,  size=min(N_CELLS, len(rest_ids)),   replace=False)

print(f"Early cells available: {len(early_ids)}  → plotting {len(sel_early)}")
print(f"Med+late cells available: {len(rest_ids)} → plotting {len(sel_rest)}")

# ── plotting helper ─────────────────────────────────────────────────────────────
def plot_page(ax_grid, cell_ids, group_label, cat_df, ts_df):
    for ax, tid in zip(ax_grid.flat, cell_ids):
        cell_ts  = ts_df[ts_df["Track.ID"] == tid].sort_values("t_from_onset")
        row      = cat_df[cat_df["Track.ID"] == tid].iloc[0]
        delay    = row["delay_green_to_red"]   # min from GFP onset

        ax.plot(cell_ts["t_from_onset"], cell_ts["Mean.ch3"],
                color="tomato", lw=1.2, alpha=0.85)

        # red threshold
        ax.axhline(RED_THRESH, color="red", ls="--", lw=0.9, alpha=0.8)

        # red onset (only for productive cells)
        if np.isfinite(delay):
            ax.axvline(delay, color="black", ls="--", lw=0.9, alpha=0.8)

        ax.set_title(f"{tid}\n{row['category']}  ({delay:.0f} min)" if np.isfinite(delay)
                     else f"{tid}\n{row['category']}  (no red)",
                     fontsize=6, pad=2)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("min from GFP onset", fontsize=5)
        ax.set_ylabel("mCherry", fontsize=5)

    # hide unused axes
    for ax in ax_grid.flat[len(cell_ids):]:
        ax.set_visible(False)

# ── build PDF ──────────────────────────────────────────────────────────────────
out_path = FIG_DIR / "mcherry_trajectories_early_vs_rest.pdf"

with PdfPages(out_path) as pdf:
    for group_label, cell_ids in [("Early  (delay ≤ 911 min)", sel_early),
                                   ("Medium + Late  (delay > 911 min)", sel_rest)]:
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(
            f"mCherry trajectories — {group_label}  (n={len(cell_ids)})\n"
            f"Red dashed line = threshold ({RED_THRESH})  |  "
            f"Black dashed line = red onset",
            fontsize=11, fontweight="bold"
        )
        plot_page(axes, cell_ids, group_label, cat, ts)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close()

print(f"Saved {out_path}")

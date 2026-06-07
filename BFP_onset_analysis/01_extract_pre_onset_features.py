"""
01_extract_pre_onset_features.py — Build leakage-free pre-onset feature table

For each cell with delay_green_to_blue > 0, extract summary features from the
time window [abs_gfp_onset_min, abs_gfp_onset_min + delay_green_to_blue).
All measurements are strictly BEFORE the blue onset event, so there is no
circularity with the outcome.

Cells with delay = 0 (simultaneous green/blue onset) have no pre-onset window
and are excluded from this table. They are used separately in script 02 via
the *_start columns already present in model_df.csv.

Output: BFP_onset_analysis/cache/pre_onset_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress

BASE      = Path("/home/labs/ginossar/talfis/LiveImaging")
CACHE     = BASE / "BFP_onset_analysis" / "cache"
TS_PATH   = BASE / "cache" / "python_export" / "timeseries_data.csv"
META_PATH = BASE / "cache" / "python_export" / "model_df.csv"

# ── load data ─────────────────────────────────────────────────────────────────
ts   = pd.read_csv(TS_PATH, low_memory=False)
meta = pd.read_csv(META_PATH)

# coerce El_long_axis to numeric (stored as string in some rows)
ts["El_long_axis"] = pd.to_numeric(ts["El_long_axis"], errors="coerce")

# ── filter meta ───────────────────────────────────────────────────────────────
n_total = len(meta)
# first-half filter
meta = meta[meta["abs_gfp_onset_min"] <= meta["movie_half_min"]].copy()
n_after_half = len(meta)
# require delay > 0  (delay = 0 → no pre-onset window)
meta = meta[meta["delay_green_to_blue"].notna() & (meta["delay_green_to_blue"] > 0)].copy()
n_delay_pos = len(meta)

print(f"Total cells in model_df:          {n_total}")
print(f"After first-half filter:           {n_after_half}")
print(f"With delay_green_to_blue > 0:      {n_delay_pos}")

meta["blue_onset_abs_min"] = meta["abs_gfp_onset_min"] + meta["delay_green_to_blue"]

# signals to summarise (all available in timeseries_data)
SIGNALS = [
    "ch2_corrected",   # GFP expression
    "Area_cell",       # cell size
    "Solidity",        # cell shape compactness
    "Shape_index",     # cell shape index
    "El_long_axis",    # cell elongation
    "Ctrst.ch4",       # brightfield contrast
    "Area_nuc",        # nuclear size
    "Circ_nuc",        # nuclear circularity
    "nuc_ratio",       # nucleus/cell area ratio
    "Mean.ch1_nuc",    # nuclear BFP (pre-onset baseline, not accumulation)
]

def summarise(vals, frames):
    """Return dict of start/mean/sd/slope for a 1-D array."""
    v = np.asarray(vals, dtype=float)
    f = np.asarray(frames, dtype=float)
    ok = np.isfinite(v) & np.isfinite(f)
    v, f = v[ok], f[ok]
    if len(v) == 0:
        return dict(start=np.nan, mean=np.nan, sd=np.nan, slope=np.nan)
    start = v[0]
    mean  = v.mean()
    sd    = v.std(ddof=1) if len(v) >= 2 else np.nan
    if len(v) >= 2:
        slope = linregress(f, v).slope
    else:
        slope = np.nan
    return dict(start=start, mean=mean, sd=sd, slope=slope)

# ── per-cell feature extraction ────────────────────────────────────────────────
records = []
n_excluded_short = 0

for _, row in meta.iterrows():
    tid        = row["Track.ID"]
    gfp_onset  = row["abs_gfp_onset_min"]
    blue_onset = row["blue_onset_abs_min"]

    # select pre-onset frames for this cell
    cell_ts = ts[ts["Track.ID"] == tid].copy()
    pre = cell_ts[
        (cell_ts["T_min"] >= gfp_onset) &
        (cell_ts["T_min"] <  blue_onset)
    ].sort_values("T_min")

    if len(pre) < 2:
        n_excluded_short += 1
        continue

    # frame index relative to first pre-onset frame (0, 1, 2, ...)
    pre = pre.reset_index(drop=True)
    frame_idx = pre.index.values

    rec = {
        "Track.ID":              tid,
        "dataset":               row["dataset"],
        "delay_green_to_blue":   row["delay_green_to_blue"],
        "delay_green_to_red":    row["delay_green_to_red"],
        "abs_gfp_onset_min":     gfp_onset,
        "movie_half_min":        row["movie_half_min"],
        "n_pre_onset_frames":    len(pre),
    }
    for sig in SIGNALS:
        if sig not in pre.columns:
            for sfx in ("start", "mean", "sd", "slope"):
                rec[f"{sig}_{sfx}"] = np.nan
            continue
        stats = summarise(pre[sig].values, frame_idx)
        for sfx, val in stats.items():
            rec[f"{sig}_{sfx}"] = val

    records.append(rec)

feat_df = pd.DataFrame(records)
feat_df.to_csv(CACHE / "pre_onset_features.csv", index=False)

n_out = len(feat_df)
print(f"Excluded (< 2 pre-onset frames):   {n_excluded_short}")
print(f"Cells in output:                   {n_out}")
print(f"Median pre-onset window:           {feat_df['n_pre_onset_frames'].median():.0f} frames  "
      f"({feat_df['n_pre_onset_frames'].median() * 15:.0f} min)")
print(f"delay_green_to_blue — median {feat_df['delay_green_to_blue'].median():.0f} min, "
      f"range [{feat_df['delay_green_to_blue'].min():.0f}, {feat_df['delay_green_to_blue'].max():.0f}]")
print(f"Saved: {CACHE / 'pre_onset_features.csv'}")

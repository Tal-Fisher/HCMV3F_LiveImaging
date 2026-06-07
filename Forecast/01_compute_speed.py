"""
01_compute_speed.py

Extracts per-frame movement speed, BF mean intensity (Mean ch4), and
cell circularity (Circ.) from the raw spots CSV files (A2/A3).

These columns exist in the raw spots files but were not included in
timeseries_data.csv.

Track ID mapping:
  Raw spots:  Track ID = 15  (integer)
  Timeseries: Track.ID = "A2_15" (string, dataset prefix + underscore)

Output:
  LiveImaging/cache/python_export/extra_features.csv
  Columns: Track.ID, Frame, speed_px_per_frame, Mean_ch4, Circ_cell
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT_CSV  = BASE / "cache" / "python_export" / "extra_features.csv"

SPOTS_FILES = {
    "A2": BASE / "CompleteImage" / "A2_Merged_spots.csv",
    "A3": BASE / "CompleteImage" / "A3_Merged_spots.csv",
}

USECOLS = ["Track ID", "Frame", "X", "Y", "Mean ch4", "Circ."]

dfs = []
for dataset, path in SPOTS_FILES.items():
    print(f"Reading {path.name} ...", flush=True)
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    df = df.rename(columns={
        "Track ID": "track_num",
        "Mean ch4": "Mean_ch4",
        "Circ.":    "Circ_cell",
    })
    df["Track.ID"] = dataset + "_" + df["track_num"].astype(str)
    df = df.drop(columns="track_num")
    dfs.append(df)
    print(f"  {len(df):,} rows, {df['Track.ID'].nunique()} tracks", flush=True)

spots = pd.concat(dfs, ignore_index=True)

# Sort within each cell by Frame before computing speed
spots = spots.sort_values(["Track.ID", "Frame"]).reset_index(drop=True)

print("Computing per-frame speed ...", flush=True)
dx = spots.groupby("Track.ID")["X"].diff()
dy = spots.groupby("Track.ID")["Y"].diff()
spots["speed_px_per_frame"] = np.sqrt(dx**2 + dy**2)
# First frame of each track: displacement undefined → 0
spots["speed_px_per_frame"] = spots["speed_px_per_frame"].fillna(0.0)

# Keep only output columns
out = spots[["Track.ID", "Frame", "speed_px_per_frame", "Mean_ch4", "Circ_cell"]].copy()

# Sanity checks
print("\n--- Sanity checks ---")
print(f"Total rows:          {len(out):,}")
print(f"Unique tracks:       {out['Track.ID'].nunique()}")
print(f"Speed NaN:           {out['speed_px_per_frame'].isna().sum()}")
print(f"Speed range:         {out['speed_px_per_frame'].min():.3f} – {out['speed_px_per_frame'].max():.3f} px/frame")
print(f"Mean_ch4 NaN:        {out['Mean_ch4'].isna().sum()} ({out['Mean_ch4'].isna().mean()*100:.2f}%)")
print(f"Mean_ch4 range:      {out['Mean_ch4'].min():.3f} – {out['Mean_ch4'].max():.3f}")
print(f"Circ_cell NaN:       {out['Circ_cell'].isna().sum()} ({out['Circ_cell'].isna().mean()*100:.2f}%)")
print(f"Circ_cell range:     {out['Circ_cell'].min():.4f} – {out['Circ_cell'].max():.4f}")

out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}", flush=True)

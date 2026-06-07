"""
02_build_3d_table.py

Builds the 3D feature tensor for TabICL forecasting.

Tensor shape: (n_cells, max_frames, n_features)
  - n_cells:    productive, first-half cells only
  - max_frames: longest track length; shorter cells zero-padded at end
  - n_features: 9 (see FEATURE_COLS below)

Frame axis: normalized — frame index 1 = first tracked frame for that cell
  (sorted by T_min within each Track.ID; may include pre-GFP-onset frames)

Also saves a mCherry target array (n_cells, max_frames) for use as
forecast target in script 03.

Outputs:
  Forecast/tensor_X.npz           — float32, shape (n_cells, max_frames, 9)
  Forecast/tensor_y_mcherry.npz   — float32, shape (n_cells, max_frames)
  Forecast/cell_metadata.csv      — one row per cell
  Forecast/tensor_summary.txt     — diagnostics
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT_DIR = BASE / "Forecast"

TS_CSV    = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV    = BASE / "cache" / "python_export" / "model_df.csv"
EXTRA_CSV = BASE / "cache" / "python_export" / "extra_features.csv"

EARLY_CUT = 911
LATE_CUT  = 2163

# Feature columns (in the merged dataframe after join)
# Order is preserved in tensor axis 2
FEATURE_COLS = [
    "ch2_corrected",       # GFP
    "Mean.ch1_nuc",        # Nucleus BFP   — NaN→0 (segmentation gaps)
    "Mean_ch4",            # BF mean       — from raw spots
    "Ctrst.ch4",           # BF contrast
    "Area_cell",           # Cell size
    "Area_nuc",            # Nucleus size  — NaN→0
    "speed_px_per_frame",  # Movement speed — from raw spots
    "Circ_cell",           # Cell circularity — from raw spots
    "Circ_nuc",            # Nucleus circularity — NaN→0
]
FEATURE_NAMES = [
    "GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
    "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc",
]
TARGET_COL = "Mean.ch3"  # mCherry — forecast target

# Columns that are NaN when nucleus not segmented → fill with 0
NUC_COLS = ["Mean.ch1_nuc", "Area_nuc", "Circ_nuc"]

lines = []  # collected for tensor_summary.txt
def log(msg=""):
    print(msg, flush=True)
    lines.append(msg)

log("=" * 60)
log("02_build_3d_table.py  — HCMV Forecast tensor")
log("=" * 60)

# ── Load data ─────────────────────────────────────────────────
log("\nLoading timeseries_data.csv ...")
ts = pd.read_csv(TS_CSV, low_memory=False)
log(f"  {len(ts):,} rows, {ts['Track.ID'].nunique()} unique cells")

log("Loading model_df.csv ...")
md = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red", "abs_gfp_onset_min",
                           "movie_half_min", "dataset"]]

log("Loading extra_features.csv ...")
extra = pd.read_csv(EXTRA_CSV)
log(f"  {len(extra):,} rows, {extra['Track.ID'].nunique()} unique cells")

# ── Merge extra features ───────────────────────────────────────
ts = ts.merge(extra[["Track.ID", "Frame", "speed_px_per_frame",
                      "Mean_ch4", "Circ_cell"]],
              on=["Track.ID", "Frame"], how="left")

n_speed_nan = ts["speed_px_per_frame"].isna().sum()
log(f"\nSpeed NaN after merge: {n_speed_nan} ({n_speed_nan/len(ts)*100:.2f}%) — "
    f"tracks in timeseries not found in raw spots")

# ── Filter to productive, first-half cells ─────────────────────
log("\nFiltering cells ...")
prod_ids = md.dropna(subset=["delay_green_to_red"])
prod_ids = prod_ids[prod_ids["delay_green_to_red"].notna() &
                    np.isfinite(prod_ids["delay_green_to_red"])]
prod_ids = prod_ids[prod_ids["abs_gfp_onset_min"] <= prod_ids["movie_half_min"]]

log(f"  Productive, first-half cells: {len(prod_ids)}")

ts_prod = ts[ts["Track.ID"].isin(prod_ids["Track.ID"])].copy()
ts_prod = ts_prod.merge(
    prod_ids[["Track.ID", "delay_green_to_red"]],
    on="Track.ID", how="left"
)

def assign_group(d):
    return "early" if d <= EARLY_CUT else ("medium" if d <= LATE_CUT else "late")

ts_prod["group"] = ts_prod["delay_green_to_red"].map(assign_group)

log(f"  Rows: {len(ts_prod):,}")

# ── NaN audit ─────────────────────────────────────────────────
log("\n--- NaN audit (before zero-fill) ---")
for col in FEATURE_COLS + [TARGET_COL]:
    n = ts_prod[col].isna().sum()
    log(f"  {col:28s}  {n:6,}  ({n/len(ts_prod)*100:.2f}%)")

# ── Flags ─────────────────────────────────────────────────────
log("\n--- Feature flags ---")
log("  ✓ BF mean intensity (Mean_ch4) extracted from raw spots — available.")
log("  ✓ Cell circularity (Circ_cell) extracted from raw spots — available.")
log(f"  ⚠ Nucleus NaN frames (segmentation gaps) zero-padded: "
    f"{ts_prod['Mean.ch1_nuc'].isna().sum():,} rows "
    f"({ts_prod['Mean.ch1_nuc'].isna().mean()*100:.2f}%)")
if n_speed_nan > 0:
    log(f"  ⚠ Speed NaN after merge: {n_speed_nan} rows — "
        f"these will be zero-filled.")

# ── Zero-fill all NaN in feature and target columns ──────────
# Nucleus columns have systematic segmentation gaps (~8% of rows).
# Ctrst.ch4 has a single NaN. All are zero-filled so the tensor is dense.
for col in FEATURE_COLS + [TARGET_COL]:
    n_before = ts_prod[col].isna().sum()
    if n_before > 0:
        ts_prod[col] = ts_prod[col].fillna(0.0)
        log(f"  Zero-filled {col}: {n_before} NaN")

remaining_nan = {c: ts_prod[c].isna().sum()
                 for c in FEATURE_COLS + [TARGET_COL]
                 if ts_prod[c].isna().sum() > 0}
if remaining_nan:
    log("\n  ⚠ Remaining NaN after zero-fill (unexpected — check data):")
    for col, n in remaining_nan.items():
        log(f"    {col}: {n}")
else:
    log("  ✓ No remaining NaN in feature/target columns after zero-fill.")

# ── Build per-cell arrays ──────────────────────────────────────
log("\nBuilding per-cell sorted arrays ...")
ts_prod = ts_prod.sort_values(["Track.ID", "T_min"])

cell_ids    = prod_ids["Track.ID"].tolist()
n_cells     = len(cell_ids)
track_lengths = ts_prod.groupby("Track.ID").size()
max_frames  = int(track_lengths.max())

log(f"  Cells:      {n_cells}")
log(f"  max_frames: {max_frames}  (min={track_lengths.min()}, "
    f"median={int(track_lengths.median())})")

n_feat = len(FEATURE_COLS)
X = np.zeros((n_cells, max_frames, n_feat), dtype=np.float32)
Y = np.zeros((n_cells, max_frames),         dtype=np.float32)

cell_index = {tid: i for i, tid in enumerate(cell_ids)}

for tid, grp in ts_prod.groupby("Track.ID"):
    if tid not in cell_index:
        continue
    i = cell_index[tid]
    n = len(grp)
    X[i, :n, :] = grp[FEATURE_COLS].values.astype(np.float32)
    Y[i, :n]    = grp[TARGET_COL].values.astype(np.float32)

log(f"\nTensor X shape: {X.shape}  (cells × frames × features)")
log(f"Tensor Y shape: {Y.shape}  (cells × frames, mCherry target)")

# ── Per-feature stats ──────────────────────────────────────────
log("\n--- Feature value ranges (over non-padded entries) ---")
# Mask padded zeros: use actual frame counts
for fi, (col, name) in enumerate(zip(FEATURE_COLS, FEATURE_NAMES)):
    vals = []
    for i, tid in enumerate(cell_ids):
        n = track_lengths.get(tid, 0)
        vals.append(X[i, :n, fi])
    v = np.concatenate(vals)
    log(f"  {name:15s}  min={np.nanmin(v):.4g}  max={np.nanmax(v):.4g}  "
        f"mean={np.nanmean(v):.4g}  std={np.nanstd(v):.4g}")

# ── Group counts ──────────────────────────────────────────────
grp_counts = ts_prod.drop_duplicates("Track.ID")["group"].value_counts()
log("\n--- Group counts (productive, first-half) ---")
for g, n in grp_counts.items():
    log(f"  {g:10s}: {n}")

# ── Cell metadata ──────────────────────────────────────────────
meta = prod_ids[["Track.ID", "delay_green_to_red", "dataset"]].copy()
meta["group"]        = meta["delay_green_to_red"].map(assign_group)
meta["n_real_frames"] = meta["Track.ID"].map(track_lengths)
meta["cell_index"]   = meta["Track.ID"].map(cell_index)
meta = meta.sort_values("cell_index").reset_index(drop=True)

meta_path = OUT_DIR / "cell_metadata.csv"
meta.to_csv(meta_path, index=False)
log(f"\nSaved → {meta_path}")

# ── Save tensors ───────────────────────────────────────────────
x_path = OUT_DIR / "tensor_X.npz"
y_path = OUT_DIR / "tensor_y_mcherry.npz"
np.savez_compressed(x_path, X=X, feature_names=np.array(FEATURE_NAMES))
np.savez_compressed(y_path, Y=Y)
log(f"Saved → {x_path}")
log(f"Saved → {y_path}")

# ── Summary txt ───────────────────────────────────────────────
summary_path = OUT_DIR / "tensor_summary.txt"
summary_path.write_text("\n".join(lines))
log(f"\nSaved → {summary_path}")
log("\nAll done.")

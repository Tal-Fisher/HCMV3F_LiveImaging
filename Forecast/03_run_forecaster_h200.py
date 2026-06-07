"""
03_run_forecaster_h200.py  —  TabICLForecaster: H=200 frames (3000 min / 50 h)

Identical setup to 03_run_forecaster.py (same tensors, same 75/25 stratified
split with RANDOM_STATE=42, same RED_THRESHOLD calibration from training cells).
Only runs horizon H=200.

Outputs:
  Forecast/predictions_h200.csv
  Forecast/forecast_scatter_h200.png
  Forecast/eval_summary.csv  (H=200 row appended / updated)

Run via:
  bsub < submit_forecast_h200.sh
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
from tabicl import TabICLForecaster

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT_DIR = BASE / "Forecast"
TS_CSV  = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"

CONTEXT_LEN   = 16
H             = 200                 # single horizon
H_MIN         = H * 15             # 3000 min = 50 h
STRIDE        = 3
MIN_PER_FRAME = 15
RANDOM_STATE  = 42

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

FEAT_NAMES = ["GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
              "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc"]

# ── Load tensors & metadata ───────────────────────────────────────────────────
print("Loading tensors ...", flush=True)
X_tensor = np.load(OUT_DIR / "tensor_X.npz")["X"]          # (n_cells, max_frames, 9)
Y_tensor = np.load(OUT_DIR / "tensor_y_mcherry.npz")["Y"]  # (n_cells, max_frames)

_meta = pd.read_csv(OUT_DIR / "cell_metadata.csv")
_md   = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
_meta = _meta.merge(_md, on="Track.ID", how="left")
meta  = _meta.set_index("Track.ID")
print(f"  {len(meta)} cells, tensor shape {X_tensor.shape}", flush=True)

# ── Frame timing ──────────────────────────────────────────────────────────────
print("Loading frame timing ...", flush=True)
ts = pd.read_csv(TS_CSV, usecols=["Track.ID", "T_min"], low_memory=False)
ts = ts.sort_values(["Track.ID", "T_min"])
md = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
ts = ts.merge(md, on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts.groupby("Track.ID")}

# ── Same 75/25 stratified split as original run ───────────────────────────────
cell_info = meta.reset_index()[["Track.ID", "group"]].drop_duplicates("Track.ID")
splitter  = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                   random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(cell_info, cell_info["group"]))
train_cells = set(cell_info.iloc[train_idx]["Track.ID"])
test_cells  = set(cell_info.iloc[test_idx]["Track.ID"])
print(f"  Train cells: {len(train_cells)}, Test cells: {len(test_cells)}", flush=True)

# ── Calibrate threshold from training cells (same as original) ───────────────
print("Calibrating mCherry threshold ...", flush=True)
threshold_vals = []
for i, tid in enumerate(meta.index):
    if tid not in train_cells or tid not in cell_tmin:
        continue
    delay   = meta.loc[tid, "delay_green_to_red"]
    gfp_min = meta.loc[tid, "abs_gfp_onset_min"]
    red_min = gfp_min + delay
    t_arr   = cell_tmin[tid]
    red_frame = int(np.argmin(np.abs(t_arr - red_min)))
    if red_frame < len(t_arr):
        threshold_vals.append(float(Y_tensor[i, red_frame]))

RED_THRESHOLD = float(np.median(threshold_vals))
print(f"  mCherry red threshold: {RED_THRESHOLD:.3f}", flush=True)

# ── Build windows for H=200 ───────────────────────────────────────────────────
print(f"\nBuilding test windows for H={H} ({H_MIN} min = {H_MIN/60:.0f} h) ...",
      flush=True)
wins_h = []
for i, tid in enumerate(meta.index):
    if tid not in test_cells or tid not in cell_tmin:
        continue
    t_arr    = cell_tmin[tid]
    n_frames = len(t_arr)
    delay    = meta.loc[tid, "delay_green_to_red"]
    gfp_min  = meta.loc[tid, "abs_gfp_onset_min"]
    group    = meta.loc[tid, "group"]
    red_min  = gfp_min + delay

    for w_start in range(0, n_frames - CONTEXT_LEN, STRIDE):
        w_end     = w_start + CONTEXT_LEN
        t_ctx_end = t_arr[w_end - 1]
        remaining = red_min - t_ctx_end
        n_future  = n_frames - w_end
        if remaining < H_MIN or n_future < H:
            continue

        ts_ctx = pd.date_range(start="2000-01-01", periods=CONTEXT_LEN,
                               freq=f"{MIN_PER_FRAME}min")
        wins_h.append({
            "item_id":       f"{tid}_w{w_start}",
            "Track.ID":      tid,
            "group":         group,
            "w_start":       w_start,
            "remaining_min": remaining,
            "ts_ctx":        ts_ctx,
            "X_ctx":         X_tensor[i, w_start:w_end, :],
            "Y_ctx":         Y_tensor[i, w_start:w_end],
            "Y_future":      Y_tensor[i, w_end:],
            "n_future":      n_future,
        })

print(f"  Qualifying windows: {len(wins_h):,}", flush=True)
if len(wins_h) == 0:
    print("No qualifying windows for H=200 — exiting.", flush=True)
    raise SystemExit(0)

# ── Build context_df ──────────────────────────────────────────────────────────
ctx_rows = []
for w in wins_h:
    for t_idx in range(CONTEXT_LEN):
        row = {"item_id":   w["item_id"],
               "timestamp": w["ts_ctx"][t_idx],
               "target":    float(w["Y_ctx"][t_idx])}
        for fi, fn in enumerate(FEAT_NAMES):
            row[fn] = float(w["X_ctx"][t_idx, fi])
        ctx_rows.append(row)
context_df = pd.DataFrame(ctx_rows)
print(f"  context_df shape: {context_df.shape}", flush=True)

# ── Run TabICLForecaster ──────────────────────────────────────────────────────
forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)
print(f"  Running TabICLForecaster.predict_df(prediction_length={H}) ...",
      flush=True)
pred_df = forecaster.predict_df(context_df, prediction_length=H)
print(f"  Predictions shape: {pred_df.shape}", flush=True)

# ── Extract per-window time-to-red ────────────────────────────────────────────
pred_grp = pred_df.groupby(level=0)["target"].apply(list).to_dict()
all_preds = []
for w in wins_h:
    iid = w["item_id"]
    if iid not in pred_grp:
        continue
    pred_traj  = np.array(pred_grp[iid])
    cross_idx  = np.where(pred_traj >= RED_THRESHOLD)[0]
    pred_remaining = (float(cross_idx[0]) * MIN_PER_FRAME
                      if len(cross_idx) > 0 else float(H) * MIN_PER_FRAME)
    all_preds.append({
        "Track.ID":             w["Track.ID"],
        "group":                w["group"],
        "w_start":              w["w_start"],
        "horizon_frames":       H,
        "remaining_min_actual": w["remaining_min"],
        "pred_remaining_min":   pred_remaining,
    })

sub = pd.DataFrame(all_preds)
sub.to_csv(OUT_DIR / f"predictions_h{H:03d}.csv", index=False)
print(f"  Saved predictions_h{H:03d}.csv ({len(sub):,} rows)", flush=True)

# ── Evaluate ──────────────────────────────────────────────────────────────────
mae_all  = mean_absolute_error(sub["remaining_min_actual"], sub["pred_remaining_min"])
rmse_all = rmse(sub["remaining_min_actual"], sub["pred_remaining_min"])
print(f"  Overall  MAE={mae_all:.1f} min  RMSE={rmse_all:.1f} min", flush=True)

eval_rows = []
for g in ["early", "medium", "late"]:
    sg = sub[sub["group"] == g]
    if len(sg) < 3:
        continue
    mae_g  = mean_absolute_error(sg["remaining_min_actual"], sg["pred_remaining_min"])
    rmse_g = rmse(sg["remaining_min_actual"], sg["pred_remaining_min"])
    print(f"    {g:8s}  n={len(sg):4,}  MAE={mae_g:.1f}  RMSE={rmse_g:.1f} min",
          flush=True)
    eval_rows.append({"horizon_frames": H, "group": g,
                      "n_windows":      len(sg),
                      "MAE_min":        round(mae_g,  1),
                      "RMSE_min":       round(rmse_g, 1)})

# append H=200 rows to existing eval_summary.csv
eval_summary_path = OUT_DIR / "eval_summary.csv"
if eval_summary_path.exists():
    existing = pd.read_csv(eval_summary_path)
    existing = existing[existing["horizon_frames"] != H]   # remove any prior H=200
    updated  = pd.concat([existing, pd.DataFrame(eval_rows)], ignore_index=True)
else:
    updated = pd.DataFrame(eval_rows)
updated.to_csv(eval_summary_path, index=False)
print(f"  Updated eval_summary.csv", flush=True)

# ── Scatter plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for g, col in GROUP_COLORS.items():
    sg = sub[sub["group"] == g]
    if len(sg):
        ax.scatter(sg["remaining_min_actual"] / 60,
                   sg["pred_remaining_min"]   / 60,
                   color=col, alpha=0.35, s=12, label=f"{g} (n={len(sg)})")

lim = max(sub["remaining_min_actual"].max(),
          sub["pred_remaining_min"].max()) / 60 * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
ax.set_xlabel("Actual remaining time to red (h)")
ax.set_ylabel("Predicted remaining time to red (h)")
ax.set_title(
    f"TabICL Forecast  H={H} frames ({H_MIN/60:.0f} h ahead)\n"
    f"n={len(sub):,} windows   MAE={mae_all:.0f} min   RMSE={rmse_all:.0f} min",
    fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / f"forecast_scatter_h{H:03d}.png", dpi=150)
plt.close()
print(f"  Saved forecast_scatter_h{H:03d}.png", flush=True)

print("All done.", flush=True)

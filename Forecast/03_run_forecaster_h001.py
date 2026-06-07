"""
03_run_forecaster_h001.py  —  TabICLForecaster: H=1 (one frame = 15 min ahead)

Same setup as 03_run_forecaster.py but with HORIZONS=[1] only.
Saves:
  predictions_h001.csv
  trajectories_h001.npz
  forecast_scatter_h001.png
  trajectory_plot_h001.png
  tabicl_forecaster_h001.pkl
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
HORIZONS      = [1]
STRIDE        = 3
MIN_PER_FRAME = 15
RANDOM_STATE  = 42

GROUP_COLORS  = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

FEAT_NAMES = ["GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
              "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc"]

# ── Load tensors & metadata ───────────────────────────────────────────────────
print("Loading tensors ...", flush=True)
X_tensor = np.load(OUT_DIR / "tensor_X.npz")["X"]
Y_tensor = np.load(OUT_DIR / "tensor_y_mcherry.npz")["Y"]

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

# ── Same 75/25 stratified split as original ───────────────────────────────────
cell_info = meta.reset_index()[["Track.ID", "group"]].drop_duplicates("Track.ID")
splitter  = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(cell_info, cell_info["group"]))
train_cells = set(cell_info.iloc[train_idx]["Track.ID"])
test_cells  = set(cell_info.iloc[test_idx]["Track.ID"])
print(f"  Train: {len(train_cells)}  Test: {len(test_cells)}", flush=True)

# ── Calibrate red threshold from training cells ───────────────────────────────
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
print(f"  RED_THRESHOLD = {RED_THRESHOLD:.3f}", flush=True)

# ── Build windows ─────────────────────────────────────────────────────────────
def build_windows(cell_ids, context_len, stride):
    for i, tid in enumerate(meta.index):
        if tid not in cell_ids or tid not in cell_tmin:
            continue
        t_arr    = cell_tmin[tid]
        n_frames = len(t_arr)
        delay    = meta.loc[tid, "delay_green_to_red"]
        gfp_min  = meta.loc[tid, "abs_gfp_onset_min"]
        group    = meta.loc[tid, "group"]
        red_min  = gfp_min + delay

        for w_start in range(0, n_frames - context_len, stride):
            w_end     = w_start + context_len
            t_ctx_end = t_arr[w_end - 1]
            remaining = red_min - t_ctx_end
            if remaining <= 0:
                continue
            ts_ctx = pd.date_range(start="2000-01-01", periods=context_len,
                                   freq=f"{MIN_PER_FRAME}min")
            yield {
                "item_id":       f"{tid}_w{w_start}",
                "Track.ID":      tid,
                "group":         group,
                "w_start":       w_start,
                "remaining_min": remaining,
                "ts_ctx":        ts_ctx,
                "X_ctx":         X_tensor[i, w_start:w_end, :],
                "Y_ctx":         Y_tensor[i, w_start:w_end],
                "Y_future":      Y_tensor[i, w_end:],
                "n_future":      n_frames - w_end,
            }

print("\nBuilding test windows ...", flush=True)
test_wins = list(build_windows(test_cells, CONTEXT_LEN, STRIDE))
print(f"  Test windows: {len(test_wins):,}", flush=True)

# ── Run forecaster ────────────────────────────────────────────────────────────
forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)
with open(OUT_DIR / "tabicl_forecaster_h001.pkl", "wb") as f:
    pickle.dump(forecaster, f)
print("Saved tabicl_forecaster_h001.pkl", flush=True)

H = 1
H_min = H * MIN_PER_FRAME

wins_h = [w for w in test_wins if w["remaining_min"] >= H_min and w["n_future"] >= H]
print(f"\nHorizon H={H} ({H_min} min): {len(wins_h):,} windows", flush=True)

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
print(f"  Running predict_df(prediction_length=1) ...", flush=True)

pred_df = forecaster.predict_df(context_df, prediction_length=1)
print(f"  Predictions shape: {pred_df.shape}", flush=True)

pred_grp = pred_df.groupby(level=0)["target"].apply(list).to_dict()

# ── Save trajectories (just 1 value each) ────────────────────────────────────
traj_dict = {}
all_preds = []

for w in wins_h:
    iid = w["item_id"]
    if iid not in pred_grp:
        continue
    pred_val = float(pred_grp[iid][0])   # single predicted mCherry value
    traj_dict[iid] = np.array([pred_val])

    actual_next = float(w["Y_future"][0]) if len(w["Y_future"]) > 0 else np.nan
    all_preds.append({
        "Track.ID":              w["Track.ID"],
        "group":                 w["group"],
        "w_start":               w["w_start"],
        "horizon_frames":        H,
        "remaining_min_actual":  w["remaining_min"],
        "pred_mcherry_next":     pred_val,
        "actual_mcherry_next":   actual_next,
        "actual_mcherry_ctx_last": float(w["Y_ctx"][-1]),
    })

np.savez_compressed(OUT_DIR / "trajectories_h001.npz", **traj_dict)
print("Saved trajectories_h001.npz", flush=True)

preds_df = pd.DataFrame(all_preds)
preds_df.to_csv(OUT_DIR / "predictions_h001.csv", index=False)
print(f"Saved predictions_h001.csv ({len(preds_df):,} rows)", flush=True)

# ── Evaluation: predicted vs actual next mCherry ─────────────────────────────
preds_clean = preds_df.dropna(subset=["actual_mcherry_next"])
mae_all  = mean_absolute_error(preds_clean["actual_mcherry_next"],
                               preds_clean["pred_mcherry_next"])
rmse_all = rmse(preds_clean["actual_mcherry_next"], preds_clean["pred_mcherry_next"])
print(f"\nOverall  MAE={mae_all:.4f}  RMSE={rmse_all:.4f}", flush=True)

eval_rows = []
for g in ["early", "medium", "late"]:
    sg = preds_clean[preds_clean["group"] == g]
    if len(sg) < 3:
        continue
    mae_g  = mean_absolute_error(sg["actual_mcherry_next"], sg["pred_mcherry_next"])
    rmse_g = rmse(sg["actual_mcherry_next"], sg["pred_mcherry_next"])
    naive_mae = mean_absolute_error(sg["actual_mcherry_next"], sg["actual_mcherry_ctx_last"])
    print(f"  {g:8s}  n={len(sg):4,}  MAE={mae_g:.4f}  RMSE={rmse_g:.4f}  "
          f"naive_MAE(last)={naive_mae:.4f}", flush=True)
    eval_rows.append({"horizon_frames": H, "group": g, "n_windows": len(sg),
                      "MAE": round(mae_g, 4), "RMSE": round(rmse_g, 4),
                      "naive_MAE_last": round(naive_mae, 4)})

pd.DataFrame(eval_rows).to_csv(OUT_DIR / "eval_summary_h001.csv", index=False)

# ── Scatter: predicted vs actual next mCherry ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("H=1 (15 min ahead): predicted vs actual mCherry", fontsize=11, fontweight="bold")

for ax, g in zip(axes, ["early", "medium", "late"]):
    sg = preds_clean[preds_clean["group"] == g]
    col = GROUP_COLORS[g]
    ax.scatter(sg["actual_mcherry_next"], sg["pred_mcherry_next"],
               color=col, alpha=0.3, s=8, linewidths=0)
    lim_max = max(sg["actual_mcherry_next"].max(), sg["pred_mcherry_next"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, alpha=0.5)
    ax.axhline(RED_THRESHOLD, color="red", lw=0.8, linestyle=":", alpha=0.7)
    ax.axvline(RED_THRESHOLD, color="red", lw=0.8, linestyle=":", alpha=0.7)
    mae_g = mean_absolute_error(sg["actual_mcherry_next"], sg["pred_mcherry_next"])
    r = np.corrcoef(sg["actual_mcherry_next"], sg["pred_mcherry_next"])[0,1]
    ax.set_title(f"{g}  n={len(sg):,}\nMAE={mae_g:.3f}  r={r:.3f}", fontsize=9)
    ax.set_xlabel("Actual mCherry (next frame)", fontsize=8)
    ax.set_ylabel("Predicted mCherry (next frame)", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "forecast_scatter_h001.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved forecast_scatter_h001.png", flush=True)

print("\nAll done.", flush=True)

"""
03_run_forecaster.py  —  TabICLForecaster: predict mCherry trajectory

TabICLForecaster is a zero-shot time-series forecaster (no fit step).
For each cell window it takes the last CONTEXT_LEN=16 frames as context
and predicts the next N frames of mCherry (target), treating the other
8 features as covariates.

Workflow:
  1. For each test cell, slide a 16-frame window (stride STRIDE) over the track.
  2. Batch all windows as separate item_ids into one predict_df() call per horizon.
  3. From the predicted mCherry trajectory, find the first frame crossing
     the red threshold → predicted time-to-red.
  4. Compare to actual time-to-red, compute MAE/RMSE per group per horizon.

Train/test split (cell level, 75/25, stratified by group):
  - Test cells are used for evaluation only.
  - Training cells are used solely to calibrate the mCherry red threshold.

Outputs:
  Forecast/predictions_hXX.csv      — per-window predictions + actual
  Forecast/eval_summary.csv         — MAE / RMSE per group per horizon
  Forecast/forecast_scatter_hXXX.png
  Forecast/forecast_mcherry_hXXX.png — example predicted trajectories

Run with:
  /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12 03_run_forecaster.py
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
HORIZONS      = [10, 50, 100]       # frames ahead
STRIDE        = 3                   # window stride (reduce batch size on CPU)
MIN_PER_FRAME = 15                  # acquisition interval
RANDOM_STATE  = 42

GROUP_COLORS  = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

FEAT_NAMES = ["GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
              "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc"]
MCHERRY_IDX = None  # mCherry is in the Y tensor (not in X features)

# ── Load tensors & metadata ───────────────────────────────────
print("Loading tensors ...", flush=True)
X_data    = np.load(OUT_DIR / "tensor_X.npz")
X_tensor  = X_data["X"]                     # (n_cells, max_frames, 9)
Y_tensor  = np.load(OUT_DIR / "tensor_y_mcherry.npz")["Y"]  # (n_cells, max_frames)

_meta = pd.read_csv(OUT_DIR / "cell_metadata.csv")
_md   = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
_meta = _meta.merge(_md, on="Track.ID", how="left")
meta  = _meta.set_index("Track.ID")
print(f"  {len(meta)} cells, tensor shape {X_tensor.shape}", flush=True)

# ── Load per-cell frame timing ────────────────────────────────
print("Loading frame timing ...", flush=True)
ts = pd.read_csv(TS_CSV, usecols=["Track.ID", "T_min"], low_memory=False)
ts = ts.sort_values(["Track.ID", "T_min"])
md = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
ts = ts.merge(md, on="Track.ID", how="left")

cell_tmin = {tid: grp["T_min"].values
             for tid, grp in ts.groupby("Track.ID")}

# ── Cell-level 75/25 stratified split ────────────────────────
cell_info = meta.reset_index()[["Track.ID", "group"]].drop_duplicates("Track.ID")
splitter  = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                   random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(cell_info, cell_info["group"]))
train_cells = set(cell_info.iloc[train_idx]["Track.ID"])
test_cells  = set(cell_info.iloc[test_idx]["Track.ID"])
print(f"  Train cells: {len(train_cells)}, Test cells: {len(test_cells)}", flush=True)
split_df = cell_info.copy()
split_df["split"] = split_df["Track.ID"].apply(
    lambda t: "train" if t in train_cells else "test")
split_df.to_csv(OUT_DIR / "train_test_split.csv", index=False)
print("  Saved train_test_split.csv", flush=True)

# ── Calibrate mCherry red-onset threshold from training cells ─
print("Calibrating mCherry threshold ...", flush=True)
threshold_vals = []
for i, tid in enumerate(meta.index):
    if tid not in train_cells or tid not in cell_tmin:
        continue
    delay   = meta.loc[tid, "delay_green_to_red"]
    gfp_min = meta.loc[tid, "abs_gfp_onset_min"]
    red_min = gfp_min + delay
    t_arr   = cell_tmin[tid]
    # find tensor frame closest to red_onset_min
    red_frame = int(np.argmin(np.abs(t_arr - red_min)))
    if red_frame < len(t_arr):
        threshold_vals.append(float(Y_tensor[i, red_frame]))

RED_THRESHOLD = float(np.median(threshold_vals))
print(f"  mCherry red threshold (median at red onset, training cells): "
      f"{RED_THRESHOLD:.3f}", flush=True)

# ── Build per-window records for test cells ───────────────────
def build_windows(cell_ids, context_len, stride):
    """Yield dicts with window metadata + arrays for context_df construction."""
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
            w_end      = w_start + context_len
            t_ctx_end  = t_arr[w_end - 1]
            remaining  = red_min - t_ctx_end   # minutes to red from context end
            if remaining <= 0:
                continue

            # fake datetime timestamps spaced 15 min apart
            ts_ctx = pd.date_range(start="2000-01-01", periods=context_len,
                                   freq=f"{MIN_PER_FRAME}min")
            yield {
                "item_id":       f"{tid}_w{w_start}",
                "Track.ID":      tid,
                "group":         group,
                "w_start":       w_start,
                "remaining_min": remaining,
                "ts_ctx":        ts_ctx,
                "X_ctx":         X_tensor[i, w_start:w_end, :],  # (16, 9)
                "Y_ctx":         Y_tensor[i, w_start:w_end],      # (16,) mCherry
                "Y_future":      Y_tensor[i, w_end:],             # future mCherry
                "n_future":      n_frames - w_end,
            }

print("\nBuilding test windows ...", flush=True)
test_wins = list(build_windows(test_cells, CONTEXT_LEN, STRIDE))
print(f"  Test windows: {len(test_wins):,}", flush=True)

# ── Run forecaster for each horizon ───────────────────────────
all_preds = []
eval_rows = []

forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)
with open(OUT_DIR / "tabicl_forecaster.pkl", "wb") as f:
    pickle.dump(forecaster, f)
print("Saved forecaster object → tabicl_forecaster.pkl", flush=True)

for H in HORIZONS:
    H_min = H * MIN_PER_FRAME
    # only windows where red is at least H frames away
    wins_h = [w for w in test_wins if w["remaining_min"] >= H_min
              and w["n_future"] >= H]
    print(f"\nHorizon H={H} ({H_min} min): {len(wins_h):,} windows", flush=True)
    if not wins_h:
        continue

    # ── Build context_df ──────────────────────────────────────
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
    print(f"  Running TabICLForecaster.predict_df(prediction_length={H}) ...",
          flush=True)

    pred_df = forecaster.predict_df(context_df, prediction_length=H)
    print(f"  Predictions shape: {pred_df.shape}", flush=True)

    # ── Extract per-window predicted mCherry trajectory ───────
    # pred_df is indexed by (item_id, timestamp); 'target' = predicted mCherry
    pred_grp = pred_df.groupby(level=0)["target"].apply(list).to_dict()

    for w in wins_h:
        iid   = w["item_id"]
        if iid not in pred_grp:
            continue
        pred_traj = np.array(pred_grp[iid])   # predicted mCherry, H steps

        # predicted time-to-red: first frame crossing RED_THRESHOLD
        cross_idx = np.where(pred_traj >= RED_THRESHOLD)[0]
        pred_remaining = (cross_idx[0] * MIN_PER_FRAME
                          if len(cross_idx) > 0
                          else H * MIN_PER_FRAME)   # no crossing → cap at H

        all_preds.append({
            "Track.ID":           w["Track.ID"],
            "group":              w["group"],
            "w_start":            w["w_start"],
            "horizon_frames":     H,
            "remaining_min_actual": w["remaining_min"],
            "pred_remaining_min": pred_remaining,
        })

    # ── Evaluate ──────────────────────────────────────────────
    sub = pd.DataFrame([p for p in all_preds if p["horizon_frames"] == H])
    if len(sub) == 0:
        continue

    # save predictions first, before any eval that could fail
    sub.to_csv(OUT_DIR / f"predictions_h{H:03d}.csv", index=False)
    print(f"  Saved predictions_h{H:03d}.csv ({len(sub):,} rows)", flush=True)

    mae_all  = mean_absolute_error(sub["remaining_min_actual"],
                                   sub["pred_remaining_min"])
    rmse_all = rmse(sub["remaining_min_actual"], sub["pred_remaining_min"])
    print(f"  Overall  MAE={mae_all:.1f} min  RMSE={rmse_all:.1f} min", flush=True)

    for g in ["early", "medium", "late"]:
        sg = sub[sub["group"] == g]
        if len(sg) < 3:
            continue
        mae_g = mean_absolute_error(sg["remaining_min_actual"],
                                    sg["pred_remaining_min"])
        rmse_g = rmse(sg["remaining_min_actual"], sg["pred_remaining_min"])
        print(f"    {g:8s}  n={len(sg):4,}  MAE={mae_g:.1f}  RMSE={rmse_g:.1f} min",
              flush=True)
        eval_rows.append({"horizon_frames": H, "group": g,
                          "n_windows": len(sg),
                          "MAE_min": round(mae_g, 1),
                          "RMSE_min": round(rmse_g, 1)})

    # ── Scatter plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for g, col in GROUP_COLORS.items():
        sg = sub[sub["group"] == g]
        if len(sg):
            ax.scatter(sg["remaining_min_actual"] / 60,
                       sg["pred_remaining_min"] / 60,
                       color=col, alpha=0.35, s=12, label=g)
    lim = max(sub["remaining_min_actual"].max(),
              sub["pred_remaining_min"].max()) / 60 * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Actual remaining time to red (h)")
    ax.set_ylabel("Predicted remaining time to red (h)")
    ax.set_title(f"H={H} frames ({H_min/60:.0f} h ahead)  n={len(sub):,} windows",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"forecast_scatter_h{H:03d}.png", dpi=150)
    plt.close()

# ── Save eval summary ─────────────────────────────────────────
if eval_rows:
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT_DIR / "eval_summary.csv", index=False)
    print(f"\nSaved eval_summary.csv", flush=True)

print("All done.", flush=True)

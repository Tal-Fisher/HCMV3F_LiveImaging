"""
03_run_forecaster_relthresh.py  —  TabICL forecast with relative threshold

Identical setup to 03_run_forecaster.py.  Two changes:
  1. The full predicted mCherry trajectory is saved per window so the
     threshold can be adjusted post-hoc without rerunning inference.
  2. Time-to-red is detected using a RELATIVE threshold:
         crossing when pred_mcherry >= context_mean + K * context_std
     where context_mean / context_std come from the 16-frame context window.
     This breaks the collapse to H*15 caused by the fixed absolute threshold.

Outputs (per horizon H):
  Forecast/predictions_relthresh_hXXX.csv   — per-window results
  Forecast/trajectories_hXXX.npz            — full predicted trajectories
  Forecast/forecast_scatter_relthresh_hXXX.png
  Forecast/eval_summary_relthresh.csv

K values tried: K_VALUES list below (plots generated for each).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabicl import TabICLForecaster

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT_DIR = BASE / "Forecast"
TS_CSV  = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"

CONTEXT_LEN   = 16
HORIZONS      = [10, 50, 100, 200]
STRIDE        = 3
MIN_PER_FRAME = 15
RANDOM_STATE  = 42
K_VALUES      = [2, 3, 5]       # multiples of context SD above context mean
K_PLOT        = 3               # which K to use for the main scatter plot

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}
FEAT_NAMES   = ["GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
                "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading tensors ...", flush=True)
X_tensor = np.load(OUT_DIR / "tensor_X.npz")["X"]
Y_tensor = np.load(OUT_DIR / "tensor_y_mcherry.npz")["Y"]

_meta = pd.read_csv(OUT_DIR / "cell_metadata.csv")
_md   = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
_meta = _meta.merge(_md, on="Track.ID", how="left")
meta  = _meta.set_index("Track.ID")
print(f"  {len(meta)} cells, tensor shape {X_tensor.shape}", flush=True)

ts = pd.read_csv(TS_CSV, usecols=["Track.ID", "T_min"], low_memory=False)
ts = ts.sort_values(["Track.ID", "T_min"])
ts = ts.merge(pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"]),
              on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts.groupby("Track.ID")}

# ── Same train/test split ─────────────────────────────────────────────────────
cell_info = meta.reset_index()[["Track.ID", "group"]].drop_duplicates("Track.ID")
splitter  = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                   random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(cell_info, cell_info["group"]))
train_cells = set(cell_info.iloc[train_idx]["Track.ID"])
test_cells  = set(cell_info.iloc[test_idx]["Track.ID"])
print(f"  Train: {len(train_cells)}, Test: {len(test_cells)}", flush=True)

# ── Absolute threshold (kept for reference only) ──────────────────────────────
threshold_vals = []
for i, tid in enumerate(meta.index):
    if tid not in train_cells or tid not in cell_tmin:
        continue
    delay   = meta.loc[tid, "delay_green_to_red"]
    gfp_min = meta.loc[tid, "abs_gfp_onset_min"]
    red_min = gfp_min + delay
    t_arr   = cell_tmin[tid]
    rf      = int(np.argmin(np.abs(t_arr - red_min)))
    if rf < len(t_arr):
        threshold_vals.append(float(Y_tensor[i, rf]))
ABS_THRESHOLD = float(np.median(threshold_vals))
print(f"  Absolute threshold (reference): {ABS_THRESHOLD:.3f}", flush=True)

# ── Build windows helper ──────────────────────────────────────────────────────
def build_windows(cell_ids, H):
    H_min = H * MIN_PER_FRAME
    wins  = []
    for i, tid in enumerate(meta.index):
        if tid not in cell_ids or tid not in cell_tmin:
            continue
        t_arr    = cell_tmin[tid]
        n_frames = len(t_arr)
        delay    = meta.loc[tid, "delay_green_to_red"]
        gfp_min  = meta.loc[tid, "abs_gfp_onset_min"]
        group    = meta.loc[tid, "group"]
        red_min  = gfp_min + delay

        for w_start in range(0, n_frames - CONTEXT_LEN, STRIDE):
            w_end      = w_start + CONTEXT_LEN
            t_ctx_end  = t_arr[w_end - 1]
            remaining  = red_min - t_ctx_end
            n_future   = n_frames - w_end
            if remaining < H_min or n_future < H:
                continue

            Y_ctx = Y_tensor[i, w_start:w_end]
            wins.append({
                "item_id":        f"{tid}_w{w_start}",
                "cell_idx":       i,
                "Track.ID":       tid,
                "group":          group,
                "w_start":        w_start,
                "remaining_min":  remaining,
                "ts_ctx":         pd.date_range(start="2000-01-01",
                                                periods=CONTEXT_LEN,
                                                freq=f"{MIN_PER_FRAME}min"),
                "X_ctx":          X_tensor[i, w_start:w_end, :],
                "Y_ctx":          Y_ctx,
                "ctx_mean":       float(np.mean(Y_ctx)),
                "ctx_std":        float(np.std(Y_ctx)) if np.std(Y_ctx) > 1e-6 else 1e-6,
                "n_future":       n_future,
            })
    return wins

# ── Time-to-crossing from predicted trajectory ────────────────────────────────
def time_to_crossing(pred_traj, threshold, H):
    idx = np.where(pred_traj >= threshold)[0]
    return float(idx[0]) * MIN_PER_FRAME if len(idx) > 0 else float(H) * MIN_PER_FRAME

# ── Main loop ─────────────────────────────────────────────────────────────────
forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)
all_eval   = []

for H in HORIZONS:
    H_min = H * MIN_PER_FRAME
    print(f"\n{'='*60}\nHorizon H={H} ({H_min} min = {H_min/60:.0f} h)", flush=True)

    wins = build_windows(test_cells, H)
    print(f"  Qualifying windows: {len(wins):,}", flush=True)
    if not wins:
        print("  No qualifying windows — skipping.", flush=True)
        continue

    # build context_df
    ctx_rows = []
    for w in wins:
        for t_idx in range(CONTEXT_LEN):
            row = {"item_id":   w["item_id"],
                   "timestamp": w["ts_ctx"][t_idx],
                   "target":    float(w["Y_ctx"][t_idx])}
            for fi, fn in enumerate(FEAT_NAMES):
                row[fn] = float(w["X_ctx"][t_idx, fi])
            ctx_rows.append(row)
    context_df = pd.DataFrame(ctx_rows)

    print(f"  context_df: {context_df.shape}  "
          f"  Running TabICLForecaster.predict_df(H={H}) ...", flush=True)
    pred_df  = forecaster.predict_df(context_df, prediction_length=H)
    pred_grp = pred_df.groupby(level=0)["target"].apply(np.array).to_dict()
    print(f"  Predictions done.", flush=True)

    # collect trajectories and compute crossing times
    traj_dict = {}   # item_id → predicted trajectory array
    rows = []
    for w in wins:
        iid = w["item_id"]
        if iid not in pred_grp:
            continue
        traj = pred_grp[iid]
        traj_dict[iid] = traj

        row = {
            "Track.ID":             w["Track.ID"],
            "group":                w["group"],
            "w_start":              w["w_start"],
            "horizon_frames":       H,
            "remaining_min_actual": w["remaining_min"],
            "ctx_mean":             w["ctx_mean"],
            "ctx_std":              w["ctx_std"],
            # absolute threshold (reference)
            "pred_abs":             time_to_crossing(traj, ABS_THRESHOLD, H),
        }
        for k in K_VALUES:
            thr_k = w["ctx_mean"] + k * w["ctx_std"]
            row[f"pred_k{k}"] = time_to_crossing(traj, thr_k, H)
        rows.append(row)

    sub = pd.DataFrame(rows)
    sub.to_csv(OUT_DIR / f"predictions_relthresh_h{H:03d}.csv", index=False)
    print(f"  Saved predictions_relthresh_h{H:03d}.csv ({len(sub):,} rows)",
          flush=True)

    # save full trajectories
    np.savez_compressed(
        OUT_DIR / f"trajectories_h{H:03d}.npz",
        **{iid.replace(".", "_").replace(" ", "_"): traj
           for iid, traj in traj_dict.items()})
    print(f"  Saved trajectories_h{H:03d}.npz ({len(traj_dict)} trajectories)",
          flush=True)

    # print metrics for absolute and each K
    actual = sub["remaining_min_actual"].values
    print(f"\n  {'Method':<20}  unique  cap%     MAE    RMSE", flush=True)
    print(f"  {'-'*55}", flush=True)

    for label, col in [("absolute", "pred_abs")] + \
                       [(f"k={k}", f"pred_k{k}") for k in K_VALUES]:
        p     = sub[col].values
        cap   = H * MIN_PER_FRAME
        cap_p = 100 * (p == cap).mean()
        mae_v = mean_absolute_error(actual, p)
        rms_v = rmse(actual, p)
        print(f"  {label:<20}  {sub[col].nunique():5d}  {cap_p:5.1f}%"
              f"  {mae_v:7.1f}  {rms_v:7.1f} min", flush=True)

    # per-group breakdown for K_PLOT
    k_col = f"pred_k{K_PLOT}"
    print(f"\n  Group breakdown  (k={K_PLOT}):", flush=True)
    for g in ["early", "medium", "late"]:
        sg = sub[sub["group"] == g]
        if len(sg) < 3:
            continue
        mae_g = mean_absolute_error(sg["remaining_min_actual"], sg[k_col])
        rms_g = rmse(sg["remaining_min_actual"], sg[k_col])
        print(f"    {g:8s}  n={len(sg):4,}  MAE={mae_g:.1f}  RMSE={rms_g:.1f} min",
              flush=True)
        all_eval.append({"horizon_frames": H, "group": g,
                         "n_windows": len(sg), "k": K_PLOT,
                         "MAE_min": round(mae_g, 1),
                         "RMSE_min": round(rms_g, 1)})

    # scatter plots: one panel per K + absolute, side by side
    n_panels = 1 + len(K_VALUES)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for ax, (label, col) in zip(axes,
            [("absolute threshold", "pred_abs")] +
            [(f"relative  k={k}", f"pred_k{k}") for k in K_VALUES]):
        for g, col_c in GROUP_COLORS.items():
            sg = sub[sub["group"] == g]
            if len(sg):
                ax.scatter(sg["remaining_min_actual"] / 60,
                           sg[col] / 60,
                           color=col_c, alpha=0.35, s=10, label=g)
        lim = max(sub["remaining_min_actual"].max(), sub[col].max()) / 60 * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        cap   = H * MIN_PER_FRAME
        mae_v = mean_absolute_error(actual, sub[col].values)
        cap_p = 100 * (sub[col].values == cap).mean()
        ax.set_title(f"{label}\ncap={cap_p:.0f}%   MAE={mae_v:.0f} min",
                     fontsize=8)
        ax.set_xlabel("Actual remaining time (h)")
        if ax is axes[0]:
            ax.set_ylabel("Predicted remaining time (h)")
        ax.legend(fontsize=7, markerscale=1.5)

    fig.suptitle(f"H={H} frames ({H_min/60:.0f} h)  |  n={len(sub):,} windows",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"forecast_scatter_relthresh_h{H:03d}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved forecast_scatter_relthresh_h{H:03d}.png", flush=True)

# ── Save eval summary ─────────────────────────────────────────────────────────
pd.DataFrame(all_eval).to_csv(OUT_DIR / "eval_summary_relthresh.csv", index=False)
print("\nSaved eval_summary_relthresh.csv")
print("All done.", flush=True)

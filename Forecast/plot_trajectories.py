"""
plot_trajectories.py  — plot predicted vs actual mCherry trajectories

For each horizon, pick a few example test cells (one per group) and show:
  - grey: actual mCherry in context window (16 frames)
  - colored solid: predicted mCherry over the horizon
  - colored dashed: actual mCherry over the horizon
  - red horizontal line: RED_THRESHOLD
  - vertical dashed: actual red-onset frame
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT   = Path("/home/labs/ginossar/talfis/LiveImaging/Forecast")
FIGS  = OUT  # save alongside other forecast figures

RED_THRESHOLD = 2.302
CONTEXT_LEN   = 16
MIN_PER_FRAME = 15

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

# ── Load data ──────────────────────────────────────────────────────────────────
meta  = pd.read_csv(OUT / "cell_metadata.csv")
split = pd.read_csv(OUT / "train_test_split.csv")
meta  = meta.merge(split[["Track.ID","split"]], on="Track.ID").set_index("Track.ID")

X_tensor = np.load(OUT / "tensor_X.npz")["X"]          # (n, max_frames, 9)
Y_tensor = np.load(OUT / "tensor_y_mcherry.npz")["Y"]  # (n, max_frames)
cell_idx = {tid: i for i, tid in enumerate(meta.index)}

ts_raw = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/timeseries_data.csv",
    usecols=["Track.ID","T_min"], low_memory=False).sort_values(["Track.ID","T_min"])
md_raw = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/model_df.csv",
    usecols=["Track.ID","abs_gfp_onset_min"])
ts_raw = ts_raw.merge(md_raw, on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts_raw.groupby("Track.ID")}

test_cells = set(meta[meta["split"]=="test"].index)

# ── For each horizon pick example cells ───────────────────────────────────────
for H in [10, 50, 100]:
    traj_data = np.load(OUT / f"trajectories_h{H:03d}.npz")
    preds_df  = pd.read_csv(OUT / f"predictions_h{H:03d}.csv")
    H_min     = H * MIN_PER_FRAME

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.suptitle(f"Predicted vs actual mCherry  |  Horizon H={H} frames "
                 f"({H_min//60}h {H_min%60}m ahead)", fontsize=12, fontweight="bold")

    for row_idx, group in enumerate(["early", "medium", "late"]):
        color = GROUP_COLORS[group]

        # pick 3 test cells from this group that have windows in traj_data
        grp_cells = [tid for tid in test_cells
                     if tid in meta.index and meta.loc[tid,"group"] == group
                     and any(k.startswith(f"{tid}_w") for k in traj_data.keys())]

        # pick cells spread across early/middle/late windows for variety
        chosen = grp_cells[:min(3, len(grp_cells))]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#f8f8f8")

            if col_idx >= len(chosen):
                ax.axis("off")
                continue

            tid = chosen[col_idx]
            ci  = cell_idx[tid]
            n_real = meta.loc[tid, "n_real_frames"]
            delay  = meta.loc[tid, "delay_green_to_red"]
            gfp_min = ts_raw[ts_raw["Track.ID"]==tid]["abs_gfp_onset_min"].iloc[0]
            red_min = gfp_min + delay
            t_arr   = cell_tmin[tid]

            # pick a qualifying window (one that's not too early, not too late)
            keys_for_cell = sorted(
                [k for k in traj_data.keys() if k.startswith(f"{tid}_w")],
                key=lambda k: int(k.split("_w")[1]))

            if not keys_for_cell:
                ax.axis("off"); continue

            # pick middle window
            key = keys_for_cell[len(keys_for_cell)//2]
            w_start = int(key.split("_w")[1])
            w_end   = w_start + CONTEXT_LEN

            pred_traj = traj_data[key]   # (H,) predicted mCherry
            act_ctx   = Y_tensor[ci, w_start:w_end]     # (16,) actual context
            act_fut   = Y_tensor[ci, w_end:w_end+H]     # (H,) actual future

            # time axis in hours relative to context start
            t_ctx_start = t_arr[w_start] if w_start < len(t_arr) else 0
            ctx_times  = np.arange(CONTEXT_LEN) * MIN_PER_FRAME / 60
            fut_times  = (CONTEXT_LEN + np.arange(len(act_fut))) * MIN_PER_FRAME / 60
            pred_times = (CONTEXT_LEN + np.arange(len(pred_traj))) * MIN_PER_FRAME / 60

            # actual red-onset frame relative to context start
            red_frame_abs = int(np.argmin(np.abs(t_arr - red_min))) if len(t_arr) else None
            red_rel_h = (red_frame_abs - w_start) * MIN_PER_FRAME / 60 if red_frame_abs else None

            # plot
            ax.plot(ctx_times, act_ctx, color="grey", lw=1.5, label="actual (context)")
            ax.plot(pred_times[:len(act_fut)], act_fut, color=color,
                    lw=1.5, linestyle="--", alpha=0.7, label="actual (future)")
            ax.plot(pred_times[:len(pred_traj)], pred_traj, color=color,
                    lw=2, label="predicted")

            # threshold line
            ax.axhline(RED_THRESHOLD, color="red", lw=0.8, linestyle=":", alpha=0.8,
                       label=f"threshold={RED_THRESHOLD:.2f}")

            # vertical line at actual red onset
            if red_rel_h is not None and 0 < red_rel_h < (CONTEXT_LEN + H) * MIN_PER_FRAME / 60:
                ax.axvline(red_rel_h, color="red", lw=1, linestyle="--", alpha=0.6)

            # context / prediction boundary
            ax.axvline(CONTEXT_LEN * MIN_PER_FRAME / 60, color="black",
                       lw=0.8, linestyle=":", alpha=0.5)

            remaining = (red_min - t_arr[w_end-1]) if w_end-1 < len(t_arr) else np.nan
            ax.set_title(f"{tid}  ({group})\nremaining={remaining/60:.1f}h",
                         fontsize=7.5)
            ax.set_xlabel("Time from context start (h)", fontsize=7)
            ax.set_ylabel("mCherry intensity", fontsize=7)
            ax.tick_params(labelsize=6.5)
            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIGS / f"trajectory_plot_h{H:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

print("Done.")

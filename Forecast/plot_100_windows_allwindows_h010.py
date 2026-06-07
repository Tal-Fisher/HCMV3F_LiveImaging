"""
plot_100_windows_allwindows_h010.py
Regenerate the 100-window PDF for the all-windows H=10 forecast using saved outputs.
Context and future lines are connected (no visual gap).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUT           = Path("/home/labs/ginossar/talfis/LiveImaging/Forecast")
BASE          = Path("/home/labs/ginossar/talfis/LiveImaging")
MIN_PER_FRAME = 15
CONTEXT_LEN   = 16
H             = 10
N_TOTAL       = 100
PER_PAGE      = 20
SEED          = 7

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
Y_tensor = np.load(OUT / "tensor_y_mcherry.npz")["Y"]
meta     = pd.read_csv(OUT / "cell_metadata.csv").set_index("Track.ID")
results  = pd.read_csv(OUT / "predictions_allwindows_h010.csv")
traj_npz = np.load(OUT / "trajectories_allwindows_h010.npz")
traj_dict = {k: traj_npz[k] for k in traj_npz.files}

cell_idx = {tid: i for i, tid in enumerate(meta.index)}

ts_raw = pd.read_csv(
    BASE / "cache/python_export/timeseries_data.csv",
    usecols=["Track.ID", "T_min"], low_memory=False
).sort_values(["Track.ID", "T_min"])
md_raw = pd.read_csv(
    BASE / "cache/python_export/model_df.csv",
    usecols=["Track.ID", "abs_gfp_onset_min"]
)
ts_raw = ts_raw.merge(md_raw, on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts_raw.groupby("Track.ID")}

# ── Sample 100 random windows ─────────────────────────────────────────────────
rng    = np.random.default_rng(SEED)
keys   = list(traj_dict.keys())
chosen = rng.choice(keys, size=min(N_TOTAL, len(keys)), replace=False)
print(f"Sampled {len(chosen)} windows from {len(keys)} available", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_cols, n_rows = 5, 4
n_pages = int(np.ceil(len(chosen) / PER_PAGE))

pdf_path = OUT / "windows_allwindows_h010_100examples.pdf"
with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        keys_page = chosen[page * PER_PAGE: (page + 1) * PER_PAGE]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11))
        fig.suptitle(
            f"H=10 all-windows — raw mCherry trajectory  |  page {page+1}/{n_pages}\n"
            "grey=context  solid=predicted future  dashed=actual future  red dashed=onset",
            fontsize=9, fontweight="bold")

        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx >= len(keys_page):
                ax.axis("off")
                continue

            key = keys_page[ax_idx]
            # parse Track.ID and w_start from item_id
            tid     = key[:key.rfind("_w")]
            w_start = int(key[key.rfind("_w") + 2:])
            w_end   = w_start + CONTEXT_LEN

            ci = cell_idx.get(tid)
            if ci is None:
                ax.axis("off")
                continue

            group  = meta.loc[tid, "group"] if tid in meta.index else "unknown"
            color  = GROUP_COLORS.get(group, "steelblue")

            act_ctx   = Y_tensor[ci, w_start:w_end]
            act_fut   = Y_tensor[ci, w_end:w_end + H]
            pred_traj = traj_dict[key]

            ctx_t  = np.arange(CONTEXT_LEN) * MIN_PER_FRAME / 60
            fut_t  = (CONTEXT_LEN + np.arange(H)) * MIN_PER_FRAME / 60

            # connect last context point so there is no visual gap
            conn_t    = np.concatenate([[ctx_t[-1]], fut_t])
            conn_act  = np.concatenate([[act_ctx[-1]], act_fut])
            conn_pred = np.concatenate([[act_ctx[-1]], pred_traj])

            ax.plot(ctx_t,   act_ctx,   color="grey",  lw=1.2, alpha=0.8)
            ax.plot(conn_t,  conn_act,  color=color,   lw=1.2, linestyle="--", alpha=0.75)
            ax.plot(conn_t,  conn_pred, color=color,   lw=1.5)
            ax.axvline(CONTEXT_LEN * MIN_PER_FRAME / 60,
                       color="black", lw=0.6, linestyle=":", alpha=0.4)

            # red onset marker
            t_arr   = cell_tmin.get(tid, np.array([]))
            delay   = meta.loc[tid, "delay_green_to_red"] if tid in meta.index else np.nan
            gfp_min = ts_raw[ts_raw["Track.ID"] == tid]["abs_gfp_onset_min"].iloc[0] \
                      if tid in ts_raw["Track.ID"].values else 0
            total_h = (CONTEXT_LEN + H) * MIN_PER_FRAME / 60
            if np.isfinite(delay) and w_start < len(t_arr):
                red_min     = gfp_min + delay
                t_win_start = t_arr[w_start]
                red_rel_h   = (red_min - t_win_start) / 60
                if 0 <= red_rel_h <= total_h:
                    ax.axvline(red_rel_h, color="red", lw=0.9, linestyle="--", alpha=0.6)

            # per-window MAE from results
            row = results[(results["Track.ID"] == tid) & (results["w_start"] == w_start)]
            mae_str = f"MAE={row['MAE'].values[0]:.3f}" if len(row) else ""
            rem_min = row["remaining_min"].values[0] if len(row) else np.nan
            rem_str = f"{rem_min/60:.1f}h" if np.isfinite(rem_min) and rem_min >= 0 \
                      else (f"post {-rem_min/60:.1f}h" if np.isfinite(rem_min) else "")

            ax.set_title(f"{tid} {group}\nrem={rem_str}  {mae_str}", fontsize=6, pad=2)
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("h", fontsize=5.5)
            if ax_idx % n_cols == 0:
                ax.set_ylabel("mCherry", fontsize=5.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=130)
        plt.close()
        print(f"  Page {page+1}/{n_pages} done", flush=True)

print(f"Saved {pdf_path}", flush=True)

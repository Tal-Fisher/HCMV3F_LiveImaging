"""
plot_100_windows_h010.py  — PDF of 100 random H=10 window examples
5 pages × 20 panels (4 rows × 5 cols) each
Each panel: grey=context, solid=predicted, dashed=actual future, red line=threshold
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUT           = Path("/home/labs/ginossar/talfis/LiveImaging/Forecast")
RED_THRESHOLD = 2.302
CONTEXT_LEN   = 16
MIN_PER_FRAME = 15
H             = 10
N_TOTAL       = 100
PER_PAGE      = 20    # 4 rows × 5 cols
SEED          = 7

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

# ── Load ──────────────────────────────────────────────────────────────────────
meta   = pd.read_csv(OUT / "cell_metadata.csv").set_index("Track.ID")
split  = pd.read_csv(OUT / "train_test_split.csv").set_index("Track.ID")
meta   = meta.join(split["split"])
preds  = pd.read_csv(OUT / "predictions_h010.csv")
trajs  = np.load(OUT / "trajectories_h010.npz")

Y_tensor  = np.load(OUT / "tensor_y_mcherry.npz")["Y"]
cell_idx  = {tid: i for i, tid in enumerate(meta.index)}

ts_raw = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/timeseries_data.csv",
    usecols=["Track.ID","T_min"], low_memory=False).sort_values(["Track.ID","T_min"])
md_raw = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/model_df.csv",
    usecols=["Track.ID","abs_gfp_onset_min"])
ts_raw = ts_raw.merge(md_raw, on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts_raw.groupby("Track.ID")}

# ── Sample 100 random windows from test cells ─────────────────────────────────
test_cells = set(meta[meta["split"]=="test"].index)
all_keys   = [k for k in trajs.keys()
              if k.split("_w")[0] in test_cells]

rng = np.random.default_rng(SEED)
chosen_keys = rng.choice(all_keys, size=min(N_TOTAL, len(all_keys)), replace=False)
print(f"Total available keys: {len(all_keys)}  |  Sampling {len(chosen_keys)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
pdf_path = OUT / "windows_h010_100examples.pdf"
n_pages  = int(np.ceil(len(chosen_keys) / PER_PAGE))
n_cols, n_rows = 5, 4

with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        keys_page = chosen_keys[page * PER_PAGE : (page + 1) * PER_PAGE]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11))
        fig.suptitle(
            f"H=10 (2.5 h ahead) — random windows  |  page {page+1}/{n_pages}  "
            f"(grey=context, solid=predicted, dashed=actual, ···=threshold)",
            fontsize=9, fontweight="bold")

        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx >= len(keys_page):
                ax.axis("off")
                continue

            key    = keys_page[ax_idx]
            tid    = "_".join(key.split("_")[:-1])   # handle Track.IDs with underscores
            # safer: split on last '_w'
            tid    = key[:key.rfind("_w")]
            w_start = int(key.split("_w")[-1])
            w_end   = w_start + CONTEXT_LEN

            ci      = cell_idx.get(tid)
            if ci is None:
                ax.axis("off"); continue

            group  = meta.loc[tid, "group"]
            delay  = meta.loc[tid, "delay_green_to_red"]
            color  = GROUP_COLORS.get(group, "steelblue")

            act_ctx  = Y_tensor[ci, w_start:w_end]
            act_fut  = Y_tensor[ci, w_end:w_end + H]
            pred_traj = trajs[key]

            ctx_t  = np.arange(CONTEXT_LEN) * MIN_PER_FRAME / 60
            fut_t  = (CONTEXT_LEN + np.arange(len(act_fut)))  * MIN_PER_FRAME / 60
            pred_t = (CONTEXT_LEN + np.arange(len(pred_traj))) * MIN_PER_FRAME / 60

            # actual red onset relative to window
            t_arr = cell_tmin.get(tid, np.array([]))
            gfp_min = ts_raw[ts_raw["Track.ID"]==tid]["abs_gfp_onset_min"].iloc[0] \
                      if tid in ts_raw["Track.ID"].values else 0
            red_min   = gfp_min + delay
            red_frame = int(np.argmin(np.abs(t_arr - red_min))) if len(t_arr) else None
            t_ctx_start = t_arr[w_start] if w_start < len(t_arr) else 0
            remaining = (red_min - t_arr[w_end-1]) / 60 if w_end-1 < len(t_arr) else np.nan

            ax.plot(ctx_t,  act_ctx,   color="grey",  lw=1.2, alpha=0.8)
            ax.plot(fut_t,  act_fut,   color=color,   lw=1.2, linestyle="--", alpha=0.75)
            ax.plot(pred_t, pred_traj, color=color,   lw=1.5)
            ax.axhline(RED_THRESHOLD, color="red", lw=0.6, linestyle=":", alpha=0.7)
            ax.axvline(CONTEXT_LEN * MIN_PER_FRAME / 60, color="black",
                       lw=0.6, linestyle=":", alpha=0.4)

            # mark actual red onset if within plot
            if red_frame is not None:
                red_rel = (red_frame - w_start) * MIN_PER_FRAME / 60
                x_end = (CONTEXT_LEN + H) * MIN_PER_FRAME / 60
                if 0 < red_rel < x_end:
                    ax.axvline(red_rel, color="red", lw=0.8, linestyle="--", alpha=0.55)

            ax.set_title(f"{tid}  {group}\nrem={remaining:.1f}h",
                         fontsize=6, pad=2)
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("h", fontsize=5.5)
            if ax_idx % n_cols == 0:
                ax.set_ylabel("mCherry", fontsize=5.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=130)
        plt.close()
        print(f"  Page {page+1}/{n_pages} done")

print(f"Saved {pdf_path}")

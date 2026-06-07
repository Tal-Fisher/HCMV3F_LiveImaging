"""
plot_100_windows_h001.py — PDF of 100 random H=1 windows
For H=1 each panel shows the 16-frame context + one predicted vs actual next point.
Grey line = context. Colored filled circle = predicted. Colored open circle = actual.
A connecting line shows the prediction error magnitude.
Red vertical dashed = actual red onset (when within range).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUT           = Path("/home/labs/ginossar/talfis/LiveImaging/Forecast")
MIN_PER_FRAME = 15
CONTEXT_LEN   = 16
N_TOTAL       = 100
PER_PAGE      = 20
SEED          = 7

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

# ── Load ──────────────────────────────────────────────────────────────────────
preds   = pd.read_csv(OUT / "predictions_h001.csv")
Y_tensor = np.load(OUT / "tensor_y_mcherry.npz")["Y"]
meta    = pd.read_csv(OUT / "cell_metadata.csv").set_index("Track.ID")
split   = pd.read_csv(OUT / "train_test_split.csv").set_index("Track.ID")
meta    = meta.join(split["split"])
cell_idx = {tid: i for i, tid in enumerate(meta.index)}

ts_raw = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/timeseries_data.csv",
    usecols=["Track.ID","T_min"], low_memory=False).sort_values(["Track.ID","T_min"])
md_raw = pd.read_csv(
    "/home/labs/ginossar/talfis/LiveImaging/cache/python_export/model_df.csv",
    usecols=["Track.ID","abs_gfp_onset_min"])
ts_raw = ts_raw.merge(md_raw, on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts_raw.groupby("Track.ID")}

# test cells only
test_cells = set(meta[meta["split"]=="test"].index)
preds = preds[preds["Track.ID"].isin(test_cells)].reset_index(drop=True)

# ── Sample 100 random windows ─────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
chosen  = preds.iloc[rng.choice(len(preds), size=min(N_TOTAL, len(preds)), replace=False)]
chosen  = chosen.reset_index(drop=True)
print(f"Sampled {len(chosen)} windows from {len(preds)} available")

# ── Plot ──────────────────────────────────────────────────────────────────────
n_cols, n_rows = 5, 4
n_pages = int(np.ceil(len(chosen) / PER_PAGE))

pdf_path = OUT / "windows_h001_100examples.pdf"
with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        rows_page = chosen.iloc[page * PER_PAGE : (page+1) * PER_PAGE]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11))
        fig.suptitle(
            f"H=1 (15 min ahead) — 100 random windows  |  page {page+1}/{n_pages}\n"
            "grey=context  ●=predicted next  ○=actual next  red dashed=onset",
            fontsize=9, fontweight="bold")

        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx >= len(rows_page):
                ax.axis("off"); continue

            row     = rows_page.iloc[ax_idx]
            tid     = row["Track.ID"]
            w_start = int(row["w_start"])
            w_end   = w_start + CONTEXT_LEN
            group   = row["group"]
            color   = GROUP_COLORS.get(group, "steelblue")

            ci = cell_idx.get(tid)
            if ci is None:
                ax.axis("off"); continue

            act_ctx   = Y_tensor[ci, w_start:w_end]
            pred_next = float(row["pred_mcherry_next"])
            act_next  = float(row["actual_mcherry_next"])
            ctx_last  = float(row["actual_mcherry_ctx_last"])

            # time axis: 0 = first context frame, CONTEXT_LEN = next (predicted) frame
            ctx_t  = np.arange(CONTEXT_LEN)       * MIN_PER_FRAME / 60
            next_t = CONTEXT_LEN                   * MIN_PER_FRAME / 60

            # context line
            ax.plot(ctx_t, act_ctx, color="grey", lw=1.2, alpha=0.85)

            # connect last context point to predicted and actual
            ax.plot([ctx_t[-1], next_t], [ctx_last, pred_next],
                    color=color, lw=1.0, alpha=0.6, linestyle="-")
            ax.plot([ctx_t[-1], next_t], [ctx_last, act_next],
                    color=color, lw=1.0, alpha=0.4, linestyle="--")

            # predicted = filled circle, actual = open circle
            ax.scatter([next_t], [pred_next], color=color, s=30, zorder=5, label="pred")
            ax.scatter([next_t], [act_next],  facecolors="none", edgecolors=color,
                       s=30, zorder=5, linewidths=1.2, label="actual")

            # context / prediction boundary
            ax.axvline(ctx_t[-1] + MIN_PER_FRAME/60/2,
                       color="black", lw=0.5, linestyle=":", alpha=0.4)

            # red onset marker
            t_arr = cell_tmin.get(tid, np.array([]))
            delay = meta.loc[tid, "delay_green_to_red"]
            gfp_min_val = ts_raw[ts_raw["Track.ID"]==tid]["abs_gfp_onset_min"].iloc[0] \
                          if tid in ts_raw["Track.ID"].values else 0
            red_min = gfp_min_val + delay
            if w_start < len(t_arr):
                t_win0   = t_arr[w_start]
                red_rel_h = (red_min - t_win0) / 60
                plot_end  = (CONTEXT_LEN + 1) * MIN_PER_FRAME / 60
                if 0 <= red_rel_h <= plot_end:
                    ax.axvline(red_rel_h, color="red", lw=0.8, linestyle="--", alpha=0.55)

            err = abs(pred_next - act_next)
            rem_h = row["remaining_min_actual"] / 60
            ax.set_title(f"{tid} {group}\nrem={rem_h:.1f}h  err={err:.3f}",
                         fontsize=6, pad=2)
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("h", fontsize=5.5)
            if ax_idx % n_cols == 0:
                ax.set_ylabel("mCherry", fontsize=5.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=130)
        plt.close()
        print(f"  Page {page+1}/{n_pages} done")

print(f"Saved {pdf_path}")

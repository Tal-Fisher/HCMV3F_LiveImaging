"""
07_run_forecaster_allwindows_h010_quantiles.py
Re-runs the H=10 all-windows forecast saving all quantile predictions.
Produces fan chart PDF (inner band=25-75%, outer band=10-90%, median line).
Quantile-based onset probability: fraction of quantiles crossing RED_THRESHOLD.
"""

import pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabicl import TabICLForecaster

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT_DIR = BASE / "Forecast"
TS_CSV  = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"

CONTEXT_LEN   = 16
H             = 10
STRIDE        = 3
MIN_PER_FRAME = 15
RANDOM_STATE  = 42
N_PLOT        = 100
RED_THRESHOLD = 2.302
QUANTILES     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}
FEAT_NAMES   = ["GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
                "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc"]

# ── Load tensors & metadata ───────────────────────────────────────────────────
print("Loading tensors ...", flush=True)
X_tensor = np.load(OUT_DIR / "tensor_X.npz")["X"]
Y_tensor = np.load(OUT_DIR / "tensor_y_mcherry.npz")["Y"]

_meta = pd.read_csv(OUT_DIR / "cell_metadata.csv")
_md   = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
_meta = _meta.merge(_md, on="Track.ID", how="left")
meta  = _meta.set_index("Track.ID")

ts = pd.read_csv(TS_CSV, usecols=["Track.ID", "T_min"], low_memory=False)
ts = ts.sort_values(["Track.ID", "T_min"])
md = pd.read_csv(MD_CSV, usecols=["Track.ID", "abs_gfp_onset_min"])
ts = ts.merge(md, on="Track.ID", how="left")
cell_tmin = {tid: grp["T_min"].values for tid, grp in ts.groupby("Track.ID")}

# ── Same 75/25 split ──────────────────────────────────────────────────────────
cell_info = meta.reset_index()[["Track.ID", "group"]].drop_duplicates("Track.ID")
splitter  = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(cell_info, cell_info["group"]))
test_cells = set(cell_info.iloc[test_idx]["Track.ID"])
print(f"  Test cells: {len(test_cells)}", flush=True)

# ── Build windows ─────────────────────────────────────────────────────────────
print("\nBuilding windows ...", flush=True)
windows = []
for i, tid in enumerate(meta.index):
    if tid not in test_cells or tid not in cell_tmin:
        continue
    t_arr    = cell_tmin[tid]
    n_frames = len(t_arr)
    group    = meta.loc[tid, "group"]
    delay    = meta.loc[tid, "delay_green_to_red"]
    gfp_min  = meta.loc[tid, "abs_gfp_onset_min"]
    red_min  = gfp_min + delay

    for w_start in range(0, n_frames - CONTEXT_LEN, STRIDE):
        w_end    = w_start + CONTEXT_LEN
        if n_frames - w_end < H:
            continue
        remaining = red_min - t_arr[w_end - 1]
        windows.append({
            "item_id":       f"{tid}_w{w_start}",
            "Track.ID":      tid,
            "group":         group,
            "w_start":       w_start,
            "remaining_min": remaining,
            "ts_ctx":        pd.date_range("2000-01-01", periods=CONTEXT_LEN,
                                           freq=f"{MIN_PER_FRAME}min"),
            "X_ctx":         X_tensor[i, w_start:w_end, :],
            "Y_ctx":         Y_tensor[i, w_start:w_end],
            "Y_future":      Y_tensor[i, w_end:w_end + H],
        })

print(f"  Total windows: {len(windows):,}", flush=True)

# ── Build context_df ──────────────────────────────────────────────────────────
print("\nBuilding context_df ...", flush=True)
ctx_rows = []
for w in windows:
    for t_idx in range(CONTEXT_LEN):
        row = {"item_id":   w["item_id"],
               "timestamp": w["ts_ctx"][t_idx],
               "target":    float(w["Y_ctx"][t_idx])}
        for fi, fn in enumerate(FEAT_NAMES):
            row[fn] = float(w["X_ctx"][t_idx, fi])
        ctx_rows.append(row)
context_df = pd.DataFrame(ctx_rows)
print(f"  context_df shape: {context_df.shape}", flush=True)

# ── Run forecaster with quantiles ─────────────────────────────────────────────
forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)
print(f"Running predict_df with quantiles {QUANTILES} ...", flush=True)
pred_df = forecaster.predict_df(context_df, prediction_length=H, quantiles=QUANTILES)
print(f"  pred_df columns: {list(pred_df.columns)}", flush=True)

# ── Collect results — save median + all quantiles ─────────────────────────────
records = []
traj_dict = {}   # median trajectory per window

for w in windows:
    iid = w["item_id"]
    window_pred = pred_df.xs(iid, level="item_id") if iid in pred_df.index.get_level_values("item_id") else None
    if window_pred is None or len(window_pred) == 0:
        continue

    pred_median = window_pred[0.5].values     # median
    pred_q10    = window_pred[0.1].values
    pred_q25    = window_pred[0.2].values     # use 0.2 as inner lower (closest to 0.25)
    pred_q75    = window_pred[0.8].values     # use 0.8 as inner upper
    pred_q90    = window_pred[0.9].values
    act_traj    = w["Y_future"]

    traj_dict[iid] = {
        "median": pred_median,
        "q10":    pred_q10,
        "q25":    pred_q25,
        "q75":    pred_q75,
        "q90":    pred_q90,
    }

    mae_w = mean_absolute_error(act_traj, pred_median)
    naive = np.full(H, w["Y_ctx"][-1])
    mae_n = mean_absolute_error(act_traj, naive)

    # quantile-based onset probability: fraction of quantile trajectories that cross threshold
    all_quantile_preds = window_pred[QUANTILES].values   # shape (H, n_quantiles)
    onset_prob = np.mean([any(all_quantile_preds[:, qi] > RED_THRESHOLD)
                          for qi in range(len(QUANTILES))])

    records.append({
        "Track.ID":      w["Track.ID"],
        "group":         w["group"],
        "w_start":       w["w_start"],
        "remaining_min": w["remaining_min"],
        "pred_median":   json.dumps(pred_median.tolist()),
        "pred_q10":      json.dumps(pred_q10.tolist()),
        "pred_q25":      json.dumps(pred_q25.tolist()),
        "pred_q75":      json.dumps(pred_q75.tolist()),
        "pred_q90":      json.dumps(pred_q90.tolist()),
        "actual_traj":   json.dumps(act_traj.tolist()),
        "ctx_last":      float(w["Y_ctx"][-1]),
        "MAE_median":    round(mae_w, 4),
        "MAE_naive":     round(mae_n, 4),
        "onset_prob":    round(onset_prob, 3),
    })

results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "predictions_allwindows_h010_quantiles.csv", index=False)
print(f"Saved predictions_allwindows_h010_quantiles.csv ({len(results_df):,} rows)", flush=True)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n=== Evaluation (median prediction) ===", flush=True)
all_act   = np.concatenate([json.loads(r) for r in results_df["actual_traj"]])
all_pred  = np.concatenate([json.loads(r) for r in results_df["pred_median"]])
all_naive = np.repeat(results_df["ctx_last"].values, H)

mae_all   = mean_absolute_error(all_act, all_pred)
rmse_all  = rmse(all_act, all_pred)
mae_naive = mean_absolute_error(all_act, all_naive)
skill     = 1 - mae_all / mae_naive
print(f"Overall  MAE={mae_all:.4f}  RMSE={rmse_all:.4f}  skill={skill:.3f}", flush=True)

print("\n=== Onset detection (quantile-based probability) ===", flush=True)
results_df["actual_max"]  = results_df["actual_traj"].apply(lambda x: max(json.loads(x)))
results_df["is_crossing"] = results_df["actual_max"] > RED_THRESHOLD

crossing     = results_df[results_df["is_crossing"]]
non_crossing = results_df[~results_df["is_crossing"] & (results_df["remaining_min"] >= 0)]

# threshold onset_prob at 0.5 for binary detection
crossing_detect     = (crossing["onset_prob"]     >= 0.5).mean()
noncrossing_detect  = (non_crossing["onset_prob"] >= 0.5).mean()
print(f"  Sensitivity (onset_prob >= 0.5): {crossing_detect:.3f}  "
      f"({(crossing['onset_prob']>=0.5).sum()}/{len(crossing)})", flush=True)
print(f"  Specificity:                     {1-noncrossing_detect:.3f}  "
      f"({(non_crossing['onset_prob']<0.5).sum()}/{len(non_crossing)})", flush=True)

# ── PDF fan chart: 100 random windows ────────────────────────────────────────
print(f"\nGenerating fan chart PDF ...", flush=True)
rng        = np.random.default_rng(7)
keys       = list(traj_dict.keys())
chosen     = rng.choice(keys, size=min(N_PLOT, len(keys)), replace=False)
win_lookup = {w["item_id"]: w for w in windows}

PER_PAGE   = 20
n_cols, n_rows = 5, 4
n_pages    = int(np.ceil(len(chosen) / PER_PAGE))

pdf_path = OUT_DIR / "windows_allwindows_h010_fanchart.pdf"
with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        keys_page = chosen[page * PER_PAGE: (page + 1) * PER_PAGE]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11))
        fig.suptitle(
            f"H=10 fan chart — page {page+1}/{n_pages}\n"
            "grey=context  solid=median pred  dark band=20-80%  light band=10-90%"
            "  dashed=actual  red dashed=onset",
            fontsize=8, fontweight="bold")

        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx >= len(keys_page):
                ax.axis("off"); continue
            key = keys_page[ax_idx]
            w   = win_lookup.get(key)
            if w is None or key not in traj_dict:
                ax.axis("off"); continue

            tid     = w["Track.ID"]
            color   = GROUP_COLORS.get(w["group"], "steelblue")
            act_ctx = w["Y_ctx"]
            act_fut = w["Y_future"]
            td      = traj_dict[key]

            ctx_t  = np.arange(CONTEXT_LEN) * MIN_PER_FRAME / 60
            fut_t  = (CONTEXT_LEN + np.arange(H)) * MIN_PER_FRAME / 60
            conn_t = np.concatenate([[ctx_t[-1]], fut_t])

            # context
            ax.plot(ctx_t, act_ctx, color="grey", lw=1.2, alpha=0.8)

            # fan bands (connect from last context point)
            q10 = np.concatenate([[act_ctx[-1]], td["q10"]])
            q25 = np.concatenate([[act_ctx[-1]], td["q25"]])
            q75 = np.concatenate([[act_ctx[-1]], td["q75"]])
            q90 = np.concatenate([[act_ctx[-1]], td["q90"]])
            med = np.concatenate([[act_ctx[-1]], td["median"]])
            act = np.concatenate([[act_ctx[-1]], act_fut])

            ax.fill_between(conn_t, q10, q90, color=color, alpha=0.15, label="10-90%")
            ax.fill_between(conn_t, q25, q75, color=color, alpha=0.30, label="20-80%")
            ax.plot(conn_t, med, color=color, lw=1.5, label="median")
            ax.plot(conn_t, act, color=color, lw=1.0, linestyle="--", alpha=0.75, label="actual")

            ax.axvline(CONTEXT_LEN * MIN_PER_FRAME / 60,
                       color="black", lw=0.5, linestyle=":", alpha=0.4)
            ax.axhline(RED_THRESHOLD, color="red", lw=0.5, linestyle=":", alpha=0.4)

            t_arr   = cell_tmin.get(tid, np.array([]))
            delay   = meta.loc[tid, "delay_green_to_red"]
            gfp_min = meta.loc[tid, "abs_gfp_onset_min"]
            total_h = (CONTEXT_LEN + H) * MIN_PER_FRAME / 60
            if np.isfinite(delay) and w["w_start"] < len(t_arr):
                red_rel_h = (gfp_min + delay - t_arr[w["w_start"]]) / 60
                if 0 <= red_rel_h <= total_h:
                    ax.axvline(red_rel_h, color="red", lw=0.9, linestyle="--", alpha=0.6)

            # onset prob from quantiles
            row = results_df[(results_df["Track.ID"] == tid) &
                             (results_df["w_start"] == w["w_start"])]
            onset_p = row["onset_prob"].values[0] if len(row) else float("nan")
            rem_h   = w["remaining_min"] / 60
            rem_str = f"{rem_h:.1f}h" if rem_h >= 0 else f"post {-rem_h:.1f}h"
            ax.set_title(f"{tid} {w['group']}\nrem={rem_str}  P(onset)={onset_p:.2f}",
                         fontsize=6, pad=2)
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("h", fontsize=5.5)
            if ax_idx % n_cols == 0:
                ax.set_ylabel("mCherry", fontsize=5.5)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, dpi=130)
        plt.close()
        print(f"  Page {page+1}/{n_pages} done", flush=True)

print(f"Saved {pdf_path}", flush=True)
print("\nAll done.", flush=True)

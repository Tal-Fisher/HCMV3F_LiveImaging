"""
05_run_forecaster_allwindows_h005.py
TabICLForecaster: predict mCherry H=5 frames (75 min = 1.25h) ahead, all windows.
No normalization. Covariates provided for context frames only.
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
H             = 5
STRIDE        = 3
MIN_PER_FRAME = 15
RANDOM_STATE  = 42
N_PLOT        = 100

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
print(f"  {len(meta)} cells, tensor shape {X_tensor.shape}", flush=True)

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
for g in ["early", "medium", "late"]:
    print(f"    {g:8s}: {sum(1 for w in windows if w['group']==g):,}", flush=True)

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

# ── Run forecaster ────────────────────────────────────────────────────────────
forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)
with open(OUT_DIR / "tabicl_forecaster_allwindows_h005.pkl", "wb") as f:
    pickle.dump(forecaster, f)
print("Saved tabicl_forecaster_allwindows_h005.pkl", flush=True)

print(f"Running predict_df(prediction_length={H}) ...", flush=True)
pred_df  = forecaster.predict_df(context_df, prediction_length=H)
print(f"  Predictions shape: {pred_df.shape}", flush=True)
pred_grp = pred_df.groupby(level=0)["target"].apply(list).to_dict()

# ── Collect results ───────────────────────────────────────────────────────────
traj_dict = {}
records   = []
for w in windows:
    iid = w["item_id"]
    if iid not in pred_grp:
        continue
    pred_traj = np.array(pred_grp[iid])
    act_traj  = w["Y_future"]
    traj_dict[iid] = pred_traj

    mae_w = mean_absolute_error(act_traj, pred_traj)
    naive = np.full(H, w["Y_ctx"][-1])
    mae_n = mean_absolute_error(act_traj, naive)

    records.append({
        "Track.ID":      w["Track.ID"],
        "group":         w["group"],
        "w_start":       w["w_start"],
        "remaining_min": w["remaining_min"],
        "pred_traj":     json.dumps(pred_traj.tolist()),
        "actual_traj":   json.dumps(act_traj.tolist()),
        "ctx_last":      float(w["Y_ctx"][-1]),
        "MAE":           round(mae_w, 4),
        "MAE_naive":     round(mae_n, 4),
    })

np.savez_compressed(OUT_DIR / "trajectories_allwindows_h005.npz", **traj_dict)
print("Saved trajectories_allwindows_h005.npz", flush=True)

results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "predictions_allwindows_h005.csv", index=False)
print(f"Saved predictions_allwindows_h005.csv ({len(results_df):,} rows)", flush=True)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n=== Evaluation ===", flush=True)
all_act   = np.concatenate([json.loads(r) for r in results_df["actual_traj"]])
all_pred  = np.concatenate([json.loads(r) for r in results_df["pred_traj"]])
all_naive = np.repeat(results_df["ctx_last"].values, H)

mae_all   = mean_absolute_error(all_act, all_pred)
rmse_all  = rmse(all_act, all_pred)
mae_naive = mean_absolute_error(all_act, all_naive)
skill     = 1 - mae_all / mae_naive
print(f"Overall  MAE={mae_all:.4f}  RMSE={rmse_all:.4f}  "
      f"naive_MAE={mae_naive:.4f}  skill={skill:.3f}", flush=True)

eval_rows = []
for g in ["early", "medium", "late"]:
    sg = results_df[results_df["group"] == g]
    if len(sg) < 3:
        continue
    act_g   = np.concatenate([json.loads(r) for r in sg["actual_traj"]])
    pred_g  = np.concatenate([json.loads(r) for r in sg["pred_traj"]])
    naive_g = np.repeat(sg["ctx_last"].values, H)
    mae_g   = mean_absolute_error(act_g, pred_g)
    rmse_g  = rmse(act_g, pred_g)
    mae_ng  = mean_absolute_error(act_g, naive_g)
    skill_g = 1 - mae_g / mae_ng
    print(f"  {g:8s}  n={len(sg):4,}  MAE={mae_g:.4f}  RMSE={rmse_g:.4f}  "
          f"naive_MAE={mae_ng:.4f}  skill={skill_g:.3f}", flush=True)
    eval_rows.append({"group": g, "n_windows": len(sg),
                      "MAE": round(mae_g,4), "RMSE": round(rmse_g,4),
                      "MAE_naive": round(mae_ng,4), "skill": round(skill_g,3)})

eval_rows.append({"group": "all", "n_windows": len(results_df),
                  "MAE": round(mae_all,4), "RMSE": round(rmse_all,4),
                  "MAE_naive": round(mae_naive,4), "skill": round(skill,3)})
pd.DataFrame(eval_rows).to_csv(OUT_DIR / "eval_allwindows_h005.csv", index=False)
print("Saved eval_allwindows_h005.csv", flush=True)

# ── PDF: 100 random windows ───────────────────────────────────────────────────
print(f"\nGenerating PDF ...", flush=True)
rng        = np.random.default_rng(7)
keys       = list(traj_dict.keys())
chosen     = rng.choice(keys, size=min(N_PLOT, len(keys)), replace=False)
win_lookup = {w["item_id"]: w for w in windows}

PER_PAGE   = 20
n_cols, n_rows = 5, 4
n_pages    = int(np.ceil(len(chosen) / PER_PAGE))

pdf_path = OUT_DIR / "windows_allwindows_h005_100examples.pdf"
with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        keys_page = chosen[page * PER_PAGE: (page + 1) * PER_PAGE]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11))
        fig.suptitle(
            f"H=5 all-windows — raw mCherry trajectory  |  page {page+1}/{n_pages}\n"
            "grey=context  solid=predicted  dashed=actual  red dashed=onset",
            fontsize=9, fontweight="bold")

        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx >= len(keys_page):
                ax.axis("off"); continue
            key = keys_page[ax_idx]
            w   = win_lookup.get(key)
            if w is None:
                ax.axis("off"); continue

            tid     = w["Track.ID"]
            color   = GROUP_COLORS.get(w["group"], "steelblue")
            act_ctx = w["Y_ctx"]
            act_fut = w["Y_future"]
            pred_t  = traj_dict[key]

            ctx_t  = np.arange(CONTEXT_LEN) * MIN_PER_FRAME / 60
            fut_t  = (CONTEXT_LEN + np.arange(H)) * MIN_PER_FRAME / 60
            conn_t    = np.concatenate([[ctx_t[-1]], fut_t])
            conn_act  = np.concatenate([[act_ctx[-1]], act_fut])
            conn_pred = np.concatenate([[act_ctx[-1]], pred_t])

            ax.plot(ctx_t,   act_ctx,   color="grey",  lw=1.2, alpha=0.8)
            ax.plot(conn_t,  conn_act,  color=color,   lw=1.2, linestyle="--", alpha=0.75)
            ax.plot(conn_t,  conn_pred, color=color,   lw=1.5)
            ax.axvline(CONTEXT_LEN * MIN_PER_FRAME / 60,
                       color="black", lw=0.6, linestyle=":", alpha=0.4)

            t_arr   = cell_tmin.get(tid, np.array([]))
            delay   = meta.loc[tid, "delay_green_to_red"]
            gfp_min = meta.loc[tid, "abs_gfp_onset_min"]
            total_h = (CONTEXT_LEN + H) * MIN_PER_FRAME / 60
            if np.isfinite(delay) and w["w_start"] < len(t_arr):
                red_rel_h = (gfp_min + delay - t_arr[w["w_start"]]) / 60
                if 0 <= red_rel_h <= total_h:
                    ax.axvline(red_rel_h, color="red", lw=0.9, linestyle="--", alpha=0.6)

            rem_h   = w["remaining_min"] / 60
            rem_str = f"{rem_h:.1f}h" if rem_h >= 0 else f"post {-rem_h:.1f}h"
            ax.set_title(f"{tid} {w['group']}\nrem={rem_str}  MAE={mean_absolute_error(act_fut,pred_t):.3f}",
                         fontsize=6, pad=2)
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("h", fontsize=5.5)
            if ax_idx % n_cols == 0:
                ax.set_ylabel("mCherry", fontsize=5.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=130)
        plt.close()
        print(f"  Page {page+1}/{n_pages} done", flush=True)

print(f"Saved {pdf_path}", flush=True)
print("\nAll done.", flush=True)

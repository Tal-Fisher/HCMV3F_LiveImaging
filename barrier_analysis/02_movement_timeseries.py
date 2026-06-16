"""
02_movement_timeseries.py

Computes per-frame rolling movement metrics from raw X,Y spot positions,
then plots trajectories aligned to IE (GFP) onset.

Rolling window W_SHORT = 8 frames (~2h):
  confinement_ratio   net_displacement / total_path_length
  speed_cv            std(speed) / mean(speed)
  persistence         mean(cos(turning_angle))
  mean_abs_turn_deg   mean(|turning_angle|) in degrees

Rolling window W_LONG = 16 frames (~4h):
  alpha               log-log slope of MSD vs lag (anomalous diffusion exponent)
                      alpha<1 subdiffusive, alpha~1 normal diffusion, alpha>1 directed

Outputs:
  cache/python_export/movement_features_timeseries.csv
  barrier_analysis/movement_timeseries.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_CSV  = os.path.join(ROOT, "cache", "python_export", "timeseries_data.csv")
MD_CSV  = os.path.join(ROOT, "cache", "python_export", "model_df.csv")
OUT_CSV = os.path.join(ROOT, "cache", "python_export", "movement_features_timeseries.csv")
OUT_PNG = os.path.join(ROOT, "barrier_analysis", "movement_timeseries.png")

SPOTS_FILES = {
    "A2": os.path.join(ROOT, "CompleteImage", "A2_Merged_spots.csv"),
    "A3": os.path.join(ROOT, "CompleteImage", "A3_Merged_spots.csv"),
}

# ── parameters ─────────────────────────────────────────────────────────────────
W_SHORT  = 8    # rolling window for confinement, CV, persistence (~2h)
W_LONG   = 16   # rolling window for alpha (~4h)
MAX_LAG  = 4    # max MSD lag for alpha fit
T_MIN_H  = -5.0
T_MAX_H  = 40.0
BIN_W_H  = 0.25
MIN_N    = 5
ROLL_WIN = 3
SG_WIN   = 11
SG_POLY  = 3

PROD_COLOR  = "#2980b9"
NP_COLOR    = "#888888"
ONSET_COLOR = "#27ae60"
RED_COLOR   = "#e74c3c"

FEATURES = [
    ("speed_px_per_frame",  "Speed (px/frame)"),
    ("confinement_ratio",   "Confinement Ratio"),
    ("speed_cv",            "Speed CV (std/mean)"),
    ("persistence",         "Directional Persistence"),
    ("mean_abs_turn_deg",   "Mean |Turning Angle| (°)"),
    ("alpha",               "Diffusion Exponent α"),
]


# ── per-track rolling metric computation ──────────────────────────────────────
def compute_track_metrics(grp):
    grp = grp.sort_values("Frame").reset_index(drop=True)
    x = grp["X"].values.astype(float)
    y = grp["Y"].values.astype(float)
    n = len(grp)

    # per-frame displacement
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dx[0] = 0.0
    dy[0] = 0.0
    speed = np.sqrt(dx**2 + dy**2)

    # turning angle (signed, radians) – undefined at frame 0 and 1
    move_angle = np.arctan2(dy, dx)
    turn = np.diff(move_angle, prepend=move_angle[0])
    turn[0] = 0.0
    # wrap to [-π, π]
    turn = (turn + np.pi) % (2 * np.pi) - np.pi

    # allocate output arrays
    conf      = np.full(n, np.nan)
    spd_cv    = np.full(n, np.nan)
    persist   = np.full(n, np.nan)
    abs_turn  = np.full(n, np.nan)
    alpha_arr = np.full(n, np.nan)

    # ── short rolling window ──
    for i in range(W_SHORT - 1, n):
        seg_x = x[i - W_SHORT + 1 : i + 1]
        seg_y = y[i - W_SHORT + 1 : i + 1]
        seg_s = speed[i - W_SHORT + 2 : i + 1]   # W_SHORT-1 steps
        seg_t = turn[i - W_SHORT + 1 : i + 1]

        net        = np.sqrt((seg_x[-1] - seg_x[0])**2 + (seg_y[-1] - seg_y[0])**2)
        total_path = seg_s.sum()
        conf[i]    = net / total_path if total_path > 1e-9 else 0.0

        if seg_s.mean() > 1e-9:
            spd_cv[i] = seg_s.std() / seg_s.mean()

        persist[i]  = np.cos(seg_t).mean()
        abs_turn[i] = np.degrees(np.abs(seg_t).mean())

    # ── long rolling window: anomalous diffusion exponent ──
    log_lags = np.log(np.arange(1, MAX_LAG + 1, dtype=float))
    for i in range(W_LONG - 1, n):
        seg_x = x[i - W_LONG + 1 : i + 1]
        seg_y = y[i - W_LONG + 1 : i + 1]
        msds  = []
        for lag in range(1, MAX_LAG + 1):
            ddx = seg_x[lag:] - seg_x[:-lag]
            ddy = seg_y[lag:] - seg_y[:-lag]
            msds.append(np.mean(ddx**2 + ddy**2))
        if min(msds) > 1e-12:
            alpha_arr[i] = np.polyfit(log_lags, np.log(msds), 1)[0]

    out = grp[["Track.ID", "Frame"]].copy()
    out["speed_px_per_frame"] = speed
    out["confinement_ratio"]  = conf
    out["speed_cv"]           = spd_cv
    out["persistence"]        = persist
    out["mean_abs_turn_deg"]  = abs_turn
    out["alpha"]              = alpha_arr
    return out


# ── load spots & compute ───────────────────────────────────────────────────────
print("Loading spots …")
ts   = pd.read_csv(TS_CSV, low_memory=False)
md   = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red"]]
valid_ids = set(ts["Track.ID"].unique())

dfs = []
for dataset, path in SPOTS_FILES.items():
    print(f"  {os.path.basename(path)} …")
    raw = pd.read_csv(path, usecols=["Track ID", "Frame", "X", "Y"],
                      low_memory=False)
    raw = raw.rename(columns={"Track ID": "track_num"})
    raw["track_num"] = pd.to_numeric(raw["track_num"], errors="coerce")
    raw = raw.dropna(subset=["track_num"])
    raw["Track.ID"] = dataset + "_" + raw["track_num"].astype(int).astype(str)
    raw = raw[raw["Track.ID"].isin(valid_ids)]
    raw["X"] = pd.to_numeric(raw["X"], errors="coerce")
    raw["Y"] = pd.to_numeric(raw["Y"], errors="coerce")
    raw = raw.dropna(subset=["X", "Y"])
    dfs.append(raw)
    print(f"    {raw['Track.ID'].nunique()} tracks, {len(raw):,} frames")

spots = pd.concat(dfs, ignore_index=True)
spots = spots.sort_values(["Track.ID", "Frame"]).reset_index(drop=True)

print("Computing rolling metrics …")
results = (spots.groupby("Track.ID", group_keys=False)
                .apply(compute_track_metrics))
print(f"  Done — {len(results):,} rows, {results['Track.ID'].nunique()} tracks")

# ── merge timing from timeseries ───────────────────────────────────────────────
timing = ts[["Track.ID", "Frame", "T_min", "abs_gfp_onset_min",
             "red_onset_min"]].drop_duplicates()
results = results.merge(timing, on=["Track.ID", "Frame"], how="left")
results = results.merge(md, on="Track.ID", how="left")
results["productive"] = (results["delay_green_to_red"].notna() &
                         np.isfinite(results["delay_green_to_red"]))

results.to_csv(OUT_CSV, index=False)
print(f"Saved CSV → {OUT_CSV}")

# ── plot ───────────────────────────────────────────────────────────────────────
results = results[results["abs_gfp_onset_min"].notna()].copy()
results["t_rel_h"] = (results["T_min"] - results["abs_gfp_onset_min"]) / 60.0

prod_count = results[results["productive"]]["Track.ID"].nunique()
np_count   = results[~results["productive"]]["Track.ID"].nunique()

delays_h   = results[results["productive"]].drop_duplicates("Track.ID")["delay_green_to_red"] / 60.0
mean_red_h = delays_h.mean()
print(f"  Productive: {prod_count}  |  Non-productive: {np_count}")
print(f"  Mean red onset: {mean_red_h:.1f}h")

# clip outliers
for feat, _ in FEATURES:
    if feat in results.columns:
        lo, hi = results[feat].quantile(0.01), results[feat].quantile(0.99)
        results[feat] = results[feat].clip(lo, hi)

bins = np.arange(T_MIN_H, T_MAX_H + BIN_W_H, BIN_W_H)
results["t_bin"] = (results["t_rel_h"] / BIN_W_H).round() * BIN_W_H

prod_df = results[results["productive"]]
np_df   = results[~results["productive"]]


def bin_stats(subdf, feat):
    valid   = subdf[["Track.ID", "t_bin", feat]].dropna(subset=[feat])
    grouped = valid.groupby("t_bin")[feat]
    mn  = grouped.mean()
    sem = grouped.std() / np.sqrt(grouped.count())
    n   = grouped.count()
    mask = (n >= MIN_N) & mn.index.isin(bins)
    return mn[mask], sem[mask]


def smooth(s):
    return s.rolling(ROLL_WIN, center=True, min_periods=1).mean()


def detect_inflection(times, values):
    times  = np.asarray(times, float)
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if finite.sum() < SG_WIN:
        return None
    vals_c = np.interp(times, times[finite], values[finite])
    try:
        sm = savgol_filter(vals_c, window_length=SG_WIN, polyorder=SG_POLY)
    except ValueError:
        return None
    d2   = np.gradient(np.gradient(sm, times), times)
    post = times > 0.0
    if post.sum() < 2:
        return None
    d2p, tp = d2[post], times[post]
    chg     = np.where(np.diff(np.sign(d2p)))[0]
    if not len(chg):
        return None
    i       = chg[0]
    t0, t1  = tp[i], tp[i + 1]
    d0, d1_ = d2p[i], d2p[i + 1]
    return float(t0 - d0 * (t1 - t0) / (d1_ - d0)) if d1_ != d0 else float((t0 + t1) / 2)


ncols = 3
nrows = math.ceil(len(FEATURES) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.2))
axes = axes.flatten()

summary = []
for ax, (feat, label) in zip(axes, FEATURES):
    mn_p,  sem_p  = bin_stats(prod_df, feat)
    mn_np, sem_np = bin_stats(np_df,   feat)
    mn_p_s,  sem_p_s  = smooth(mn_p),  smooth(sem_p)
    mn_np_s, sem_np_s = smooth(mn_np), smooth(sem_np)

    infl_h = detect_inflection(mn_p_s.index.values, mn_p_s.values)

    ax.fill_between(mn_np_s.index, mn_np_s - sem_np_s, mn_np_s + sem_np_s,
                    color=NP_COLOR, alpha=0.25)
    ax.plot(mn_np_s.index, mn_np_s, color=NP_COLOR, lw=1.2,
            label=f"Non-productive (n={np_count})")

    ax.fill_between(mn_p_s.index, mn_p_s - sem_p_s, mn_p_s + sem_p_s,
                    color=PROD_COLOR, alpha=0.25)
    ax.plot(mn_p_s.index, mn_p_s, color=PROD_COLOR, lw=1.8,
            label=f"Productive (n={prod_count})")

    ax.axvline(0,          color=ONSET_COLOR, lw=1.4, ls="--", label="IE onset")
    ax.axvline(mean_red_h, color=RED_COLOR,   lw=1.6, ls="--",
               label=f"Red onset ({mean_red_h:.1f}h)")

    if infl_h is not None:
        idx_n   = np.argmin(np.abs(mn_p_s.index.values - infl_h))
        y_infl  = mn_p_s.iloc[idx_n]
        y_range = mn_p_s.max() - mn_p_s.min() or 1.0
        ax.annotate(
            f"inflects\n+{infl_h:.1f}h",
            xy=(infl_h, y_infl),
            xytext=(infl_h + 1.5, y_infl + y_range * 0.15),
            fontsize=7.5, color=PROD_COLOR,
            arrowprops=dict(arrowstyle="->", color=PROD_COLOR, lw=0.9),
            ha="left", va="bottom",
        )
    summary.append((feat, f"{infl_h:+.1f}h" if infl_h is not None else "n/d"))

    ax.set_xlim(T_MIN_H, T_MAX_H)
    ax.set_xlabel("Time relative to IE onset (h)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

for ax in axes[len(FEATURES):]:
    ax.set_visible(False)

fig.suptitle(
    f"Movement Parameter Trajectories Aligned to IE Onset\n"
    f"mean ± SEM  |  rolling window {W_SHORT} frames (~2h) / {W_LONG} frames (~4h) for α  |  "
    f"dashed red = mean red onset {mean_red_h:.1f}h",
    fontsize=10, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved figure → {OUT_PNG}")

print("\nInflection times after IE onset:")
print(f"  {'Feature':<24}  Inflection")
print(f"  {'-'*24}  ----------")
for feat, t in summary:
    print(f"  {feat:<24}  {t}")

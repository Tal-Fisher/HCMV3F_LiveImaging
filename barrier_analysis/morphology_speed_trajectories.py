"""
Morphology & movement-speed trajectories aligned to GFP (IE) onset.

Produces two figures:
  morphology_speed_trajectories.png      — 6-panel original feature set
  morphology_speed_trajectories_all.png  — 12-panel full available feature set
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
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_CSV   = os.path.join(ROOT, "cache", "python_export", "timeseries_data.csv")
EF_CSV   = os.path.join(ROOT, "cache", "python_export", "extra_features.csv")
MD_CSV   = os.path.join(ROOT, "cache", "python_export", "model_df.csv")
OUT_6    = os.path.join(ROOT, "barrier_analysis", "morphology_speed_trajectories.png")
OUT_ALL  = os.path.join(ROOT, "barrier_analysis", "morphology_speed_trajectories_all.png")

# ── parameters ─────────────────────────────────────────────────────────────────
T_MIN_H  = -5.0
T_MAX_H  = 40.0
BIN_W_H  = 0.25
MIN_N    = 5
ROLL_WIN = 3
SG_WIN   = 11
SG_POLY  = 3

FEATURES_6 = [
    ("Area_cell",          "Cell Area (px²)"),
    ("Circ_cell",          "Cell Circularity"),
    ("Solidity",           "Solidity"),
    ("Shape_index",        "Shape Index"),
    ("El_long_axis",       "Elliptic Long Axis (px)"),
    ("speed_px_per_frame", "Movement Speed (px/frame)"),
]

FEATURES_ALL = [
    ("Area_cell",          "Cell Area (px²)"),
    ("Area_nuc",           "Nucleus Area (px²)"),
    ("nuc_ratio",          "Nucleus/Cell Area Ratio"),
    ("Circ_cell",          "Cell Circularity"),
    ("Circ_nuc",           "Nucleus Circularity"),
    ("Solidity",           "Solidity"),
    ("Shape_index",        "Shape Index"),
    ("El_long_axis",       "Elliptic Long Axis (px)"),
    ("Ctrst.ch4",          "BF Contrast"),
    ("Mean_ch4",           "BF Mean Intensity"),
    ("ch2_corrected",      "GFP Intensity"),
    ("Mean.ch1",           "BFP Intensity (cytoplasm)"),
    ("speed_px_per_frame", "Movement Speed (px/frame)"),
]

PROD_COLOR  = "#2980b9"
NP_COLOR    = "#888888"
ONSET_COLOR = "#27ae60"
RED_COLOR   = "#e74c3c"

# ── load & merge ───────────────────────────────────────────────────────────────
print("Loading data …")
ts = pd.read_csv(TS_CSV, low_memory=False)
ef = pd.read_csv(EF_CSV)
md = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red"]]

ts["El_long_axis"] = pd.to_numeric(ts["El_long_axis"], errors="coerce")

df = ts.merge(ef[["Track.ID", "Frame", "speed_px_per_frame", "Circ_cell", "Mean_ch4"]],
              on=["Track.ID", "Frame"], how="left")
df = df.merge(md, on="Track.ID", how="left")

df["productive"] = df["delay_green_to_red"].notna() & np.isfinite(df["delay_green_to_red"])
df = df[df["abs_gfp_onset_min"].notna()].copy()
df["t_rel_h"] = (df["T_min"] - df["abs_gfp_onset_min"]) / 60.0

prod_count = df[df["productive"]]["Track.ID"].nunique()
np_count   = df[~df["productive"]]["Track.ID"].nunique()
print(f"  Productive: {prod_count}  |  Non-productive: {np_count}")

# ── red onset reference ────────────────────────────────────────────────────────
delays_h   = df[df["productive"]].drop_duplicates("Track.ID")["delay_green_to_red"] / 60.0
mean_red_h = delays_h.mean()
print(f"  Mean red onset: {mean_red_h:.1f}h")

# ── outlier clip (1–99th pct per feature) ──────────────────────────────────────
all_feats = list({f for f, _ in FEATURES_6 + FEATURES_ALL})
for feat in all_feats:
    if feat in df.columns:
        lo, hi = df[feat].quantile(0.01), df[feat].quantile(0.99)
        df[feat] = df[feat].clip(lo, hi)

# ── time bins ─────────────────────────────────────────────────────────────────
bins = np.arange(T_MIN_H, T_MAX_H + BIN_W_H, BIN_W_H)
df["t_bin"] = (df["t_rel_h"] / BIN_W_H).round() * BIN_W_H


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
    d2      = np.gradient(np.gradient(sm, times), times)
    post    = times > 0.0
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


def draw_panel(ax, feat, label, prod_df, np_df):
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
        y_range = mn_p_s.max() - mn_p_s.min()
        ax.annotate(
            f"inflects\n+{infl_h:.1f}h",
            xy=(infl_h, y_infl),
            xytext=(infl_h + 1.5, y_infl + y_range * 0.15),
            fontsize=7.5, color=PROD_COLOR,
            arrowprops=dict(arrowstyle="->", color=PROD_COLOR, lw=0.9),
            ha="left", va="bottom",
        )

    ax.set_xlim(T_MIN_H, T_MAX_H)
    ax.set_xlabel("Time relative to IE onset (h)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    return infl_h


def make_figure(features, outpath, ncols=3):
    n     = len(features)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 4.2))
    axes = axes.flatten()
    prod_df = df[df["productive"]]
    np_df   = df[~df["productive"]]

    summary = []
    for ax, (feat, label) in zip(axes, features):
        infl_h = draw_panel(ax, feat, label, prod_df, np_df)
        summary.append((feat, f"{infl_h:+.1f}h" if infl_h is not None else "n/d"))

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Morphology & Speed Trajectories Aligned to IE Onset\n"
        f"mean ± SEM  |  dashed red = mean red onset {mean_red_h:.1f}h",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")
    return summary


# ── generate figures ───────────────────────────────────────────────────────────
print("\n── Figure 1: 6-panel ──")
summary6 = make_figure(FEATURES_6, OUT_6, ncols=3)

print("\n── Figure 2: all features ──")
summary_all = make_figure(FEATURES_ALL, OUT_ALL, ncols=4)

# ── print summary ─────────────────────────────────────────────────────────────
print("\nInflection times after IE onset (all features):")
print(f"  {'Feature':<24}  Inflection")
print(f"  {'-'*24}  ----------")
for feat, t in summary_all:
    print(f"  {feat:<24}  {t}")

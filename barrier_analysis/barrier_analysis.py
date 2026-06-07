"""
barrier_analysis.py — Feature barrier analysis for HCMV red onset

Barrier definition:
  A feature value T is a "barrier" if 85% of productive cells crossed T
  at any point between GFP onset and red onset, while fewer non-productive
  cells ever reached T during their entire observation.

  Direction per feature:
    "max" — cell must REACH AT LEAST T (e.g. GFP must rise above T)
    "min" — cell must DROP TO AT MOST T (e.g. solidity must fall below T)

Threshold logic (85% strictness):
  max features: T = 15th percentile of peak(feature, GFP onset → red onset)
  min features: T = 85th percentile of trough(feature, GFP onset → red onset)

Parameters:
  STRICTNESS = 0.15 (→ 85% of productive cells pass the barrier)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import gaussian_kde

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
TS_CSV  = BASE / "cache" / "python_export" / "timeseries_data.csv"
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"
OUT_DIR = BASE / "barrier_analysis"

STRICTNESS = 0.15  # 15th/85th percentile → 85% pass

EARLY_CUT = 911
LATE_CUT  = 2163

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9",
                "late": "#27ae60",  "non-productive": "#888888"}

# Each entry: (col, label, direction)
# direction "max" = must rise above T; "min" = must fall below T
# Features with both min and max appear twice (different biological questions)
# mCherry excluded — trivially reached by productive cells by definition
FEAT_SPECS = [
    ("ch2_corrected", "GFP (corrected) ↑",        "max"),
    ("Mean.ch1",      "BFP cytoplasm ↑",           "max"),
    ("Mean.ch1_nuc",  "BFP nucleus ↑",             "max"),
    ("Area_cell",     "Cell area ↑",               "max"),
    ("P.stuck",       "P(stuck) ↑",                "max"),
    ("Area_nuc",      "Nucleus area ↓min",         "min"),
    ("Area_nuc",      "Nucleus area ↑max",         "max"),
    ("Circ_nuc",      "Nucleus circularity ↓min",  "min"),
    ("Circ_nuc",      "Nucleus circularity ↑max",  "max"),
    ("nuc_ratio",     "Nucleus/cell area ↓min",    "min"),
    ("nuc_ratio",     "Nucleus/cell area ↑max",    "max"),
    ("Solidity",      "Cell solidity ↓min",        "min"),
    ("Solidity",      "Cell solidity ↑max",        "max"),
    ("Ctrst.ch4",     "BF contrast ↓min",          "min"),
    ("Ctrst.ch4",     "BF contrast ↑max",          "max"),
    ("Shape_index",   "Shape index ↓min",          "min"),
    ("gfp_bfp_ratio", "GFP/BFP ratio ↑",           "max"),
]

# unique key per spec for dicts
def spec_key(col, dirn):
    return f"{col}__{dirn}"

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
ts = pd.read_csv(TS_CSV, low_memory=False)
md = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red"]]
ts = ts.merge(md, on="Track.ID", how="left")

ts["t_rel_min"]    = ts["T_min"] - ts["abs_gfp_onset_min"]
ts["t_to_red_min"] = ts["red_onset_min"] - ts["t_rel_min"]   # positive = still before red
ts["productive"]   = ts["delay_green_to_red"].notna() & np.isfinite(ts["delay_green_to_red"])

# GFP / nuclear-BFP ratio (IE gene expression relative to early gene)
ts["gfp_bfp_ratio"] = ts["ch2_corrected"] / ts["Mean.ch1_nuc"].replace(0, np.nan)

def assign_group(row):
    if not row["productive"]:
        return "non-productive"
    d = row["delay_green_to_red"]
    return "early" if d <= EARLY_CUT else ("medium" if d <= LATE_CUT else "late")

ts["group"] = ts.apply(assign_group, axis=1)

prod_cells = ts[ts["productive"]]["Track.ID"].nunique()
np_cells   = ts[~ts["productive"]]["Track.ID"].nunique()
print(f"  {prod_cells} productive  {np_cells} non-productive", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Compute barrier thresholds
# ══════════════════════════════════════════════════════════════════════════════
print("Computing barriers ...", flush=True)

results = []
cell_crossing = {}   # spec_key → DataFrame of (Track.ID, group, first_cross_h, delay_h)

prod_window = ts[ts["productive"] & (ts["t_to_red_min"] >= 0)]
np_ts       = ts[~ts["productive"]]

for col, label, direction in FEAT_SPECS:
    key    = spec_key(col, direction)
    agg_fn = "max" if direction == "max" else "min"

    # ── productive: extreme value over GFP onset → red onset ──────────────────
    early_ext = (prod_window.groupby("Track.ID")[col]
                             .agg(agg_fn)
                             .dropna()
                             .reset_index()
                             .rename(columns={col: "early_ext"}))

    n_elig = len(early_ext)

    if n_elig < 10:
        results.append({"key": key, "feature": col, "label": label,
                        "direction": direction, "threshold": np.nan,
                        "n_prod_eligible": n_elig,
                        "pct_prod_cross": np.nan, "pct_nonprod_reach": np.nan,
                        "gap": np.nan})
        cell_crossing[key] = pd.DataFrame()
        continue

    pct_for_T = STRICTNESS * 100 if direction == "max" else (1 - STRICTNESS) * 100
    T = np.nanpercentile(early_ext["early_ext"], pct_for_T)

    if direction == "max":
        pct_prod    = (early_ext["early_ext"] >= T).mean() * 100
        np_extreme  = np_ts.groupby("Track.ID")[col].max().dropna()
        pct_nonprod = (np_extreme >= T).mean() * 100 if len(np_extreme) > 0 else np.nan
    else:
        pct_prod    = (early_ext["early_ext"] <= T).mean() * 100
        np_extreme  = np_ts.groupby("Track.ID")[col].min().dropna()
        pct_nonprod = (np_extreme <= T).mean() * 100 if len(np_extreme) > 0 else np.nan

    gap = pct_prod - pct_nonprod

    results.append({
        "key": key, "feature": col, "label": label, "direction": direction,
        "threshold": T, "n_prod_eligible": n_elig,
        "pct_prod_cross": pct_prod, "pct_nonprod_reach": pct_nonprod, "gap": gap,
    })

    # ── crossing time for productive cells ─────────────────────────────────────
    rows = []
    for tid, grp in ts[ts["productive"]].groupby("Track.ID"):
        grp   = grp.sort_values("t_rel_min")
        red_t = grp["red_onset_min"].iloc[0]
        delay = grp["delay_green_to_red"].iloc[0]
        group = grp["group"].iloc[0]
        vals  = grp[col].values
        times = grp["t_rel_min"].values
        lead  = red_t - times
        mask  = (vals >= T) & (lead >= 0) if direction == "max" \
                else (vals <= T) & (lead >= 0)
        cross_h = (red_t - times[mask][0]) / 60 if mask.any() else np.nan
        rows.append({"Track.ID": tid, "group": group,
                     "cross_h_before_red": cross_h, "delay_h": delay / 60})
    cell_crossing[key] = pd.DataFrame(rows)

barrier_df = pd.DataFrame(results).sort_values("gap", ascending=False)
barrier_df.to_csv(OUT_DIR / "barrier_thresholds.csv", index=False)
print("  Saved barrier_thresholds.csv", flush=True)
print(barrier_df[["label", "threshold", "pct_prod_cross",
                   "pct_nonprod_reach", "gap"]].to_string(index=False), flush=True)
thr_map = dict(zip(barrier_df["key"], barrier_df["threshold"]))
dir_map = dict(zip(barrier_df["key"], barrier_df["direction"]))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Barrier summary: productive vs non-productive crossing rates
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: barrier summary ...", flush=True)

valid = barrier_df.dropna(subset=["gap"]).copy()
valid = valid.sort_values("gap", ascending=True)   # lowest gap at top for hbar

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle(
    "Barrier analysis — feature crossing rates\n"
    "Threshold set so ≥85% of productive cells cross it before red onset\n"
    "(↑ = must rise above T;  ↓ = must fall below T)",
    fontsize=11, fontweight="bold"
)

y = np.arange(len(valid))
h = 0.35

ax.barh(y + h/2, valid["pct_prod_cross"],  h, color="#27ae60", alpha=0.85,
        label="Productive: crossed T before red onset (↑reach / ↓drop)")
ax.barh(y - h/2, valid["pct_nonprod_reach"], h, color="#888888", alpha=0.70,
        label="Non-productive: ever crossed T (raw, no window correction)")

for i, (_, row) in enumerate(valid.iterrows()):
    ax.text(row["pct_prod_cross"]  + 0.8, i + h/2,
            f"{row['pct_prod_cross']:.0f}%",
            va="center", fontsize=7.5, color="#27ae60", fontweight="bold")
    ax.text(row["pct_nonprod_reach"] + 0.8, i - h/2,
            f"{row['pct_nonprod_reach']:.0f}%",
            va="center", fontsize=7.5, color="#555")
    ax.text(102, i, f"gap={row['gap']:+.0f}pp",
            va="center", fontsize=7, color="#c0392b")

ax.set_yticks(y)
ax.set_yticklabels(valid["label"], fontsize=9)
ax.set_xlabel("Percentage of cells (%)", fontsize=9)
ax.set_xlim(0, 115)
ax.axvline(85, color="#c0392b", lw=0.8, linestyle="--", alpha=0.5, label="85% target")
ax.legend(fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

note = ("Note: non-productive cells may have shorter observation windows,\n"
        "which could underestimate their rate of reaching the threshold.")
ax.text(0.5, -0.10, note, transform=ax.transAxes, fontsize=7.5,
        ha="center", color="#888", style="italic")

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_summary.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Trajectories with barrier threshold line
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: trajectories + barrier lines ...", flush=True)

T_MAX_H = 72
t_bins  = np.linspace(0, T_MAX_H * 60, 145)

def traj_stats(sub, col, bins, min_cells=5):
    sub = sub[["t_rel_min", col]].dropna()
    idx = np.digitize(sub["t_rel_min"], bins) - 1
    ctr = 0.5 * (bins[:-1] + bins[1:])
    med, q25, q75 = [], [], []
    for i in range(len(ctr)):
        v = sub.loc[idx == i, col].values
        if len(v) >= min_cells:
            med.append(np.nanmedian(v)); q25.append(np.nanpercentile(v, 25))
            q75.append(np.nanpercentile(v, 75))
        else:
            med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
    return ctr, np.array(med), np.array(q25), np.array(q75)

# unique columns for trajectory panels (each column plotted once)
seen_cols = []
for col, label, _ in FEAT_SPECS:
    if col not in seen_cols:
        seen_cols.append(col)
unique_col_labels = {col: label.split(" ↑")[0].split(" ↓")[0]
                     for col, label, _ in FEAT_SPECS}

n_panels = len(seen_cols)
ncols_grid = 3
nrows_grid = int(np.ceil(n_panels / ncols_grid))

fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(15, nrows_grid * 3.5))
fig.suptitle(
    "Feature trajectories from GFP onset + barrier thresholds\n"
    "(dashed lines = barrier T; red=↑max threshold, blue=↓min threshold)",
    fontsize=11, fontweight="bold"
)

for ax, col in zip(axes.flat, seen_cols):
    for grp, grp_col in GROUP_COLORS.items():
        sub = ts[ts["group"] == grp]
        ctr, med, q25, q75 = traj_stats(sub, col, t_bins)
        ctr_h = ctr / 60
        v  = np.isfinite(med)
        ls = "--" if grp == "non-productive" else "-"
        lw = 1.4 if grp == "non-productive" else 1.8
        ax.plot(ctr_h[v], med[v], color=grp_col, lw=lw, linestyle=ls, label=grp)
        if grp != "non-productive":
            ax.fill_between(ctr_h[v], q25[v], q75[v], color=grp_col, alpha=0.13)

    # draw threshold lines for all specs using this column
    for _, lbl, dirn in [(c, l, d) for c, l, d in FEAT_SPECS if c == col]:
        k = spec_key(col, dirn)
        T = thr_map.get(k)
        if T is not None and np.isfinite(float(T)):
            thr_col = "#c0392b" if dirn == "max" else "#2980b9"
            arrow   = "↑" if dirn == "max" else "↓"
            ax.axhline(T, color=thr_col, lw=1.4, linestyle="--", alpha=0.85,
                       label=f"{arrow} T={T:.2g}")

    ax.set_xlabel("Time from GFP onset (h)", fontsize=7)
    ax.set_ylabel(unique_col_labels.get(col, col), fontsize=7)
    ax.set_title(unique_col_labels.get(col, col), fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, T_MAX_H)
    ax.tick_params(labelsize=6)

axes.flat[0].legend(fontsize=6, loc="upper left")
for ax in axes.flat[n_panels:]:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_trajectories_gfp.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_trajectories_gfp.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Crossing time distribution (top features by gap)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: crossing time ...", flush=True)

top4 = barrier_df.dropna(subset=["gap"]).nlargest(4, "gap")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "When do productive cells first cross the barrier?\n"
    "(x = hours before red onset that threshold was first crossed;\n"
    " shown for top 4 features by specificity gap)",
    fontsize=11, fontweight="bold"
)

for ax, (_, row) in zip(axes.flat, top4.iterrows()):
    key   = row["key"]
    label = row["label"]
    T     = row["threshold"]
    df_c  = cell_crossing[key].dropna(subset=["cross_h_before_red"])

    n_cross = len(df_c)
    n_total = prod_cells
    pct_cross = n_cross / n_total * 100

    for grp, col in [("early", GROUP_COLORS["early"]),
                     ("medium", GROUP_COLORS["medium"]),
                     ("late",   GROUP_COLORS["late"])]:
        vals = df_c.loc[df_c["group"] == grp, "cross_h_before_red"].values
        if len(vals) > 3:
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                x_g = np.linspace(0, df_c["cross_h_before_red"].max() + 2, 300)
                ax.plot(x_g, kde(x_g), color=col, lw=1.8,
                        label=f"{grp} (n={len(vals)})")
                ax.fill_between(x_g, kde(x_g), alpha=0.12, color=col)
            except Exception:
                pass

    ax.set_xlabel(f"Hours before red onset that barrier was crossed", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(
        f"{label}  (T={T:.2g})\n"
        f"{pct_cross:.0f}% of productive cells crossed before red  "
        f"| gap={row['gap']:+.0f}pp",
        fontsize=8, fontweight="bold"
    )
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(left=0)

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_crossing.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_crossing.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Scatter: crossing time vs total delay (top 4 features)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: crossing time vs delay scatter ...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Does crossing the barrier earlier predict faster red onset?\n"
    "(each dot = one productive cell; cells that never cross shown as triangles at x=0)",
    fontsize=11, fontweight="bold"
)

for ax, (_, row) in zip(axes.flat, top4.iterrows()):
    key   = row["key"]
    label = row["label"]
    T     = row["threshold"]
    df_c  = cell_crossing[key].copy()

    crossed     = df_c.dropna(subset=["cross_h_before_red"])
    not_crossed = df_c[df_c["cross_h_before_red"].isna()]

    for grp, col in [("early", GROUP_COLORS["early"]),
                     ("medium", GROUP_COLORS["medium"]),
                     ("late",   GROUP_COLORS["late"])]:
        sub = crossed[crossed["group"] == grp]
        if len(sub):
            ax.scatter(sub["cross_h_before_red"], sub["delay_h"],
                       color=col, alpha=0.5, s=14, label=grp)
        sub_nc = not_crossed[not_crossed["group"] == grp]
        if len(sub_nc):
            ax.scatter([0] * len(sub_nc), sub_nc["delay_h"],
                       color=col, alpha=0.3, s=14, marker="v")

    ax.set_xlabel("Hours before red onset that barrier was first crossed\n(triangles at x=0 = never crossed)",
                  fontsize=7.5)
    ax.set_ylabel("Total GFP→red delay (h)", fontsize=7.5)
    ax.set_title(f"{label}  (T={T:.2g})", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_scatter.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Rolling-averaged trajectories before mCherry onset
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: rolling-averaged trajectories ...", flush=True)

ROLL_FEATS = [
    ("Mean.ch1_nuc",  "BFP – nucleus",   "max"),
    ("ch2_corrected", "GFP (corrected)", "max"),
    ("Area_cell",     "Cell area",       "max"),
]
ROLL_WIN = 5   # frames ≈ 75 min

# x-axis cap: 90th percentile of per-cell max t_rel_min (captures post-red tail)
max_h = np.nanpercentile(
    ts[ts["productive"]].groupby("Track.ID")["t_rel_min"].max().values, 90
) / 60

# all productive cells — no cutoff at red onset
prod_all = ts[ts["productive"]].copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Rolling-averaged feature trajectories — productive cells (full track)\n"
    f"(individual cells = thin lines; thick line = group mean; "
    f"rolling window = {ROLL_WIN} frames ≈ {ROLL_WIN*15} min;\n"
    "x = 0 at GFP onset = track start; includes post-mCherry-onset period)",
    fontsize=10, fontweight="bold"
)

for ax, (col, label, dirn) in zip(axes, ROLL_FEATS):

    for grp, grp_col in [("early",  GROUP_COLORS["early"]),
                          ("medium", GROUP_COLORS["medium"]),
                          ("late",   GROUP_COLORS["late"])]:

        grp_sub = prod_all[prod_all["group"] == grp]
        all_t, all_y = [], []

        for tid, cell_df in grp_sub.groupby("Track.ID"):
            cell_df  = cell_df.sort_values("t_rel_min")
            t_h      = cell_df["t_rel_min"].values / 60
            y_raw    = cell_df[col].values.astype(float)
            y_smooth = (pd.Series(y_raw)
                          .rolling(ROLL_WIN, center=True, min_periods=1)
                          .mean()
                          .values)
            # individual cell line (very transparent)
            ax.plot(t_h, y_smooth, color=grp_col, alpha=0.06, lw=0.6)
            all_t.append(t_h)
            all_y.append(y_smooth)

        # group median on a common time grid
        t_grid = np.linspace(0, max_h, 200)
        mean_y = []
        for tg in t_grid:
            vals = []
            for t_h, y_s in zip(all_t, all_y):
                idx = np.searchsorted(t_h, tg)
                if 0 < idx < len(y_s):
                    vals.append(y_s[idx])
            mean_y.append(np.nanmean(vals) if len(vals) >= 3 else np.nan)

        mean_y = np.array(mean_y)
        valid  = np.isfinite(mean_y)
        ax.plot(t_grid[valid], mean_y[valid], color=grp_col, lw=2.2,
                label=f"{grp} (n={len(all_t)})")

    ax.set_xlabel("Time from GFP onset (h)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, max_h)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_rolling_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_rolling_trajectories.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Methodology text
# ══════════════════════════════════════════════════════════════════════════════
top4_summary = "\n".join(
    f"  {r['label']:30s}  T={r['threshold']:.3g}  "
    f"prod={r['pct_prod_cross']:.0f}%  nonprod={r['pct_nonprod_reach']:.0f}%  gap={r['gap']:+.0f}pp"
    for _, r in top4.iterrows()
)

text = f"""Barrier Analysis — HCMV Live Imaging
======================================
Generated by: barrier_analysis/barrier_analysis.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BARRIER DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A feature value T is a "barrier" for red onset if 85% of productive cells
crossed T at some point between GFP onset and red onset, while fewer
non-productive cells ever reached T during their full observation.

The analysis does not impose any minimum lead time — the crossing can occur
at any point before red onset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. THRESHOLD CALCULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each feature:

Step 1 — For each productive cell, compute the extreme value over the
  window from GFP onset to red onset (all frames where t_to_red_min ≥ 0):
    max features: peak   = max(feature) over that window
    min features: trough = min(feature) over that window

Step 2 — Set T:
  max features: T = {int(STRICTNESS*100)}th percentile of peak across productive cells
    → ≥{int((1-STRICTNESS)*100)}% of productive cells had peak ≥ T before red onset
  min features: T = {int((1-STRICTNESS)*100)}th percentile of trough across productive cells
    → ≥{int((1-STRICTNESS)*100)}% of productive cells had trough ≤ T before red onset

Step 3 — Non-productive comparison: what fraction of non-productive cells
  ever crossed T during their full observed track?
  Caveat: non-productive cells may have shorter observation windows,
  underestimating how often they would reach T if observed longer.

Parameters used:
  STRICTNESS = {STRICTNESS} (→ {int((1-STRICTNESS)*100)}% of productive cells must cross the barrier)

Feature directions (max = must rise above T; min = must fall below T):
  max: ch2_corrected, Mean.ch1, Mean.ch1_nuc, Area_cell, P.stuck
  min: Area_nuc, Circ_nuc, nuc_ratio, Solidity, Shape_index, Ctrst.ch4
  excluded: Mean.ch3 (mCherry — trivially reached by productive cells by definition)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Source:  A2 + A3, first-half filter (abs_gfp_onset_min ≤ movie_half_min)
  Cells:   {prod_cells} productive + {np_cells} non-productive (total 809)
  Groups:  early ≤ {EARLY_CUT} min, medium {EARLY_CUT}–{LATE_CUT} min, late > {LATE_CUT} min

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. FIGURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
barrier_summary.png:
  Horizontal bar chart showing, for each feature:
  - Green bar: % of productive cells that crossed T before red onset
  - Gray bar:  % of non-productive cells that ever reached T (raw)
  - Gap (pp):  difference — the "specificity" of the barrier
  Features sorted by gap (largest gap = best barrier).

barrier_trajectories_gfp.png:
  Median ± IQR trajectories from GFP onset for each group.
  Dashed red horizontal line = barrier threshold T.
  Shows visually at what level in the typical trajectory the barrier sits.

barrier_crossing.png:
  For the top 4 features by gap: KDE of how many hours before red onset
  each productive cell first crossed the barrier. Colored by timing group.
  A distribution shifted right means cells crossed the barrier long before
  turning red (consistent with a prerequisite rather than a concurrent event).

barrier_scatter.png:
  For the top 4 features: scatter of crossing time (x) vs total GFP→red
  delay (y) for each productive cell. Triangles at x=0 = cells that never
  crossed before red onset. Tests whether crossing earlier predicts
  turning red sooner.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. TOP 4 FEATURES (by gap)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{top4_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. INTERPRETATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Large gap (prod% >> nonprod%):
  Most productive cells reached T well before red; most non-productive cells
  never reached T → strong barrier candidate.

Small or negative gap:
  Non-productive cells reach the level just as often → T is not a meaningful
  barrier; the feature just rises with time regardless of infection outcome.

mCherry (Mean.ch3) is expected to have a near-trivial result (productive cells
are defined as those that turn red, so they trivially reach high mCherry). It
serves as a positive control / sanity check.
"""

(OUT_DIR / "methodology.txt").write_text(text)
print("  Saved methodology.txt", flush=True)
print("All done.", flush=True)

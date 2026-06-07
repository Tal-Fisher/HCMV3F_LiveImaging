"""
barrier_analysis_100pct.py — Barrier analysis: 100% productive-cell crossing

Barrier definition:
  A feature value T is the "100% barrier" if EVERY productive cell crossed T
  at some point between GFP onset and red onset.

  Direction per feature:
    "max" — cell must REACH AT LEAST T (e.g. GFP must rise above T)
             T = minimum of (max value per productive cell before red onset)
    "min" — cell must DROP TO AT MOST T (e.g. solidity must fall below T)
             T = maximum of (min value per productive cell before red onset)

This is the most conservative possible barrier: the value guaranteed to be
crossed by 100% of productive cells. Compare to the 85% version in barrier_analysis.py.
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

EARLY_CUT = 911
LATE_CUT  = 2163

GROUP_COLORS = {"early": "#e67e22", "medium": "#2980b9",
                "late": "#27ae60",  "non-productive": "#888888"}

FEAT_SPECS = [
    ("ch2_corrected", "GFP corrected ↑",             "max"),
    ("Mean.ch1",      "BFP cytoplasm ↑",             "max"),
    ("Mean.ch1_nuc",  "BFP nucleus ↑",               "max"),
    ("Area_cell",     "Cell area ↑",                 "max"),
    ("P.stuck",       "P(stuck) ↑",                  "max"),
    ("Area_nuc",      "Nucleus area ↓",              "min"),
    ("Circ_nuc",      "Nucleus circularity ↓",       "min"),
    ("nuc_ratio",     "Nucleus/cell area ↓",         "min"),
    ("Solidity",      "Cell solidity ↓",             "min"),
    ("Ctrst.ch4",     "BF contrast ↓",               "min"),
    ("Shape_index",   "Shape index ↓",               "min"),
    ("gfp_bfp_ratio", "GFP/BFP ratio ↑",             "max"),
]

def spec_key(col, dirn):
    return f"{col}__{dirn}"

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
ts = pd.read_csv(TS_CSV, low_memory=False)
md = pd.read_csv(MD_CSV)[["Track.ID", "delay_green_to_red"]]
ts = ts.merge(md, on="Track.ID", how="left")

ts["t_rel_min"]    = ts["T_min"] - ts["abs_gfp_onset_min"]
ts["t_to_red_min"] = ts["red_onset_min"] - ts["t_rel_min"]
ts["productive"]   = ts["delay_green_to_red"].notna() & np.isfinite(ts["delay_green_to_red"])
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
# Compute 100% barriers
# ══════════════════════════════════════════════════════════════════════════════
print("Computing 100% barriers ...", flush=True)

results = []
cell_crossing = {}

prod_window = ts[ts["productive"] & (ts["t_to_red_min"] >= 0)]
np_ts       = ts[~ts["productive"]]

for col, label, direction in FEAT_SPECS:
    key    = spec_key(col, direction)
    agg_fn = "max" if direction == "max" else "min"

    # extreme value per productive cell from GFP onset to red onset
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

    # 100% threshold: min of max-values (max features) or max of min-values (min features)
    if direction == "max":
        T = float(early_ext["early_ext"].min())
        pct_prod    = (early_ext["early_ext"] >= T).mean() * 100
        np_extreme  = np_ts.groupby("Track.ID")[col].max().dropna()
        pct_nonprod = (np_extreme >= T).mean() * 100 if len(np_extreme) > 0 else np.nan
    else:
        T = float(early_ext["early_ext"].max())
        pct_prod    = (early_ext["early_ext"] <= T).mean() * 100
        np_extreme  = np_ts.groupby("Track.ID")[col].min().dropna()
        pct_nonprod = (np_extreme <= T).mean() * 100 if len(np_extreme) > 0 else np.nan

    gap = pct_prod - pct_nonprod

    results.append({
        "key": key, "feature": col, "label": label, "direction": direction,
        "threshold": T, "n_prod_eligible": n_elig,
        "pct_prod_cross": pct_prod, "pct_nonprod_reach": pct_nonprod, "gap": gap,
    })

    # crossing time for productive cells
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
barrier_df.to_csv(OUT_DIR / "barrier_thresholds_100pct.csv", index=False)
print("  Saved barrier_thresholds_100pct.csv", flush=True)
print(barrier_df[["label", "threshold", "pct_prod_cross",
                   "pct_nonprod_reach", "gap"]].to_string(index=False), flush=True)
thr_map = dict(zip(barrier_df["key"], barrier_df["threshold"]))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Barrier summary: productive vs non-productive crossing rates
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: barrier summary ...", flush=True)

valid = barrier_df.dropna(subset=["gap"]).copy()
valid = valid.sort_values("gap", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle(
    "100% barrier analysis — feature crossing rates\n"
    "Threshold = value reached by ALL productive cells before red onset\n"
    "(↑ = must rise above T;  ↓ = must fall below T)",
    fontsize=11, fontweight="bold"
)

y = np.arange(len(valid))
h = 0.35

ax.barh(y + h/2, valid["pct_prod_cross"],  h, color="#27ae60", alpha=0.85,
        label="Productive: reached 100% threshold before red onset")
ax.barh(y - h/2, valid["pct_nonprod_reach"], h, color="#888888", alpha=0.70,
        label="Non-productive: ever crossed same threshold")

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
ax.axvline(100, color="#27ae60", lw=0.8, linestyle="--", alpha=0.6, label="100% target")
ax.legend(fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

note = ("Note: non-productive cells may have shorter observation windows,\n"
        "which could underestimate their rate of reaching the threshold.")
ax.text(0.5, -0.10, note, transform=ax.transAxes, fontsize=7.5,
        ha="center", color="#888", style="italic")

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_summary_100pct.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_summary_100pct.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Trajectories with barrier threshold lines
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

seen_cols = []
for col, label, _ in FEAT_SPECS:
    if col not in seen_cols:
        seen_cols.append(col)
unique_col_labels = {col: label.replace(" ↑", "").replace(" ↓", "")
                     for col, label, _ in FEAT_SPECS}

n_panels   = len(seen_cols)
ncols_grid = 3
nrows_grid = int(np.ceil(n_panels / ncols_grid))

fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(15, nrows_grid * 3.5))
fig.suptitle(
    "Feature trajectories from GFP onset + 100% barrier thresholds\n"
    "(dashed red = ↑ threshold; dashed blue = ↓ threshold)",
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

    for _, lbl, dirn in [(c, l, d) for c, l, d in FEAT_SPECS if c == col]:
        k = spec_key(col, dirn)
        T = thr_map.get(k)
        if T is not None and np.isfinite(float(T)):
            thr_col = "#c0392b" if dirn == "max" else "#2980b9"
            arrow   = "↑" if dirn == "max" else "↓"
            ax.axhline(T, color=thr_col, lw=1.4, linestyle="--", alpha=0.85,
                       label=f"{arrow} 100% T={T:.2g}")

    ax.set_xlabel("Time from GFP onset (h)", fontsize=7)
    ax.set_ylabel(unique_col_labels.get(col, col), fontsize=7)
    ax.set_title(unique_col_labels.get(col, col), fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, T_MAX_H)
    ax.tick_params(labelsize=6)

axes.flat[0].legend(fontsize=6, loc="upper left")
for ax in axes.flat[n_panels:]:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_trajectories_100pct.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_trajectories_100pct.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Crossing time distribution (top features by gap)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: crossing time ...", flush=True)

top4 = barrier_df.dropna(subset=["gap"]).nlargest(4, "gap")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "When do productive cells first cross the 100% barrier?\n"
    "(x = hours before red onset; top 4 features by non-productive gap)",
    fontsize=11, fontweight="bold"
)

for ax, (_, row) in zip(axes.flat, top4.iterrows()):
    key   = row["key"]
    label = row["label"]
    T     = row["threshold"]
    df_c  = cell_crossing[key].dropna(subset=["cross_h_before_red"])

    pct_cross = len(df_c) / prod_cells * 100

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

    ax.set_xlabel("Hours before red onset that 100% barrier was crossed", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(
        f"{label}  (T={T:.2g})\n"
        f"{pct_cross:.0f}% of productive cells crossed | gap={row['gap']:+.0f}pp",
        fontsize=8, fontweight="bold"
    )
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(left=0)

plt.tight_layout()
fig.savefig(OUT_DIR / "barrier_crossing_100pct.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved barrier_crossing_100pct.png", flush=True)

print("\nAll done.", flush=True)

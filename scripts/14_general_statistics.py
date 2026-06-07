"""
14_general_statistics.py — General descriptive statistics & distribution plots

Metrics:
  1.  delay_green_to_blue   — GFP → BFP onset (min)
  2.  delay_green_to_red    — GFP → mCherry onset (min, productive only)
  3.  delay_blue_to_red     — BFP → mCherry onset (min, cells with all 3)
  4.  red_slope             — mCherry rising rate after onset (intensity/min, from R export)
  5.  abs_gfp_onset_min     — absolute GFP onset time in movie (min)
  6.  green_onset_min       — GFP onset relative to track start (min)
  7.  productive fraction   — % cells that turn red, by dataset
  8.  gfp_corr_slope        — GFP rising rate (intensity/min, feature window)
  9.  nuc_bfp_slope         — BFP rising rate (feature window)
  10. gfp_ratio_slope       — GFP/BFP ratio slope (feature window)
  11. gfp_corr_start        — GFP intensity at onset
  12. nuc_bfp_start         — BFP level when GFP turns on

No first-half filter applied — uses all cells with valid measurements.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "general_stats"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DS_COLORS = {"A2": "#2196F3", "A3": "#FF9800"}   # blue, orange
PROD_COLORS = {"productive": "#27ae60", "non-productive": "#e74c3c"}

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
red_slope = pd.read_csv(EXPORT_DIR / "red_slope.csv")
df = df.merge(red_slope[["Track.ID", "red_slope"]], on="Track.ID", how="left")

delay = df["delay_green_to_red"].values.astype(float)
df["productive"]      = np.isfinite(delay)
df["delay_green_to_red_fin"] = np.where(df["productive"], delay, np.nan)
df["delay_blue_to_red"] = df["delay_green_to_red_fin"] - df["delay_green_to_blue"]

print(f"Total cells: {len(df)}  (A2={( df['dataset']=='A2').sum()}  A3={(df['dataset']=='A3').sum()})")
print(f"Productive:  {df['productive'].sum()}  ({100*df['productive'].mean():.1f}%)")
print(f"With BFP onset: {df['delay_green_to_blue'].notna().sum()}")
print(f"With all 3:     {df['delay_blue_to_red'].notna().sum()}")
print(f"With red_slope: {df['red_slope'].notna().sum()}")

# ── summary table ──────────────────────────────────────────────────────────────
def summarise(series, name, subset_label="all"):
    s = series.dropna()
    return {
        "metric": name, "subset": subset_label,
        "N": len(s), "mean": round(s.mean(), 2), "median": round(s.median(), 2),
        "SD": round(s.std(), 2), "Q25": round(s.quantile(0.25), 2),
        "Q75": round(s.quantile(0.75), 2),
        "min": round(s.min(), 2), "max": round(s.max(), 2),
    }

rows = []
for ds in ["A2", "A3", "all"]:
    sub = df if ds == "all" else df[df["dataset"] == ds]
    prod = sub[sub["productive"]]
    rows += [
        summarise(sub["delay_green_to_blue"],       "delay_green_to_blue (min)",    ds),
        summarise(prod["delay_green_to_red_fin"],   "delay_green_to_red (min)",     ds),
        summarise(prod["delay_blue_to_red"],        "delay_blue_to_red (min)",      ds),
        summarise(prod["red_slope"],                "red_slope (intensity/min)",    ds),
        summarise(sub["abs_gfp_onset_min"],         "abs_gfp_onset_min (min)",      ds),
        summarise(sub["green_onset_min"],           "green_onset_min (min)",        ds),
        summarise(sub["gfp_corr_slope"],            "gfp_corr_slope",               ds),
        summarise(sub["nuc_bfp_slope"],             "nuc_bfp_slope",                ds),
        summarise(sub["gfp_ratio_slope"],           "gfp_ratio_slope",              ds),
        summarise(sub["gfp_corr_start"],            "gfp_corr_start",               ds),
        summarise(sub["nuc_bfp_start"],             "nuc_bfp_start",                ds),
    ]
summary_df = pd.DataFrame(rows)
summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
print("\nSummary (all cells):")
print(summary_df[summary_df["subset"] == "all"].to_string(index=False))

# ── helper: KDE panel ──────────────────────────────────────────────────────────
def kde_panel(ax, data_dict, title, xlabel, colors, vline_fn=np.median,
              xlim=None, log_x=False):
    """data_dict: {label: array-like}"""
    any_plotted = False
    for label, vals in data_dict.items():
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            continue
        if log_x:
            v = v[v > 0]
        x = np.log10(v) if log_x else v
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        grid = np.linspace(lo, hi, 300)
        try:
            kde = gaussian_kde(x, bw_method="scott")
            ax.plot(10**grid if log_x else grid, kde(grid),
                    color=colors[label], lw=2, label=label)
            med = np.median(x)
            ax.axvline(10**med if log_x else med,
                       color=colors[label], lw=1, linestyle="--", alpha=0.7)
        except Exception:
            pass
        any_plotted = True
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    if xlim:
        ax.set_xlim(xlim)
    if any_plotted:
        ax.legend(fontsize=7)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Timing distributions
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle("HCMV live imaging — Timing statistics\n"
              f"(all cells, no first-half filter, n={len(df)})",
              fontsize=13, fontweight="bold")

prod = df[df["productive"]]
has3 = df[df["delay_blue_to_red"].notna()]

# 1. delay_green_to_blue
kde_panel(axes1[0, 0],
          {ds: df[df["dataset"]==ds]["delay_green_to_blue"] for ds in ["A2","A3"]},
          f"GFP → BFP delay\n(n={df['delay_green_to_blue'].notna().sum()})",
          "minutes", DS_COLORS)

# 2. delay_green_to_red
kde_panel(axes1[0, 1],
          {ds: prod[prod["dataset"]==ds]["delay_green_to_red_fin"] for ds in ["A2","A3"]},
          f"GFP → mCherry delay (productive only)\n(n={len(prod)})",
          "minutes", DS_COLORS)

# 3. delay_blue_to_red
kde_panel(axes1[0, 2],
          {ds: has3[has3["dataset"]==ds]["delay_blue_to_red"] for ds in ["A2","A3"]},
          f"BFP → mCherry delay (cells with all 3 onsets)\n(n={len(has3)})",
          "minutes", DS_COLORS)

# 4. abs_gfp_onset_min
kde_panel(axes1[1, 0],
          {ds: df[df["dataset"]==ds]["abs_gfp_onset_min"] for ds in ["A2","A3"]},
          f"GFP onset time (from movie start)\n(n={df['abs_gfp_onset_min'].notna().sum()})",
          "minutes", DS_COLORS)

# 5. green_onset_min (relative to track start)
kde_panel(axes1[1, 1],
          {ds: df[df["dataset"]==ds]["green_onset_min"] for ds in ["A2","A3"]},
          f"GFP onset (from track start)\n(n={df['green_onset_min'].notna().sum()})",
          "minutes", DS_COLORS)

# 6. Productive fraction bar chart
ax = axes1[1, 2]
for i, ds in enumerate(["A2", "A3"]):
    sub = df[df["dataset"] == ds]
    frac = sub["productive"].mean() * 100
    n    = len(sub)
    ax.bar(i, frac, color=DS_COLORS[ds], width=0.5, label=f"{ds} ({frac:.1f}%, n={n})")
    ax.text(i, frac + 1, f"{frac:.1f}%", ha="center", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["A2", "A3"])
ax.set_ylabel("% productive (turn red)", fontsize=8)
ax.set_ylim(0, 100)
ax.set_title("Productive fraction by dataset", fontsize=9, pad=4)
ax.legend(fontsize=7)

plt.tight_layout()
out1 = FIG_DIR / "general_stats_timing.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Dynamics distributions (split by productive / non-productive)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
fig2.suptitle("HCMV live imaging — Dynamics statistics\n"
              f"(productive vs non-productive, n={len(df)})",
              fontsize=13, fontweight="bold")

prod_colors = {"productive": "#27ae60", "non-productive": "#e74c3c"}
p_data  = {"productive": df[df["productive"]], "non-productive": df[~df["productive"]]}

def pn_dict(col):
    return {k: v[col] for k, v in p_data.items()}

kde_panel(axes2[0, 0], pn_dict("gfp_corr_slope"),
          "GFP rising rate (gfp_corr_slope)\nfeature window at GFP onset",
          "slope (a.u./frame)", prod_colors)

kde_panel(axes2[0, 1], pn_dict("nuc_bfp_slope"),
          "BFP rising rate (nuc_bfp_slope)\nfeature window at GFP onset",
          "slope (a.u./frame)", prod_colors)

# red_slope — productive only (non-productive have no red onset)
n_rs = df["red_slope"].notna().sum()
kde_panel(axes2[0, 2],
          {"A2": prod[prod["dataset"]=="A2"]["red_slope"],
           "A3": prod[prod["dataset"]=="A3"]["red_slope"]},
          f"mCherry rising rate after onset\n(productive cells, n={n_rs})",
          "slope (intensity/min)", DS_COLORS)

kde_panel(axes2[1, 0], pn_dict("gfp_ratio_slope"),
          "GFP/BFP ratio slope (gfp_ratio_slope)\nfeature window at GFP onset",
          "slope (ratio/frame)", prod_colors)

kde_panel(axes2[1, 1], pn_dict("gfp_corr_start"),
          "GFP intensity at onset (gfp_corr_start)",
          "intensity (a.u.)", prod_colors)

kde_panel(axes2[1, 2], pn_dict("nuc_bfp_start"),
          "Nuclear BFP level at GFP onset (nuc_bfp_start)",
          "intensity (a.u.)", prod_colors)

plt.tight_layout()
out2 = FIG_DIR / "general_stats_dynamics.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")
print(f"Summary table saved to {RESULTS_DIR / 'summary_table.csv'}")
print("\nDone.")

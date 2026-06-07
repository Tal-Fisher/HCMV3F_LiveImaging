"""
02_characterization.py — Descriptive characterization of blue (BFP) onset timing

Two complementary analyses:

  A. Delay=0 vs delay>0 comparison (all cells after first-half filter):
     What distinguishes cells that turn blue simultaneously with GFP from those
     that take longer? Uses first-frame (*_start) features from model_df.csv.

  B. Continuous analysis within delay>0 cells:
     What cell state, measured BEFORE blue onset, predicts how long the wait
     will be? Uses pre-onset window features from script 01.

Figures saved to: BFP_onset_analysis/figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy.stats import spearmanr, gaussian_kde, mannwhitneyu

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
ANA_DIR = BASE / "BFP_onset_analysis"
FIG_DIR = ANA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEAT_CSV  = ANA_DIR / "cache" / "pre_onset_features.csv"
META_PATH = BASE / "cache" / "python_export" / "model_df.csv"

# ── color scheme ──────────────────────────────────────────────────────────────
C_FAST = "#3498db"   # fast blue onset (short delay)
C_SLOW = "#e74c3c"   # slow blue onset (long delay)
C_ZERO = "#2ecc71"   # delay = 0 (simultaneous)
C_SIG  = "#2c3e50"   # significant bar fill (default)
C_NS   = "#bdc3c7"   # non-significant bar fill

# signal → category color for correlation bar chart
SIG_COLORS = {
    "ch2_corrected": "#27ae60",   # GFP — green
    "Area_cell":     "#7f8c8d",   # cell morphology — grey
    "Solidity":      "#7f8c8d",
    "Shape_index":   "#7f8c8d",
    "El_long_axis":  "#7f8c8d",
    "Ctrst.ch4":     "#e67e22",   # brightfield — orange
    "Area_nuc":      "#2980b9",   # nuclear morphology — blue
    "Circ_nuc":      "#2980b9",
    "nuc_ratio":     "#2980b9",
    "Mean.ch1_nuc":  "#8e44ad",   # nuclear BFP baseline — purple
}

SIG_ALPHA = 0.05

# ── load data ─────────────────────────────────────────────────────────────────
pre = pd.read_csv(FEAT_CSV)
meta = pd.read_csv(META_PATH)
meta_half = meta[meta["abs_gfp_onset_min"] <= meta["movie_half_min"]].copy()

# feature columns from pre-onset table
META_COLS = {"Track.ID", "dataset", "delay_green_to_blue", "delay_green_to_red",
             "abs_gfp_onset_min", "movie_half_min", "n_pre_onset_frames"}
feat_cols = [c for c in pre.columns if c not in META_COLS]

# tertile groups within delay>0 cells
t33 = pre["delay_green_to_blue"].quantile(1/3)
t67 = pre["delay_green_to_blue"].quantile(2/3)
fast_mask = pre["delay_green_to_blue"] <= t33
slow_mask = pre["delay_green_to_blue"] >  t67
n_fast = fast_mask.sum()
n_slow = slow_mask.sum()
n_mid  = (~fast_mask & ~slow_mask).sum()

print(f"Pre-onset feature table: {len(pre)} cells")
print(f"  Tertile cuts: {t33:.0f} min (fast≤) / {t67:.0f} min (slow>)")
print(f"  Fast: {n_fast}  Middle: {n_mid}  Slow: {n_slow}")
print(f"model_df (first-half): {len(meta_half)} cells  "
      f"(delay=0: {(meta_half['delay_green_to_blue']==0).sum()}, "
      f"delay>0: {(meta_half['delay_green_to_blue']>0).sum()})")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Spearman correlation bar: pre-onset features vs delay_green_to_blue
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Spearman correlations with delay_green_to_blue ...")

rhos, pvals, bar_colors = [], [], []
y_vals = pre["delay_green_to_blue"].values

for col in feat_cols:
    x = pre[col].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y_vals)
    if ok.sum() < 10:
        rhos.append(np.nan); pvals.append(np.nan)
    else:
        r, p = spearmanr(x[ok], y_vals[ok])
        rhos.append(r); pvals.append(p)
    # derive base signal name (strip suffix)
    base = col.rsplit("_", 1)[0] if "_" in col else col
    # try to find in SIG_COLORS; default grey
    bc = SIG_COLORS.get(base, "#7f8c8d")
    bar_colors.append(bc)

rhos   = np.array(rhos)
pvals  = np.array(pvals)
valid  = np.isfinite(rhos)

# sort by rho value
sort_idx      = np.argsort(rhos[valid])
feat_sorted   = [feat_cols[i] for i in np.where(valid)[0][sort_idx]]
rho_sorted    = rhos[valid][sort_idx]
pval_sorted   = pvals[valid][sort_idx]
color_sorted  = [bar_colors[i] for i in np.where(valid)[0][sort_idx]]
# desaturate non-significant
final_colors  = [c if p < SIG_ALPHA else C_NS
                 for c, p in zip(color_sorted, pval_sorted)]

fig, ax = plt.subplots(figsize=(10, max(6, len(feat_sorted) * 0.28)))
y_pos = range(len(feat_sorted))
ax.barh(y_pos, rho_sorted, color=final_colors, edgecolor="none", height=0.7)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-1, 1)
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_sorted, fontsize=7)
ax.set_xlabel("Spearman ρ  (pre-onset features vs delay_green_to_blue)", fontsize=9)
ax.set_title(
    f"Spearman correlation with BFP onset delay\n"
    f"Pre-onset features, cells with delay > 0  (n={valid.sum()})\n"
    f"Features from [{'{'}GFP onset, blue onset{'}'}) window — no data leakage",
    fontsize=10, fontweight="bold"
)

# legend: feature categories + significance
patches = [
    mpatches.Patch(color="#27ae60", label="GFP"),
    mpatches.Patch(color="#7f8c8d", label="Cell morphology"),
    mpatches.Patch(color="#2980b9", label="Nuclear morphology"),
    mpatches.Patch(color="#e67e22", label="Brightfield contrast"),
    mpatches.Patch(color="#8e44ad", label="Nuclear BFP (pre-onset baseline)"),
    mpatches.Patch(color=C_NS,      label="p ≥ 0.05 (not significant)"),
]
ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.85)

plt.tight_layout()
out1 = FIG_DIR / "feature_correlations.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — KDE per feature: fast vs slow tertile (delay>0 cells)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: KDE distributions fast vs slow ...")

FEATS_PER_PAGE = 6
n_pages = int(np.ceil(len(feat_cols) / FEATS_PER_PAGE))

pdf_path = FIG_DIR / "feature_distributions_fast_vs_slow.pdf"
with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        feats_page = feat_cols[page * FEATS_PER_PAGE : (page + 1) * FEATS_PER_PAGE]
        fig, axes  = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat  = axes.flatten()
        fig.suptitle(
            f"Feature distributions: fast (≤{t33:.0f} min, n={n_fast}) vs "
            f"slow (>{t67:.0f} min, n={n_slow}) blue onset\n"
            f"Pre-onset window features — page {page+1}/{n_pages}",
            fontsize=11, fontweight="bold"
        )
        for i, feat in enumerate(feats_page):
            ax = axes_flat[i]
            v_fast = pre.loc[fast_mask, feat].values.astype(float)
            v_slow = pre.loc[slow_mask, feat].values.astype(float)
            v_fast = v_fast[np.isfinite(v_fast)]
            v_slow = v_slow[np.isfinite(v_slow)]

            if len(v_fast) < 5 or len(v_slow) < 5:
                ax.set_title(feat, fontsize=8)
                ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8)
                continue

            stat, mw_p = mannwhitneyu(v_fast, v_slow, alternative="two-sided")
            p_str = f"MWU p={mw_p:.3f}" if mw_p >= 0.001 else "MWU p<0.001"

            x_lo = np.nanpercentile(np.concatenate([v_fast, v_slow]), 1)
            x_hi = np.nanpercentile(np.concatenate([v_fast, v_slow]), 99)
            x_grid = np.linspace(x_lo, x_hi, 300)

            for v, color, label in [
                (v_fast, C_FAST, f"fast ≤{t33:.0f} min (n={len(v_fast)})"),
                (v_slow, C_SLOW, f"slow >{t67:.0f} min (n={len(v_slow)})"),
            ]:
                try:
                    kde = gaussian_kde(v, bw_method="scott")
                    ax.plot(x_grid, kde(x_grid), color=color, lw=1.8, label=label)
                    ax.axvline(np.median(v), color=color, lw=0.8, linestyle="--", alpha=0.7)
                except Exception:
                    pass

            ax.set_title(f"{feat}\n{p_str}", fontsize=8, pad=3)
            ax.set_xlabel("value", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right")

        for j in range(len(feats_page), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print(f"  Saved {pdf_path}  ({n_pages} pages)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — delay=0 vs delay>0: first-frame feature comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: delay=0 vs delay>0 first-frame comparison ...")

# first-frame features available in model_df.csv (measured at GFP onset)
START_FEATS = [c for c in meta_half.columns if c.endswith("_start")
               and c not in ("gfp_snr_mean", "bf_snr_mean")]

mask0   = meta_half["delay_green_to_blue"] == 0
mask_gt = meta_half["delay_green_to_blue"] >  0
n0   = mask0.sum()
n_gt = mask_gt.sum()

n_cols = 3
n_rows = int(np.ceil(len(START_FEATS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(6, n_rows * 2.8)))
axes_flat = axes.flatten() if n_rows > 1 else axes
fig.suptitle(
    f"First-frame features at GFP onset: delay=0 (n={n0}) vs delay>0 (n={n_gt})\n"
    f"All cells after first-half filter  (combined A2+A3)",
    fontsize=11, fontweight="bold"
)

for i, feat in enumerate(START_FEATS):
    ax = axes_flat[i]
    v0  = meta_half.loc[mask0,   feat].values.astype(float)
    vgt = meta_half.loc[mask_gt, feat].values.astype(float)
    v0  = v0 [np.isfinite(v0 )]
    vgt = vgt[np.isfinite(vgt)]

    if len(v0) < 5 or len(vgt) < 5:
        ax.set_title(feat, fontsize=8)
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        continue

    _, mw_p = mannwhitneyu(v0, vgt, alternative="two-sided")
    p_str = f"MWU p={mw_p:.3f}" if mw_p >= 0.001 else "MWU p<0.001"

    x_lo = np.nanpercentile(np.concatenate([v0, vgt]), 1)
    x_hi = np.nanpercentile(np.concatenate([v0, vgt]), 99)
    x_grid = np.linspace(x_lo, x_hi, 300)

    for v, color, label in [
        (v0,  C_ZERO,  f"delay=0 (n={len(v0)})"),
        (vgt, C_SLOW,  f"delay>0 (n={len(vgt)})"),
    ]:
        try:
            kde = gaussian_kde(v, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), color=color, lw=1.8, label=label)
            ax.axvline(np.median(v), color=color, lw=0.8, linestyle="--", alpha=0.7)
        except Exception:
            pass

    ax.set_title(f"{feat}\n{p_str}", fontsize=8, pad=3)
    ax.set_xlabel("value", fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")

for j in range(len(START_FEATS), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
out3 = FIG_DIR / "first_frame_delay0_vs_delayed.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out3}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Blue vs red delay scatter (productive cells)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: blue vs red delay scatter ...")

prod = meta_half[
    meta_half["delay_green_to_blue"].notna() &
    meta_half["delay_green_to_red"].notna() &
    np.isfinite(meta_half["delay_green_to_red"])
].copy()

rho_br, p_br = spearmanr(prod["delay_green_to_blue"], prod["delay_green_to_red"])

fig, ax = plt.subplots(figsize=(7, 6))
for ds, color, marker in [("A2", "#3498db", "o"), ("A3", "#e67e22", "s")]:
    sub = prod[prod["dataset"] == ds]
    ax.scatter(sub["delay_green_to_blue"], sub["delay_green_to_red"],
               c=color, marker=marker, alpha=0.5, s=18, label=ds, edgecolors="none")

ax.set_xlabel("GFP → BFP onset delay  (min)", fontsize=11)
ax.set_ylabel("GFP → mCherry onset delay  (min)", fontsize=11)
ax.set_title(
    f"Blue onset delay vs Red onset delay\n"
    f"Productive cells  (n={len(prod)})  |  Spearman ρ={rho_br:.3f}  p={p_br:.3g}",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
plt.tight_layout()
out4 = FIG_DIR / "blue_vs_red_delay.png"
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out4}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Absolute GFP onset time: delay=0 vs delay>0
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: absolute GFP onset timing ...")

fig, ax = plt.subplots(figsize=(7, 5))
for v, color, label in [
    (meta_half.loc[mask0,   "abs_gfp_onset_min"].values, C_ZERO, f"delay=0 (n={n0})"),
    (meta_half.loc[mask_gt, "abs_gfp_onset_min"].values, C_SLOW, f"delay>0 (n={n_gt})"),
]:
    v = v[np.isfinite(v)]
    x_lo, x_hi = np.percentile(v, [1, 99])
    x_grid = np.linspace(x_lo, x_hi, 300)
    try:
        kde = gaussian_kde(v, bw_method="scott")
        ax.plot(x_grid, kde(x_grid), color=color, lw=2, label=label)
        ax.axvline(np.median(v), color=color, lw=1, linestyle="--", alpha=0.7)
    except Exception:
        pass

v0_t  = meta_half.loc[mask0,   "abs_gfp_onset_min"].dropna().values
vgt_t = meta_half.loc[mask_gt, "abs_gfp_onset_min"].dropna().values
_, mw_p_t = mannwhitneyu(v0_t, vgt_t, alternative="two-sided")
p_str_t = f"MWU p={mw_p_t:.3f}" if mw_p_t >= 0.001 else "MWU p<0.001"

ax.set_xlabel("Absolute GFP onset time in movie (min)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    f"When in the movie did GFP appear?\ndelay=0 vs delay>0  |  {p_str_t}",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
plt.tight_layout()
out5 = FIG_DIR / "gfp_onset_timing_by_group.png"
fig.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out5}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS CSV
# ══════════════════════════════════════════════════════════════════════════════
print("Summary stats CSV ...")

def stats_row(df_sub, label):
    d_blue = df_sub["delay_green_to_blue"].values.astype(float)
    d_red  = df_sub["delay_green_to_red"].values.astype(float)
    prod_frac = np.isfinite(d_red).mean()
    return {
        "group":                label,
        "n":                    len(df_sub),
        "delay_blue_median":    np.nanmedian(d_blue),
        "delay_blue_Q25":       np.nanpercentile(d_blue, 25),
        "delay_blue_Q75":       np.nanpercentile(d_blue, 75),
        "pct_productive":       round(prod_frac * 100, 1),
        "delay_red_median":     np.nanmedian(d_red[np.isfinite(d_red)]),
    }

# tertile groups in pre-onset table (delay > 0)
rows = [
    stats_row(meta_half[mask0],   "delay=0 (simultaneous)"),
    stats_row(pre[fast_mask],     f"delay>0, fast tertile (≤{t33:.0f} min)"),
    stats_row(pre[~fast_mask & ~slow_mask], f"delay>0, middle tertile"),
    stats_row(pre[slow_mask],     f"delay>0, slow tertile (>{t67:.0f} min)"),
]
stats_df = pd.DataFrame(rows)
stats_path = FIG_DIR / "summary_stats.csv"
stats_df.to_csv(stats_path, index=False)
print(f"  Saved {stats_path}")
print(stats_df.to_string(index=False))

print("\nDone.")

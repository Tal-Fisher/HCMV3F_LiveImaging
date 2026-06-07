"""
binned_proximity_analysis.py — Proximity vs delay_green_to_red within onset-time bins

Divides productive cells into 4 equal-count bins by abs_gfp_onset_min (quartiles).
Within each bin, cells were infected at roughly the same time in the movie,
so variation in infected-neighbour counts reflects genuine spatial proximity,
not just the infection-wave temporal confound.

Computes Spearman rho (proximity feature vs delay_green_to_red) per bin and
plots them alongside the pooled (raw) correlation for comparison.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
from pathlib import Path


def stouffer_combine(rho_n_list):
    """Stouffer's method: combine per-bin Spearman rhos into one Z and two-tailed p."""
    zs = [np.arctanh(r) * np.sqrt(n - 3) for r, n in rho_n_list if n > 3]
    S  = sum(zs) / np.sqrt(len(zs))
    p  = 2 * norm.sf(abs(S))
    wmean = sum(r * n for r, n in rho_n_list) / sum(n for _, n in rho_n_list)
    return float(S), float(p), float(wmean)

BASE = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT  = Path("/home/labs/ginossar/talfis/LiveImaging/proximity_analysis")

# ── load data ─────────────────────────────────────────────────────────────────
model = pd.read_csv(BASE / "cache" / "python_export" / "model_df.csv")
model = model.rename(columns={"Track.ID": "Track_ID"})

prox = pd.read_csv(OUT / "results" / "proximity_features.csv")

delay = model["delay_green_to_red"].values.astype(float)
prod  = model[np.isfinite(delay)].copy()
prod["delay_min"] = prod["delay_green_to_red"].astype(float)

df = prod.merge(
    prox[["Track_ID", "dist_nearest", "n_within_50", "n_within_100", "n_within_200"]],
    on="Track_ID", how="left"
)
df = df[df["dist_nearest"].notna()].copy()   # keep only cells with proximity data
print(f"Productive cells with proximity data: {len(df)}")

# ── merge GMM group labels ────────────────────────────────────────────────────
meta = pd.read_csv(BASE / "Forecast" / "cell_metadata.csv",
                   usecols=["Track.ID", "group"])
meta = meta.rename(columns={"Track.ID": "Track_ID"})
df = df.merge(meta, on="Track_ID", how="left")
print(f"Cells with group label: {df['group'].notna().sum()}")

# ── onset-time quartile bins ──────────────────────────────────────────────────
df["onset_h"] = df["abs_gfp_onset_min"] / 60.0
df["bin"] = pd.qcut(df["onset_h"], q=4, labels=False)   # 0,1,2,3

bin_edges = pd.qcut(df["onset_h"], q=4).cat.categories
bin_labels = [f"Q{i+1}\n{iv.left:.0f}–{iv.right:.0f} h" for i, iv in enumerate(bin_edges)]
print("\nBin composition:")
for b, lab in enumerate(bin_labels):
    sub = df[df["bin"] == b]
    print(f"  {lab.replace(chr(10),' ')}: n={len(sub)}  "
          f"onset {sub['onset_h'].min():.1f}–{sub['onset_h'].max():.1f} h")

# ── compute per-bin Spearman correlations ─────────────────────────────────────
FEATS = {
    "dist_nearest":  "Distance to nearest\ninfected neighbour (px)",
    "n_within_100":  "# infected cells\nwithin 100 px",
}

results = {}
for feat in FEATS:
    rows = []
    for b in range(4):
        sub = df[df["bin"] == b]
        x = sub[feat].values.astype(float)
        y = sub["delay_min"].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            rows.append(dict(bin=b, rho=np.nan, p=np.nan, n=0))
            continue
        rho, p = spearmanr(x[valid], y[valid])
        rows.append(dict(bin=b, rho=float(rho), p=float(p), n=int(valid.sum())))
    results[feat] = pd.DataFrame(rows)

# pooled (raw, no binning)
pooled = {}
for feat in FEATS:
    x = df[feat].values.astype(float)
    y = df["delay_min"].values.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    rho, p = spearmanr(x[valid], y[valid])
    pooled[feat] = (float(rho), float(p), int(valid.sum()))

# ── print summary ─────────────────────────────────────────────────────────────
print("\nPer-bin Spearman rho (proximity vs delay_green_to_red):")
print(f"{'Bin':<30} {'dist_nearest rho (p)':<25} {'n_within_100 rho (p)'}")
for b, lab in enumerate(bin_labels):
    r1 = results["dist_nearest"].iloc[b]
    r2 = results["n_within_100"].iloc[b]
    print(f"  {lab.replace(chr(10),' '):<28} "
          f"rho={r1['rho']:+.3f} (p={r1['p']:.3f})    "
          f"rho={r2['rho']:+.3f} (p={r2['p']:.3f})  n={r2['n']}")
r1p, r2p = pooled["dist_nearest"], pooled["n_within_100"]
print(f"  {'Pooled (all cells)':<28} "
      f"rho={r1p[0]:+.3f} (p={r1p[1]:.4f})   "
      f"rho={r2p[0]:+.3f} (p={r2p[1]:.4f})  n={r2p[2]}")

# Stouffer's combined test across bins
stouffer = {}
for feat in FEATS:
    res = results[feat]
    rho_n = [(row["rho"], row["n"]) for _, row in res.iterrows()
             if np.isfinite(row["rho"])]
    S, p, wmean = stouffer_combine(rho_n)
    stouffer[feat] = (S, p, wmean)
    print(f"\nStouffer combined ({feat}): Z={S:.3f}, p={p:.4f}, "
          f"weighted mean rho={wmean:.3f}")

# ── per-bin × per-group Spearman correlations ─────────────────────────────────
GROUPS      = ["early", "medium", "late"]
GROUP_COLS  = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}

grp_results = {}
for feat in FEATS:
    grp_rows = []
    for b in range(4):
        for g in GROUPS:
            sub = df[(df["bin"] == b) & (df["group"] == g)]
            x = sub[feat].values.astype(float)
            y = sub["delay_min"].values.astype(float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 5:
                grp_rows.append(dict(bin=b, group=g, rho=np.nan, p=np.nan, n=int(valid.sum())))
                continue
            rho, p = spearmanr(x[valid], y[valid])
            grp_rows.append(dict(bin=b, group=g, rho=float(rho), p=float(p), n=int(valid.sum())))
    grp_results[feat] = pd.DataFrame(grp_rows)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
fig.suptitle(
    "Proximity effect within onset-time bins — early / medium / late\n"
    "(each bin = cells infected at roughly the same point in the movie)",
    fontsize=12, fontweight="bold"
)

POOLED_COLOR = "#c0392b"
n_grp   = len(GROUPS)
bar_w   = 0.22
offsets = np.array([-1, 0, 1]) * bar_w   # positions for early/medium/late within each bin

for ax, (feat, ylabel) in zip(axes, FEATS.items()):
    res   = grp_results[feat]
    rho_p, p_p, n_p = pooled[feat]

    for gi, g in enumerate(GROUPS):
        sub  = res[res["group"] == g]
        rhos = sub["rho"].values
        ps   = sub["p"].values
        ns   = sub["n"].values
        x_pos = np.arange(4) + offsets[gi]

        ax.bar(x_pos, rhos, width=bar_w * 0.9,
               color=GROUP_COLS[g], edgecolor="white", linewidth=0.6,
               label=g, alpha=0.85)

        # significance markers and n labels
        for i, (r, p, n) in enumerate(zip(rhos, ps, ns)):
            if np.isnan(r) or n < 5:
                continue
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            if star:
                yoff = 0.012 if r >= 0 else -0.035
                ax.text(x_pos[i], r + yoff, star, ha="center",
                        va="bottom", fontsize=8, color=GROUP_COLS[g])
            ax.text(x_pos[i], -0.41, f"{n}", ha="center",
                    fontsize=6.5, color=GROUP_COLS[g])

    # pooled reference line
    ax.axhline(rho_p, color=POOLED_COLOR, lw=1.8, linestyle="--",
               label=f"Pooled ρ={rho_p:+.3f} (p={p_p:.4f}, n={n_p})")

    # Stouffer line (all cells combined across bins)
    S_st, p_st, wm_st = stouffer[feat]
    ax.axhline(wm_st, color="black", lw=1.4, linestyle=":",
               label=f"Within-bin ρ={wm_st:+.3f} (Stouffer p={p_st:.4f})")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(bin_labels, fontsize=8.5)
    ax.set_xlabel("Onset-time bin (quartile of abs_gfp_onset_min)", fontsize=9)
    ax.set_ylabel("Spearman ρ  (vs delay_green_to_red)", fontsize=9)
    ax.set_title(ylabel, fontsize=10)
    ax.set_ylim(-0.48, 0.48)
    ax.legend(fontsize=7.5, ncol=2)
    ax.tick_params(labelsize=8)
    # n labels row label
    ax.text(-0.5, -0.41, "n:", ha="right", fontsize=6.5, color="#555",
            transform=ax.get_xaxis_transform())

plt.tight_layout()
out = OUT / "figures" / "binned_proximity_correlations.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

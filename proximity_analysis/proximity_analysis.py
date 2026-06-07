"""
proximity_analysis.py — Does proximity to already-infected cells predict infection progression?

For each focal cell at its GFP onset frame, counts how many already-infected
(earlier-onset) cells are nearby and measures distance to nearest infected neighbour.
Tests whether these proximity features correlate with delay_green_to_red.

Scale: X/Y in pixels; cell radius ≈ 16 px (median TrackMate R).
Radii tested: 50, 100, 200 px  (≈ 3, 6, 12 cell radii).

Track ID mapping: raw spots "Track ID" integer → model_df "Track.ID" = "A2_<id>" or "A3_<id>"
A2 and A3 are separate wells — proximity computed within each well independently.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, rankdata
from pathlib import Path

BASE   = Path("/home/labs/ginossar/talfis/LiveImaging")
OUT    = Path("/home/labs/ginossar/talfis/LiveImaging/proximity_analysis")
FIG    = OUT / "figures"
RES    = OUT / "results"

RADII  = [50, 100, 200]   # pixels
CUT_EARLY = 911
CUT_MED   = 2163
DS_COLORS  = {"A2": "#2196F3", "A3": "#FF9800"}
GRP_COLORS = {"early": "#e67e22", "medium": "#2980b9", "late": "#27ae60"}


# ── helpers ───────────────────────────────────────────────────────────────────
def partial_spearman(y, x, z):
    """Spearman rho between y and x after partialling out z (rank regression)."""
    valid = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    y, x, z = y[valid], x[valid], z[valid]
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    def resid(v, cov):
        b = np.cov(v, cov)[0, 1] / np.var(cov)
        return v - b * cov
    rho, p = spearmanr(resid(ry, rz), resid(rx, rz))
    return float(rho), float(p), int(valid.sum())


def compute_proximity_for_dataset(ds_name, spots_path, radii):
    """Load spots for one dataset, compute per-cell proximity features."""
    print(f"  Loading {ds_name}...")
    spots = pd.read_csv(spots_path, usecols=["Track ID", "X", "Y", "Frame"])
    spots = spots.rename(columns={"Track ID": "Track_ID_raw"})

    # onset frame per raw track id
    onset_frame = spots.groupby("Track_ID_raw")["Frame"].min().to_dict()

    # position at onset frame per track
    onset_pos = (
        spots.sort_values("Frame")
             .groupby("Track_ID_raw", as_index=False)
             .first()[["Track_ID_raw", "X", "Y", "Frame"]]
    )
    onset_pos = onset_pos.rename(columns={"Frame": "onset_frame_num"})

    # per-frame lookup
    frame_groups = {f: sub for f, sub in spots.groupby("Frame")[["Track_ID_raw", "X", "Y"]]}

    print(f"    {len(spots):,} spots, {len(onset_frame)} tracks, {len(frame_groups)} frames")

    records = []
    for _, row in onset_pos.iterrows():
        tid = row["Track_ID_raw"]
        Fi  = row["onset_frame_num"]
        xi  = row["X"]
        yi  = row["Y"]

        sub = frame_groups.get(Fi)
        if sub is None:
            continue

        # already-infected: present at Fi with onset strictly before Fi
        sub_inf = sub[sub["Track_ID_raw"].map(onset_frame) < Fi]
        n_infected = len(sub_inf)

        if n_infected == 0:
            dist_nearest = np.nan
            ns = [0] * len(radii)
        else:
            dx = sub_inf["X"].values - xi
            dy = sub_inf["Y"].values - yi
            dists = np.sqrt(dx * dx + dy * dy)
            dist_nearest = float(dists.min())
            ns = [int((dists < r).sum()) for r in radii]

        rec = {
            "Track_ID": f"{ds_name}_{int(tid)}",
            "dataset":   ds_name,
            "onset_frame": Fi,
            "dist_nearest":         dist_nearest,
            "n_infected_at_onset":  n_infected,
        }
        for r, n in zip(radii, ns):
            rec[f"n_within_{r}"] = n
        records.append(rec)

    return pd.DataFrame(records)


# ── compute proximity ─────────────────────────────────────────────────────────
print("Computing per-cell proximity features...")
prox_parts = []
for ds, fname in [("A2", "A2_Merged_spots.csv"), ("A3", "A3_Merged_spots.csv")]:
    p = compute_proximity_for_dataset(ds, BASE / "CompleteImage" / fname, RADII)
    prox_parts.append(p)
prox = pd.concat(prox_parts, ignore_index=True)
print(f"  Total: {len(prox)} cells with proximity data")


# ── merge with model_df ───────────────────────────────────────────────────────
model = pd.read_csv(BASE / "cache" / "python_export" / "model_df.csv")
model = model.rename(columns={"Track.ID": "Track_ID"})

df = model.merge(
    prox[["Track_ID", "dist_nearest", "n_infected_at_onset"] +
         [f"n_within_{r}" for r in RADII]],
    on="Track_ID", how="left"
)

delay = df["delay_green_to_red"].values.astype(float)
df["productive"] = np.isfinite(delay)
prod = df[df["productive"]].copy()
prod["delay_min"] = prod["delay_green_to_red"].astype(float)
prod["delay_h"]   = prod["delay_min"] / 60.0

def group_label(d):
    if d <= CUT_EARLY: return "early"
    elif d <= CUT_MED:  return "medium"
    return "late"

prod["group"] = prod["delay_min"].apply(group_label)

n_with_prox = prod["dist_nearest"].notna().sum()
print(f"  Productive cells with proximity data: {n_with_prox} / {len(prod)}")


# ── correlation table ─────────────────────────────────────────────────────────
feat_cols = ["dist_nearest"] + [f"n_within_{r}" for r in RADII] + ["n_infected_at_onset"]
y = prod["delay_min"].values
z = prod["abs_gfp_onset_min"].values.astype(float)

corr_rows = []
for fc in feat_cols:
    x = prod[fc].values.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        continue
    rho,   p    = spearmanr(x[valid], y[valid])
    rho_p, p_p, n_p = partial_spearman(y, x, z)
    corr_rows.append({
        "feature":               fc,
        "N":                     int(valid.sum()),
        "spearman_rho":          round(float(rho), 3),
        "p_value":               round(float(p), 4),
        "partial_rho_ctrl_onset": round(rho_p, 3),
        "partial_p":             round(p_p, 4),
        "N_partial":             n_p,
    })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(RES / "proximity_correlations.csv", index=False)
print("\nCorrelations with delay_green_to_red:")
print(corr_df.to_string(index=False))

prox.to_csv(RES / "proximity_features.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Scatter plots
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(13, 10))
fig1.suptitle("Proximity to already-infected cells at GFP onset vs infection progression",
              fontsize=12, fontweight="bold")

# Panel 1: dist_nearest vs delay, by dataset
ax = axes[0, 0]
for ds, c in DS_COLORS.items():
    sub = prod[prod["dataset"] == ds]
    ax.scatter(sub["dist_nearest"], sub["delay_h"],
               alpha=0.4, s=12, color=c, label=f"{ds} (n={len(sub)})")
valid_d = np.isfinite(prod["dist_nearest"].values) & np.isfinite(y)
rho_d, p_d = spearmanr(prod["dist_nearest"].values[valid_d], y[valid_d])
ax.set_xlabel("Distance to nearest already-infected cell (px)")
ax.set_ylabel("GFP→Red delay (hours)")
ax.set_title(f"Nearest infected neighbour\nρ={rho_d:.3f}  p={p_d:.3f}")
ax.legend(fontsize=8)

# Panel 2: n_within_100 vs delay, by dataset
ax = axes[0, 1]
for ds, c in DS_COLORS.items():
    sub = prod[prod["dataset"] == ds]
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(sub))
    ax.scatter(sub["n_within_100"] + jitter, sub["delay_h"],
               alpha=0.4, s=12, color=c, label=f"{ds}")
x2 = prod["n_within_100"].values.astype(float)
v2 = np.isfinite(x2) & np.isfinite(y)
rho2, p2 = spearmanr(x2[v2], y[v2])
ax.set_xlabel("# already-infected cells within 100 px")
ax.set_ylabel("GFP→Red delay (hours)")
ax.set_title(f"Local infected density (r=100 px)\nρ={rho2:.3f}  p={p2:.3f}")
ax.legend(fontsize=8)

# Panel 3: dist_nearest vs delay, by timing group
ax = axes[1, 0]
for grp, c in GRP_COLORS.items():
    sub = prod[prod["group"] == grp]
    ax.scatter(sub["dist_nearest"], sub["delay_h"],
               alpha=0.5, s=12, color=c,
               label=f"{grp} (n={len(sub)})")
ax.set_xlabel("Distance to nearest already-infected cell (px)")
ax.set_ylabel("GFP→Red delay (hours)")
ax.set_title("Coloured by timing group")
ax.legend(fontsize=8)

# Panel 4: infection wave confound
ax = axes[1, 1]
ax.scatter(prod["abs_gfp_onset_min"] / 60, prod["n_infected_at_onset"],
           alpha=0.3, s=10, color="#555")
r_conf, p_conf = spearmanr(prod["abs_gfp_onset_min"], prod["n_infected_at_onset"])
ax.set_xlabel("Absolute GFP onset time (hours from movie start)")
ax.set_ylabel("# already-infected cells at onset frame")
ax.set_title(f"Infection wave confound\n(ρ={r_conf:.3f}  p={p_conf:.4f})")

plt.tight_layout()
out1 = FIG / "proximity_scatter.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Proximity features by timing group (bar chart)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.suptitle("Proximity features by timing group (early / medium / late)",
              fontsize=12, fontweight="bold")

groups_ord = ["early", "medium", "late"]
for ax, feat, ylabel in zip(
    axes2,
    ["dist_nearest", "n_within_100"],
    ["Distance to nearest infected neighbour (px)",
     "# infected cells within 100 px"]
):
    vals = [prod[prod["group"] == g][feat].dropna().values for g in groups_ord]
    means = [v.mean() if len(v) > 0 else 0 for v in vals]
    sems  = [v.std() / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals]
    colors = [GRP_COLORS[g] for g in groups_ord]
    bars = ax.bar(groups_ord, means, yerr=sems, color=colors,
                  capsize=4, edgecolor="white", linewidth=0.8)
    for bar, m, s, v in zip(bars, means, sems, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + max(means) * 0.01,
                f"n={len(v)}", ha="center", fontsize=8)
    nonempty = [v for v in vals if len(v) >= 2]
    if len(nonempty) >= 2:
        try:
            kw_stat, kw_p = kruskal(*nonempty)
            ax.set_title(f"{feat}\nKruskal-Wallis p={kw_p:.3f}")
        except Exception:
            ax.set_title(feat)
    else:
        ax.set_title(feat)
    ax.set_ylabel(ylabel, fontsize=9)

plt.tight_layout()
out2 = FIG / "proximity_by_group.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Spatial map snapshot (A2 mid-movie)
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path as _P

spots_a2 = pd.read_csv(BASE / "CompleteImage" / "A2_Merged_spots.csv",
                        usecols=["Track ID", "X", "Y", "Frame"])
spots_a2 = spots_a2.rename(columns={"Track ID": "Track_ID_raw"})
spots_a2["Track_ID"] = "A2_" + spots_a2["Track_ID_raw"].astype(str)

mid_frame = int(spots_a2["Frame"].max() // 2)
snap = spots_a2[spots_a2["Frame"] == mid_frame].copy()
snap = snap.merge(
    prod[prod["dataset"] == "A2"][["Track_ID", "delay_h", "group"]],
    on="Track_ID", how="left"
)

fig3, ax3 = plt.subplots(figsize=(7, 7))
no_g = snap[snap["group"].isna()]
ax3.scatter(no_g["X"], no_g["Y"], s=18, color="#bbb", alpha=0.5,
            label="non-productive / late onset")
for grp, c in GRP_COLORS.items():
    sub = snap[snap["group"] == grp]
    ax3.scatter(sub["X"], sub["Y"], s=30, color=c, alpha=0.75,
                label=f"{grp} (n={len(sub)})")
ax3.set_title(f"A2 — Cell positions at frame {mid_frame} (~{mid_frame*14.98/60:.0f} h)\n"
              f"coloured by timing group")
ax3.set_xlabel("X (px)"); ax3.set_ylabel("Y (px)")
ax3.legend(fontsize=9, markerscale=1.5)
ax3.set_aspect("equal")
plt.tight_layout()
out3 = FIG / "spatial_map_midmovie.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")

print("\nDone.")

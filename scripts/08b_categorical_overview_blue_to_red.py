"""
08b_categorical_overview_blue_to_red.py

Replicates categorical_overview.png for blue-to-red (BFP→mCherry) delay.
GMM G=3 on productive cells, Bayes-optimal cutoffs, same 1×2 figure layout.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
FIG_DIR    = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── load & compute blue-to-red delay ─────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]

productive  = np.isfinite(df["delay_blue_to_red"].values.astype(float))
n_nonprod   = (~productive).sum()
delays      = df.loc[productive, "delay_blue_to_red"].values.astype(float)

print(f"Total cells:      {len(df)}")
print(f"Productive (B2R): {productive.sum()}")
print(f"Non-productive:   {n_nonprod}")
print(f"Delay range:      {delays.min():.0f}–{delays.max():.0f} min  (median {np.median(delays):.0f})")

# ── GMM G=3 ───────────────────────────────────────────────────────────────────
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=10)
gmm.fit(delays.reshape(-1, 1))

# sort components by mean so early < medium < late
order = np.argsort(gmm.means_.ravel())
mu    = gmm.means_.ravel()[order]
sig   = np.sqrt(gmm.covariances_.ravel()[order])
pro   = gmm.weights_[order]

print(f"\nGMM G=3 means: {mu[0]:.0f} / {mu[1]:.0f} / {mu[2]:.0f} min")

# ── Bayes-optimal cutoffs ─────────────────────────────────────────────────────
x_grid   = np.arange(0, delays.max() + 1, 1.0)
dens_mat = np.column_stack([pro[i] * norm.pdf(x_grid, mu[i], sig[i]) for i in range(3)])
cls_pred = dens_mat.argmax(axis=1)   # 0=early, 1=medium, 2=late

def last_dominant(cls_pred, x_grid, k):
    idx = np.where(cls_pred == k)[0]
    return x_grid[idx[-1]] if len(idx) > 0 else None

cutoff1 = last_dominant(cls_pred, x_grid, 0)
cutoff2 = last_dominant(cls_pred, x_grid, 1)

# fallback: posterior crossover P2=P3 if medium never dominant
if cutoff2 is None:
    diff23 = dens_mat[:, 1] - dens_mat[:, 2]
    cross  = np.where(np.diff(np.sign(diff23)))[0]
    cutoff2 = x_grid[cross[0]] if len(cross) > 0 else (mu[1] + mu[2]) / 2

print(f"Bayes cutoffs:  {cutoff1:.0f} min  |  {cutoff2:.0f} min")

# ── assign categories ─────────────────────────────────────────────────────────
cats = np.where(delays <= cutoff1, "early",
         np.where(delays <= cutoff2, "medium", "late"))
n_early  = (cats == "early").sum()
n_medium = (cats == "medium").sum()
n_late   = (cats == "late").sum()
print(f"Categories — early: {n_early}  medium: {n_medium}  late: {n_late}")

# ── figure ────────────────────────────────────────────────────────────────────
cat_cols = {"early": "steelblue", "medium": "darkorange", "late": "tomato"}
gmm_cols = ["steelblue", "darkorange", "tomato"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Categorical analysis — BFP→mCherry delay (Blue-to-Red)", fontsize=13, fontweight="bold")

# panel A: histogram + GMM components + cutoffs
ax = axes[0]
ax.hist(delays, bins=50, density=True, color="lightgrey", edgecolor="white", linewidth=0.4)

x_seq = np.linspace(0, delays.max(), 1000)
total_dens = np.zeros(len(x_seq))
for i in range(3):
    comp = pro[i] * norm.pdf(x_seq, mu[i], sig[i])
    ax.plot(x_seq, comp, color=gmm_cols[i], lw=2,
            label=f"{'early' if i==0 else 'medium' if i==1 else 'late'} (μ={mu[i]:.0f})")
    total_dens += comp
ax.plot(x_seq, total_dens, color="black", lw=1.5, ls="--", label="GMM total")

ax.axvline(cutoff1, color="#555555", lw=2, ls="--")
ax.axvline(cutoff2, color="#555555", lw=2, ls="--")

y_top = ax.get_ylim()[1]
label_x = [cutoff1 / 2,
           (cutoff1 + cutoff2) / 2,
           cutoff2 + (delays.max() - cutoff2) / 2]
label_n = [n_early, n_medium, n_late]
label_c = ["early", "medium", "late"]
for lx, ln, lc in zip(label_x, label_n, label_c):
    ax.text(lx, y_top * 0.97, f"{lc}\nn={ln}",
            ha="center", va="top", color=cat_cols[lc], fontsize=9, fontweight="bold")

ax.set_xlabel("BFP→mCherry delay (min)")
ax.set_ylabel("Density")
ax.set_title("BFP→mCherry delay — GMM G=3")
ax.spines[["top", "right"]].set_visible(False)

# panel B: bar chart including non-productive
ax = axes[1]
bar_labels = ["early", "medium", "late", "not yet\nred"]
bar_vals   = [n_early, n_medium, n_late, n_nonprod]
bar_colors = [cat_cols["early"], cat_cols["medium"], cat_cols["late"], "#999999"]
x_pos = np.arange(len(bar_labels))
bars  = ax.bar(x_pos, bar_vals, color=bar_colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_vals) * 0.02,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=10)
ax.set_ylabel("Number of cells")
ax.set_title("Cells per category")
ax.set_ylim(0, max(bar_vals) * 1.18)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = FIG_DIR / "categorical_overview_blue_to_red.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")

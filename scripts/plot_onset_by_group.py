"""
plot_onset_by_group.py — KDE of abs_gfp_onset_min by timing group (early/medium/late)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kruskal
from pathlib import Path

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
FIG_DIR = BASE / "figures" / "combined"

CUT_EARLY = 911
CUT_MED   = 2163
COLORS = {"early": "#e74c3c", "medium": "#f39c12", "late": "#3498db"}

df = pd.read_csv(BASE / "cache" / "python_export" / "model_df.csv")
delay = df["delay_green_to_red"].values.astype(float)
prod = df[np.isfinite(delay)].copy()
prod["delay_min"] = prod["delay_green_to_red"].astype(float)

def group_label(d):
    if d <= CUT_EARLY: return "early"
    elif d <= CUT_MED:  return "medium"
    return "late"

prod["group"] = prod["delay_min"].apply(group_label)
prod["onset_h"] = prod["abs_gfp_onset_min"] / 60.0

groups = ["early", "medium", "late"]
vals   = [prod[prod["group"] == g]["onset_h"].dropna().values for g in groups]
ns     = [len(v) for v in vals]

kw_stat, kw_p = kruskal(*vals)

fig, ax = plt.subplots(figsize=(8, 5))

for g, v, c in zip(groups, vals, [COLORS[g] for g in groups]):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    grid = np.linspace(lo, hi, 400)
    kde  = gaussian_kde(v, bw_method="scott")
    n    = len(v)
    med  = np.median(v)
    ax.plot(grid, kde(grid), color=c, lw=2.2,
            label=f"{g}  (n={n}, median={med:.1f} h)")
    ax.axvline(med, color=c, lw=1.2, linestyle="--", alpha=0.7)

ax.set_xlabel("Absolute GFP onset time (hours from movie start)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    f"When in the movie do early / medium / late cells become infected?\n"
    f"Kruskal-Wallis p = {kw_p:.4f}",
    fontsize=11
)
ax.legend(fontsize=9)
ax.tick_params(labelsize=9)

plt.tight_layout()
out = FIG_DIR / "onset_timing_by_group.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

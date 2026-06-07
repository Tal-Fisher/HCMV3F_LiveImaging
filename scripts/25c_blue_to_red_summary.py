"""
25c_blue_to_red_summary.py

Load results from scripts 25 and 25b, produce:
  1. Combined summary table (Pearson r / Spearman r / R²) — printed and saved
  2. Per-cell scatter plots for all 3 methods on blue-to-red target
  3. Comparison bar chart: green-to-red vs blue-to-red for all methods

Run AFTER 25_blue_to_red_en_xgb.py and 25b_blue_to_red_tabicl.py:
  python3 25c_blue_to_red_summary.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
RESULTS_DIR = BASE / "results" / "blue_to_red"
FIG_DIR     = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── load metrics ──────────────────────────────────────────────────────────────
m_en_xgb = pd.read_csv(RESULTS_DIR / "metrics_en_xgb.csv")
m_tabicl  = pd.read_csv(RESULTS_DIR / "metrics_tabicl.csv")
metrics   = pd.concat([m_en_xgb, m_tabicl], ignore_index=True)

# ── load per-cell predictions ─────────────────────────────────────────────────
preds_en_xgb = pd.read_csv(RESULTS_DIR / "per_cell_predictions_en_xgb.csv")
preds_tabicl = pd.read_csv(RESULTS_DIR / "per_cell_predictions_tabicl.csv")

# merge into one frame (same cells, same order — verified by Track.ID)
preds = preds_en_xgb.merge(
    preds_tabicl[["Track.ID", "tabicl_pred_g2r", "tabicl_pred_b2r"]],
    on="Track.ID", how="inner"
)
print(f"Merged predictions: {len(preds)} cells")

y_g2r  = preds["y_green_to_red"].values
y_b2r  = preds["y_blue_to_red"].values
strata = preds["category"].values

# ── summary table ─────────────────────────────────────────────────────────────
def pivot_table(metrics_df):
    rows = []
    for method in ["ElasticNet", "XGBoost", "TabICL"]:
        for target in ["green_to_red", "blue_to_red"]:
            row = metrics_df[(metrics_df["method"] == method) &
                             (metrics_df["target"] == target)]
            if len(row) == 0:
                continue
            rows.append({
                "Method":     method,
                "Target":     "Green→Red" if target == "green_to_red" else "Blue→Red",
                "Pearson r":  round(float(row["pearson_r"].iloc[0]), 3),
                "Spearman r": round(float(row["spearman_r"].iloc[0]), 3),
                "R²":         round(float(row["R2"].iloc[0]), 3),
            })
    return pd.DataFrame(rows)

table = pivot_table(metrics)

bar_width = max(len(str(v)) for v in table.values.flatten())
print("\n" + "=" * 68)
print("  REGRESSION SUMMARY — A2+A3 productive cells (n=497)")
print("=" * 68)
print(table.to_string(index=False))
print("=" * 68)

table.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
print(f"\nSaved → results/blue_to_red/summary_table.csv")

# ── figure 1: scatter plots for all 3 methods × blue-to-red ──────────────────
colors  = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}
methods = [
    ("ElasticNet", "en_pred_b2r"),
    ("XGBoost",    "xgb_pred_b2r"),
    ("TabICL",     "tabicl_pred_b2r"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Blue→Red Delay — CV Predictions vs Actual\nA2+A3 productive cells  |  35 features",
    fontsize=13, fontweight="bold"
)

for ax, (method_name, pred_col) in zip(axes, methods):
    y_pred = preds[pred_col].values
    r2  = r2_score(y_b2r, y_pred)
    r   = float(pearsonr(y_b2r, y_pred)[0])
    rho = float(spearmanr(y_b2r, y_pred)[0])
    for cat in ["early", "medium", "late"]:
        mask = strata == cat
        ax.scatter(y_b2r[mask] / 60, y_pred[mask] / 60,
                   color=colors[cat], alpha=0.45, s=14, label=cat)
    lo = min(y_b2r.min(), y_pred.min()) / 60 * 0.92
    hi = max(y_b2r.max(), y_pred.max()) / 60 * 1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Actual BFP→mCherry delay (h)", fontsize=10)
    ax.set_ylabel("CV predicted delay (h)", fontsize=10)
    ax.set_title(f"{method_name}\nR²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

plt.tight_layout()
fig.savefig(FIG_DIR / "blue_to_red_all_methods_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/blue_to_red_all_methods_scatter.png")

# ── figure 2: comparison bar chart green-to-red vs blue-to-red ───────────────
method_order = ["ElasticNet", "XGBoost", "TabICL"]
metric_names = ["Pearson r", "Spearman r", "R²"]
metric_cols  = ["pearson_r", "spearman_r", "R2"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Green→Red vs Blue→Red regression — all methods",
    fontsize=13, fontweight="bold"
)

for ax, mname, mcol in zip(axes, metric_names, metric_cols):
    x      = np.arange(len(method_order))
    width  = 0.35
    g2r_v  = [float(metrics[(metrics["method"] == m) & (metrics["target"] == "green_to_red")][mcol].iloc[0])
               for m in method_order]
    b2r_v  = [float(metrics[(metrics["method"] == m) & (metrics["target"] == "blue_to_red")][mcol].iloc[0])
               for m in method_order]
    bars1  = ax.bar(x - width/2, g2r_v, width, label="Green→Red", color="#2ecc71", alpha=0.85)
    bars2  = ax.bar(x + width/2, b2r_v, width, label="Blue→Red",  color="#9b59b6", alpha=0.85)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(method_order, fontsize=10)
    ax.set_ylabel(mname, fontsize=11)
    ax.set_title(mname, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(g2r_v), max(b2r_v)) * 1.2)

plt.tight_layout()
fig.savefig(FIG_DIR / "blue_to_red_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/blue_to_red_comparison_bars.png")

# ── figure 3: delay distribution (green-to-red vs blue-to-red) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Delay distributions — productive cells  (n=497)", fontsize=12, fontweight="bold")

for ax, vals, color, label in [
    (axes[0], y_g2r / 60, "#2ecc71", "Green→Red (GFP→mCherry)"),
    (axes[1], y_b2r / 60, "#9b59b6", "Blue→Red  (BFP→mCherry)"),
]:
    ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(np.median(vals), color="black", lw=1.5, ls="--",
               label=f"Median {np.median(vals):.1f}h")
    ax.set_xlabel("Delay (h)", fontsize=10)
    ax.set_ylabel("Cell count", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9, frameon=False)

plt.tight_layout()
fig.savefig(FIG_DIR / "blue_to_red_delay_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/blue_to_red_delay_distributions.png")

print("\nAll done.")

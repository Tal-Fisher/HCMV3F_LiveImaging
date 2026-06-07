"""
plot_xgboost_summary.py — Comprehensive XGBoost results plot

4 panels:
  1. Regression: predicted vs true delay (productive cells)
  2. ROC curve (binary: productive vs not)
  3. Feature importances — regression (top 15)
  4. Model comparison: XGBoost vs TabPFN on all shared metrics
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
RES_XG  = BASE / "results" / "xgboost"
RES_TAB = BASE / "results" / "tabpfn"
FIG_DIR = BASE / "figures" / "combined"

# ── load data ─────────────────────────────────────────────────────────────────
preds   = pd.read_csv(RES_XG / "per_cell_predictions.csv")
roc     = pd.read_csv(RES_XG / "roc_curve.csv")
fimp    = pd.read_csv(RES_XG / "feature_importances.csv")
metrics_xg  = pd.read_csv(RES_XG  / "summary_metrics.csv")
metrics_tab = pd.read_csv(RES_TAB / "summary_metrics.csv")

prod = preds[preds["productive"] == True].copy()
prod["true_h"] = prod["delay_green_to_red"].astype(float) / 60.0
prod["pred_h"] = prod["pred_delay"].astype(float) / 60.0
prod = prod.dropna(subset=["true_h", "pred_h"])

r2_xg  = metrics_xg[metrics_xg["metric"] == "R²"]["value"].values[0]
rho_xg = metrics_xg[metrics_xg["metric"] == "Spearman rho"]["value"].values[0]
auc_xg = metrics_xg[metrics_xg["metric"] == "AUC-ROC"]["value"].values[0]

# per-fold stats from log
fold_reg = [(0.034, 28), (-0.016, 158), (0.080, 56)]   # (test R², n_est)
fold_bin = [(0.601,  9), ( 0.592,   1), (0.669, 23)]   # (test AUC, n_est)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "XGBoost + Optuna  —  HCMV live imaging  (A2+A3, first-half filter)\n"
    "3-fold outer CV  |  50 Optuna trials/fold  |  Bug-fixed: retrained on full fold",
    fontsize=12, fontweight="bold"
)

DS_COLORS = {"A2": "#2196F3", "A3": "#FF9800"}

# ── Panel 1: regression scatter ───────────────────────────────────────────────
ax = axes[0, 0]
for ds, c in DS_COLORS.items():
    sub = prod[prod["dataset"] == ds]
    ax.scatter(sub["pred_h"], sub["true_h"], alpha=0.45, s=12, color=c, label=ds)

lo = min(prod["pred_h"].min(), prod["true_h"].min()) - 2
hi = max(prod["pred_h"].max(), prod["true_h"].max()) + 2
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("CV predicted delay (hours)", fontsize=9)
ax.set_ylabel("True delay (hours)", fontsize=9)
ax.set_title(
    f"Regression  (productive only, n={len(prod)})\n"
    f"R²={r2_xg:.3f}   Spearman ρ={rho_xg:.3f}\n"
    f"Per-fold R²: {fold_reg[0][0]:+.3f}  {fold_reg[1][0]:+.3f}  {fold_reg[2][0]:+.3f}",
    fontsize=9
)
ax.legend(fontsize=8)

# ── Panel 2: ROC curve ────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(roc["fpr"], roc["tpr"], color="#e74c3c", lw=2,
        label=f"XGBoost  AUC={auc_xg:.3f}")
# TabPFN AUC reference
auc_tab = float(metrics_tab[metrics_tab["metric"] == "AUC-ROC"]["value"].values[0])
ax.axhline(auc_tab, color="#3498db", lw=1.3, linestyle="--",
           label=f"TabPFN AUC={auc_tab:.3f} (reference)")
ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4, label="Chance")
ax.set_xlabel("False positive rate", fontsize=9)
ax.set_ylabel("True positive rate", fontsize=9)
ax.set_title("ROC curve — Binary (productive vs non-productive)", fontsize=9)
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── Panel 3: feature importances (regression) ─────────────────────────────────
ax = axes[1, 0]
top = fimp.nlargest(15, "importance_reg")
y_pos = np.arange(len(top))
ax.barh(y_pos, top["importance_reg"], color="#27ae60", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top["feature"], fontsize=8)
ax.set_xlabel("Mean feature importance (weight — split frequency)", fontsize=9)
ax.set_title("Top 15 features — Regression task", fontsize=9)
ax.invert_yaxis()

# ── Panel 4: model comparison bar chart ──────────────────────────────────────
ax = axes[1, 1]

metrics_compare = [
    ("R²\n(regression)", r2_xg,
     float(metrics_tab[metrics_tab["metric"] == "R²"]["value"].values[0])),
    ("Spearman ρ\n(regression)", rho_xg,
     float(metrics_tab[metrics_tab["metric"] == "Spearman rho"]["value"].values[0])),
    ("AUC\n(binary)", auc_xg, auc_tab),
]

x = np.arange(len(metrics_compare))
w = 0.32
labels   = [m[0] for m in metrics_compare]
xgb_vals = [m[1] for m in metrics_compare]
tab_vals = [m[2] for m in metrics_compare]

bars1 = ax.bar(x - w/2, xgb_vals, w, color="#e74c3c", label="XGBoost (fixed)", alpha=0.85)
bars2 = ax.bar(x + w/2, tab_vals, w, color="#3498db", label="TabPFN",          alpha=0.85)

for bar, val in zip(bars1, xgb_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=8)
for bar, val in zip(bars2, tab_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("XGBoost vs TabPFN", fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, max(max(xgb_vals), max(tab_vals)) * 1.18)
ax.axhline(0, color="black", lw=0.6)

plt.tight_layout()
out = FIG_DIR / "xgboost_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

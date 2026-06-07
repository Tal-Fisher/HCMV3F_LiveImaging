"""
21_early_vs_rest_summary.py

Merges classification results from all three methods (Elastic Net, TabICL, XGBoost)
and produces combined comparison figures.

Run after scripts 13, 19, 20 have completed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR = BASE / "results" / "early_vs_rest_summary"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.15   # used for TabICL and XGBoost; EN uses its native 0.5 threshold

# ── load per-cell predictions ─────────────────────────────────────────────────
en_df  = pd.read_csv(BASE / "results/elasticnet_early/per_cell_predictions.csv")
tab_df = pd.read_csv(BASE / "results/tabicl_early/per_cell_predictions.csv")
xgb_df = pd.read_csv(BASE / "results/xgboost_early/per_cell_predictions.csv")

df = en_df[["Track.ID", "true_label"]].copy()
df = df.merge(en_df[["Track.ID",  "prob_early", "pred_label"]].rename(
                  columns={"prob_early": "prob_en",  "pred_label": "pred_en"}),  on="Track.ID")
df = df.merge(tab_df[["Track.ID", "prob_early", "pred_label"]].rename(
                  columns={"prob_early": "prob_tab", "pred_label": "pred_tab"}), on="Track.ID")
df = df.merge(xgb_df[["Track.ID", "prob_early", "pred_label"]].rename(
                  columns={"prob_early": "prob_xgb", "pred_label": "pred_xgb"}), on="Track.ID")

print(f"Merged predictions: {len(df)} cells")

y = df["true_label"].values

# ── compute metrics — use saved pred_label (each method's native threshold) ──
def metrics(y, proba, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc  = roc_auc_score(y, proba)
    bal  = balanced_accuracy_score(y, pred)
    return {"AUC": auc, "Sensitivity": sens, "Specificity": spec, "Balanced acc": bal}

m_en  = metrics(y, df["prob_en"].values,  df["pred_en"].values)
m_tab = metrics(y, df["prob_tab"].values, df["pred_tab"].values)
m_xgb = metrics(y, df["prob_xgb"].values, df["pred_xgb"].values)

print(f"\n{'='*65}")
print(f"  CLASSIFICATION SUMMARY — early vs medium+late  (n={len(df)} cells)")
print(f"{'='*65}")
print(f"{'Method':<14}  {'AUC':>6}  {'Sensitivity':>11}  {'Specificity':>11}  {'BalAcc':>8}")
for name, m in [("ElasticNet", m_en), ("TabICL", m_tab), ("XGBoost", m_xgb)]:
    print(f"{name:<14}  {m['AUC']:>6.3f}  {m['Sensitivity']:>11.3f}  "
          f"{m['Specificity']:>11.3f}  {m['Balanced acc']:>8.3f}")
print(f"{'='*65}")

# ── save summary table ────────────────────────────────────────────────────────
rows = []
for name, m in [("ElasticNet", m_en), ("TabICL", m_tab), ("XGBoost", m_xgb)]:
    rows.append({"method": name, **{k: round(v, 4) for k, v in m.items()}})
pd.DataFrame(rows).to_csv(RESULTS_DIR / "summary_table.csv", index=False)

# ── figure 1: overlaid ROC curves ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
for name, proba, color in [
    ("Elastic Net", df["prob_en"].values,  "#95a5a6"),
    ("TabICL",      df["prob_tab"].values, "steelblue"),
    ("XGBoost",     df["prob_xgb"].values, "darkorange"),
]:
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC curves — early vs medium+late\n"
             f"n={len(df)} productive cells  |  45 features  |  5-fold CV")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(FIG_DIR / "early_vs_rest_all_methods_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/early_vs_rest_all_methods_roc.png")

# ── figure 2: comparison bar chart ───────────────────────────────────────────
metric_labels = ["AUC", "Sensitivity", "Specificity", "Balanced acc"]
methods       = [("Elastic Net", m_en, "#95a5a6"),
                 ("TabICL",      m_tab, "steelblue"),
                 ("XGBoost",     m_xgb, "darkorange")]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metric_labels))
w = 0.25
for i, (name, m, color) in enumerate(methods):
    vals = [m[k] for k in metric_labels]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=color)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="red", lw=0.8, ls="--", alpha=0.5, label="Chance")
ax.set_ylabel("Score")
ax.set_title(f"Early vs Medium+Late classification — all methods\n"
             f"n={len(df)} productive cells  |  45 features  |  threshold={THRESHOLD}")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "early_vs_rest_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/early_vs_rest_comparison_bars.png")

# ── figure 3: predicted probability distributions ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(
    f"Predicted probability distributions — early vs medium+late\n"
    f"(blue=early cells, grey=medium+late cells  |  dashed line=threshold {THRESHOLD})",
    fontsize=11, fontweight="bold"
)
for ax, name, proba, color in [
    (axes[0], "Elastic Net", df["prob_en"].values,  "#95a5a6"),
    (axes[1], "TabICL",      df["prob_tab"].values, "steelblue"),
    (axes[2], "XGBoost",     df["prob_xgb"].values, "darkorange"),
]:
    ax.hist(proba[y == 0], bins=30, alpha=0.5, color="grey",  label="Med+Late", density=True)
    ax.hist(proba[y == 1], bins=30, alpha=0.7, color="#2471a3", label="Early",   density=True)
    ax.axvline(THRESHOLD, color="black", ls="--", lw=1.2, label=f"thr={THRESHOLD}")
    auc = roc_auc_score(y, proba)
    ax.set_xlabel("P(early)")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}\nAUC={auc:.3f}")
    ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "early_vs_rest_prob_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/early_vs_rest_prob_distributions.png")

print("\nAll done.")

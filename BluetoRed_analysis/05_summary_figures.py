"""
05_summary_figures.py
Summary comparison figures across all methods for both tasks.
Run after all four analysis scripts have completed.
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ANA_DIR = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR = ANA_DIR / "results"
FIG_DIR = ANA_DIR / "figures"

# ── load metrics ───────────────────────────────────────────────────────────────
reg_en_xgb = pd.read_csv(RES_DIR / "reg_en_xgb_metrics.csv")
reg_tabicl  = pd.read_csv(RES_DIR / "reg_tabicl_metrics.csv")
cls_en      = pd.read_csv(RES_DIR / "cls_en_metrics.csv")
cls_xgb     = pd.read_csv(RES_DIR / "cls_xgb_metrics.csv")
cls_tabicl  = pd.read_csv(RES_DIR / "cls_tabicl_metrics.csv")

reg_df = pd.concat([reg_en_xgb, reg_tabicl], ignore_index=True)
cls_df = pd.concat([cls_en, cls_xgb, cls_tabicl], ignore_index=True)

print("Regression results:")
print(reg_df[["method","R2","pearson_r","spearman_rho","perm_p"]].to_string(index=False))
print("\nClassification results:")
print(cls_df[["method","AUC","sensitivity","specificity","bal_acc","spearman_rho","perm_p"]].to_string(index=False))

methods    = reg_df["method"].tolist()
method_cols = {"ElasticNet": "#27ae60", "XGBoost": "#8e44ad", "TabICL": "#e67e22"}
bar_colors  = [method_cols.get(m, "#7f8c8d") for m in methods]
display_names = {"ElasticNet": "Linear regression\n+ regularization", "XGBoost": "XGBoost", "TabICL": "TabICL"}
display_labels = [display_names.get(m, m) for m in methods]

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION summary — 2 metrics side by side
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
fig.suptitle("Regression: delay BFP→mCherry (blue-to-red)\nA2+A3, 451 cells, 45 features, no half-movie filter",
             fontsize=12, fontweight="bold")

for ax, col, ylabel, title in [
    (axes[0], "R2",          "R²",           "R²"),
    (axes[1], "spearman_rho","Spearman ρ",   "Spearman ρ"),
]:
    vals = reg_df[col].values
    bars = ax.bar(display_labels, vals, color=bar_colors, width=0.5,
                  edgecolor="white", zorder=3)
    for bar, v, m in zip(bars, vals, methods):
        p_val = reg_df.loc[reg_df["method"]==m, "perm_p"].values[0]
        star  = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.02,
                f"{v:.3f}\n{star}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.22)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.tick_params(axis="x", labelsize=8)

plt.tight_layout()
fig.savefig(str(FIG_DIR/"summary_regression.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("\nSaved figures/summary_regression.png")

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION summary — AUC + BalAcc + Spearman ρ
# ══════════════════════════════════════════════════════════════════════════════
cls_methods       = cls_df["method"].tolist()
cls_colors        = [method_cols.get(m, "#7f8c8d") for m in cls_methods]
cls_display_names = {"ElasticNet": "Linear regression\n+ regularization", "XGBoost": "XGBoost", "TabICL": "TabICL"}
cls_display_labels = [cls_display_names.get(m, m) for m in cls_methods]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
n_early = int(cls_df["n_early"].iloc[0])
cutoff  = int(cls_df["gmm_cutoff_min"].iloc[0])
fig.suptitle(f"Classification: early vs med+late (BFP→mCherry ≤ {cutoff} min)\n"
             f"A2+A3, {cls_df['n'].iloc[0]} cells, {n_early} early, 33 features",
             fontsize=12, fontweight="bold")

for ax, col, ylabel, title in [
    (axes[0], "AUC",          "AUC",           "AUC-ROC"),
    (axes[1], "bal_acc",      "Balanced Acc",  "Balanced Accuracy"),
    (axes[2], "spearman_rho", "Spearman ρ",    "Spearman ρ(delay, proba)"),
]:
    vals = cls_df[col].values
    bars = ax.bar(cls_display_labels, vals, color=cls_colors, width=0.5,
                  edgecolor="white", zorder=3)
    for bar, v, m in zip(bars, vals, cls_methods):
        p_val = cls_df.loc[cls_df["method"]==m, "perm_p"].values[0]
        star  = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.02,
                f"{v:.3f}\n{star}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    if col == "AUC":
        ax.axhline(0.5, color="#95a5a6", lw=1.2, ls="--", alpha=0.7, label="Chance = 0.500")
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    if col == "AUC":
        ax.set_ylim(0, 1)
    else:
        ymin = max(0, min(vals) - 0.05)
        ax.set_ylim(ymin, max(vals)*1.22)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

patches = [mpatches.Patch(color=method_cols[m], label=cls_display_names.get(m, m)) for m in cls_methods]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
fig.savefig(str(FIG_DIR/"summary_classification.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved figures/summary_classification.png")

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED ROC — all three classifiers on one plot
# ══════════════════════════════════════════════════════════════════════════════
en_pred  = pd.read_csv(RES_DIR/"cls_en_xgb_predictions.csv")
xgb_pred = en_pred.copy()
tab_pred = pd.read_csv(RES_DIR/"cls_tabicl_predictions.csv")

from sklearn.metrics import roc_curve, roc_auc_score
y_true = en_pred["y_true"].values

fig, ax = plt.subplots(figsize=(5, 5))
for proba_col, label, color, df_src in [
    ("en_proba",     "ElasticNet", "#27ae60", en_pred),
    ("xgb_proba",    "XGBoost",    "#8e44ad", xgb_pred),
    ("tabicl_proba", "TabICL",     "#e67e22", tab_pred),
]:
    proba = df_src[proba_col].values
    auc   = roc_auc_score(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{label}  AUC={auc:.3f}")

ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random AUC=0.500")
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title(f"Early vs Med+Late — BFP→mCherry\n(n={len(y_true)}, {y_true.sum()} early, cutoff={cutoff} min)")
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"summary_roc_all.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved figures/summary_roc_all.png")

# ── combined metrics table ─────────────────────────────────────────────────────
all_metrics = pd.concat([
    reg_df.assign(task="regression")[["task","method","R2","pearson_r","spearman_rho","perm_p"]],
    cls_df.assign(task="classification")[["task","method","AUC","bal_acc","spearman_rho","perm_p"]],
], ignore_index=True)
all_metrics.to_csv(RES_DIR/"all_metrics_summary.csv", index=False)
print("\nSaved results/all_metrics_summary.csv")
print("\nAll done.")

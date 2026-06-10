"""
Figure 4: Overlaid ROC curves for Blue→Red early-vs-slow classification.
Three models: GLM, XGBoost, TabICL. AUC scores annotated on the curves.
TabICL permutation-test null ROC shown as shaded band.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# ── data ──────────────────────────────────────────────────────────────────────
base = "/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/results/"

en     = pd.read_csv(base + "cls_en_predictions.csv")
xgb    = pd.read_csv(base + "cls_xgb_predictions.csv")
tabicl = pd.read_csv(base + "cls_tabicl_predictions.csv")

MODELS = [
    ("GLM",     en,     "en_proba",     "#1f77b4"),
    ("XGBoost", xgb,    "xgb_proba",    "#ff7f0e"),
    ("TabICL",  tabicl, "tabicl_proba", "#2ca02c"),
]

# ── null ROC for TabICL (500-permutation Monte Carlo) ─────────────────────────
N_PERM = 500
rng = np.random.default_rng(42)
mean_fpr = np.linspace(0, 1, 300)
null_tprs = []
null_aucs = []

y_true = tabicl["y_true"].values
y_proba = tabicl["tabicl_proba"].values

for _ in range(N_PERM):
    y_shuf = rng.permutation(y_true)
    fpr_n, tpr_n, _ = roc_curve(y_shuf, y_proba)
    null_tprs.append(np.interp(mean_fpr, fpr_n, tpr_n))
    null_aucs.append(auc(fpr_n, tpr_n))

null_tprs = np.array(null_tprs)
null_mean_tpr = null_tprs.mean(axis=0)
null_std_tpr  = null_tprs.std(axis=0)
null_mean_auc = float(np.mean(null_aucs))

# ── figure ────────────────────────────────────────────────────────────────────
FONTSIZE = 34
LW = 3.0

fig, ax = plt.subplots(figsize=(10, 10))

# Null ROC band (behind real curves)
ax.fill_between(mean_fpr,
                null_mean_tpr - null_std_tpr,
                null_mean_tpr + null_std_tpr,
                color="#2ca02c", alpha=0.15)
ax.plot(mean_fpr, null_mean_tpr, color="#2ca02c", linewidth=LW,
        linestyle=":", label=f"TabICL permutation  (AUC = {null_mean_auc:.3f})")

for name, df, proba_col, color in MODELS:
    fpr, tpr, _ = roc_curve(df["y_true"], df[proba_col])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=LW,
            label=f"{name}  (AUC = {roc_auc:.3f})")

# Chance diagonal
ax.plot([0, 1], [0, 1], color="0.6", linewidth=1.5, linestyle="--")

ax.set_xlabel("False Positive Rate", fontsize=FONTSIZE)
ax.set_ylabel("True Positive Rate", fontsize=FONTSIZE, labelpad=16)
ax.tick_params(axis="both", labelsize=FONTSIZE - 4)

ax.legend(fontsize=FONTSIZE - 6, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, -0.15),
          ncol=1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.subplots_adjust(bottom=0.35)

script_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(script_dir, "fig4_classification_roc.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(script_dir, "fig4_classification_roc.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig4.  Null AUC = {null_mean_auc:.3f}")

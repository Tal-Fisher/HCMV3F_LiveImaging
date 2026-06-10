"""
Figure 4: Overlaid ROC curves for Blue→Red early-vs-slow classification.
Three models: GLM, XGBoost, TabICL. AUC scores annotated on the curves.
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

# ── figure ────────────────────────────────────────────────────────────────────
FONTSIZE = 34
LW = 3.0

fig, ax = plt.subplots(figsize=(10, 10))

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

ax.legend(fontsize=FONTSIZE - 6, frameon=False, loc="lower right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(script_dir, "fig4_classification_roc.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(script_dir, "fig4_classification_roc.png"), dpi=150, bbox_inches="tight")
print("Saved fig4.")

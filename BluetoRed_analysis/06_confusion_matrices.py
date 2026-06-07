"""
06_confusion_matrices.py
Confusion matrix plots for BFP→mCherry early vs medium+late classification.
Three models: ElasticNet, XGBoost, TabICL. Threshold = Youden's J per model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RES_DIR = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/results")
FIG_DIR = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/figures")

# load predictions
en_df  = pd.read_csv(RES_DIR / "cls_en_predictions.csv")
xgb_df = pd.read_csv(RES_DIR / "cls_xgb_predictions.csv")
tab_df = pd.read_csv(RES_DIR / "cls_tabicl_predictions.csv")

# load per-model Youden thresholds
thresh_en  = float(pd.read_csv(RES_DIR / "cls_en_metrics.csv")["threshold"].values[0])
thresh_xgb = float(pd.read_csv(RES_DIR / "cls_xgb_metrics.csv")["threshold"].values[0])
thresh_tab = float(pd.read_csv(RES_DIR / "cls_tabicl_metrics.csv")["threshold"].values[0])

# merge on Track.ID so all share the same rows
df = en_df[["Track.ID","y_true","en_proba"]].merge(
         xgb_df[["Track.ID","xgb_proba"]], on="Track.ID").merge(
         tab_df[["Track.ID","tabicl_proba"]], on="Track.ID")

y_true    = df["y_true"].values
n_early   = int(y_true.sum())
n_medlate = int((y_true == 0).sum())

models = [
    ("Linear reg.\n+ regularization", df["en_proba"].values,  thresh_en,  "#27ae60"),
    ("XGBoost",                        df["xgb_proba"].values, thresh_xgb, "#8e44ad"),
    ("TabICL",                         df["tabicl_proba"].values, thresh_tab, "#e67e22"),
]

CLASS_LABELS = ["Slow", "Fast"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    f"Confusion matrices — BFP→mCherry early vs med+late\n"
    f"A2+A3, n={len(y_true)} cells  ({n_early} early, {n_medlate} med+late)  |  Youden's J threshold",
    fontsize=12, fontweight="bold"
)

for ax, (name, proba, thresh, color) in zip(axes, models):
    y_pred = (proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100  # row-normalized: each row sums to 100%

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_pct, display_labels=CLASS_LABELS)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".1f")

    # replace auto-generated text with "X.X%\n(n)" format
    for text_obj in ax.texts:
        text_obj.set_visible(False)
    for i in range(2):
        for j in range(2):
            pct = cm_pct[i, j]
            n   = cm[i, j]
            txt_color = "white" if pct > 60 else "black"
            ax.text(j, i - 0.1, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=txt_color)
            ax.text(j, i + 0.22, f"(n={n})",
                    ha="center", va="center", fontsize=9, color=txt_color)

    tp = cm[1, 1]; fn = cm[1, 0]; tn = cm[0, 0]; fp = cm[0, 1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal  = (sens + spec) / 2

    ax.set_title(
        f"{name}\nthresh={thresh:.2f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}",
        fontsize=9, fontweight="bold", color=color
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)

plt.tight_layout()
out = FIG_DIR / "confusion_matrices.png"
fig.savefig(str(out), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

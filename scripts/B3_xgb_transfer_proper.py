"""
B3_xgb_transfer_proper.py

Apply the saved A2+A3 XGBoost model to B3 with all proper filters:
  - first-half filter (track start <= movie half)
  - nucleus-assigned cells only
  - observed mCherry onset (finite positive delay)

Data exported by R: cache/python_export/model_df_B3_proper.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
RESULTS_DIR = BASE / "results" / "xgboost"
FIG_DIR     = BASE / "figures" / "B3"

# ── load model and training scaler ────────────────────────────────────────────
model = XGBRegressor()
model.load_model(str(RESULTS_DIR / "xgb_regression_A2A3.json"))

scaler_mean  = np.load(str(RESULTS_DIR / "xgb_scaler_mean.npy"))
scaler_scale = np.load(str(RESULTS_DIR / "xgb_scaler_scale.npy"))

with open(RESULTS_DIR / "xgb_feat_cols.json") as f:
    feat_cols = json.load(f)

print(f"Model loaded — {len(feat_cols)} features, {model.n_estimators} trees")

# ── load properly filtered B3 data ───────────────────────────────────────────
df = pd.read_csv(BASE / "cache/python_export/model_df_B3_proper.csv")
print(f"B3 cells (first-half + nucleus + mCherry): {len(df)}")

# build feature matrix — median-impute with training column medians
X_raw = df[feat_cols].values.astype(float)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    if bad.any():
        X_raw[bad, j] = scaler_mean[j]   # impute with training mean

# apply training scaler
X = (X_raw - scaler_mean) / scaler_scale

# ── predict ───────────────────────────────────────────────────────────────────
y_pred_min   = model.predict(X)
y_actual_min = df["delay_green_to_red"].values.astype(float)

actual_h = y_actual_min / 60
pred_h   = y_pred_min   / 60

r2  = r2_score(actual_h, pred_h)
r   = float(np.corrcoef(actual_h, pred_h)[0, 1])
rho = float(spearmanr(actual_h, pred_h).statistic)
mae = float(np.mean(np.abs(actual_h - pred_h)))

print(f"\nB3 transfer (proper filters):")
print(f"  n={len(df)}  R²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}  MAE={mae:.1f} h")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.2))

ax.scatter(actual_h, pred_h, color="steelblue", alpha=0.55, s=20, linewidths=0)

lims = [min(actual_h.min(), pred_h.min()) * 0.92,
        max(actual_h.max(), pred_h.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=0.9, alpha=0.5, label="1:1")

m, b = np.polyfit(actual_h, pred_h, 1)
xs = np.array(lims)
ax.plot(xs, m * xs + b, color="steelblue", lw=1.8, label="fit")

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual GFP→mCherry delay (h)", fontsize=11)
ax.set_ylabel("Predicted delay — XGBoost A2+A3 (h)", fontsize=11)
ax.set_title(
    f"B3 — XGBoost transfer (A2+A3 model)\n"
    f"n={len(df)}  R²={r2:.2f}  r={r:.2f}  ρ={rho:.2f}",
    fontsize=11
)
ax.legend(fontsize=9, frameon=False)
ax.text(0.03, 0.97, "Filters: first-half + nucleus + mCherry",
        transform=ax.transAxes, fontsize=8, color="grey",
        va="top")

plt.tight_layout()
fig.savefig(FIG_DIR / "xgb_transfer_B3_proper.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved figures/B3/xgb_transfer_B3_proper.png")

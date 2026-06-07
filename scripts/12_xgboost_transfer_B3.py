"""
12_xgboost_transfer_B3.py

Train XGBoost regression on full A2+A3 data using the hyperparameters
already selected by Optuna (averaged across the 3 CV folds), save the
model, then apply it to B3 and report transfer performance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
FIG_DIR     = BASE / "figures" / "B3"
RESULTS_DIR = BASE / "results" / "xgboost"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── averaged hyperparameters from the 3 Optuna folds ─────────────────────────
params_df = pd.read_csv(RESULTS_DIR / "best_params_regression.csv")
avg_params = {
    "max_depth":          int(round(params_df["max_depth"].mean())),
    "learning_rate":      float(params_df["learning_rate"].mean()),
    "subsample":          float(params_df["subsample"].mean()),
    "colsample_bytree":   float(params_df["colsample_bytree"].mean()),
    "min_child_weight":   int(round(params_df["min_child_weight"].mean())),
    "reg_alpha":          float(params_df["reg_alpha"].mean()),
    "reg_lambda":         float(params_df["reg_lambda"].mean()),
    "gamma":              float(params_df["gamma"].mean()),
    "n_estimators":       int(round(params_df["n_estimators"].mean())),
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             -1,
}
print("Averaged hyperparameters:")
for k, v in avg_params.items():
    print(f"  {k}: {v}")

# ── load A2+A3 ────────────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)
print(f"\nA2+A3: {len(df)} cells after first-half filter")

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y", "gfp_snr_mean", "bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

delay     = df["delay_green_to_red"].values.astype(float)
prod_mask = np.isfinite(delay)
X_raw     = df[feat_cols].values.astype(float)
col_med   = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler().fit(X_raw[prod_mask])
X_prod = scaler.transform(X_raw[prod_mask])
y_prod = delay[prod_mask]
print(f"Training on {len(y_prod)} productive cells, {len(feat_cols)} features")

# ── train on full A2+A3 and save ──────────────────────────────────────────────
model = XGBRegressor(**avg_params)
model.fit(X_prod, y_prod, verbose=False)

model.save_model(str(RESULTS_DIR / "xgb_regression_A2A3.json"))
np.save(str(RESULTS_DIR / "xgb_scaler_mean.npy"), scaler.mean_)
np.save(str(RESULTS_DIR / "xgb_scaler_scale.npy"), scaler.scale_)
with open(RESULTS_DIR / "xgb_feat_cols.json", "w") as f:
    json.dump(feat_cols, f)
print(f"Model saved to results/xgboost/xgb_regression_A2A3.json")

# in-sample check
pred_train = model.predict(X_prod)
r2_tr  = r2_score(y_prod, pred_train)
rho_tr = spearmanr(y_prod, pred_train).statistic
print(f"In-sample (A2+A3): R²={r2_tr:.3f}  ρ={rho_tr:.3f}")

# ── load B3 and predict ───────────────────────────────────────────────────────
b3 = pd.read_csv(EXPORT_DIR / "model_df_B3.csv")
delay_b3 = b3["delay_green_to_red"].values.astype(float)
prod_b3  = np.isfinite(delay_b3)
print(f"\nB3: {prod_b3.sum()} productive cells")

# align features — fill missing with 0 after scaling
X_b3_raw = np.zeros((len(b3), len(feat_cols)))
for j, fc in enumerate(feat_cols):
    if fc in b3.columns:
        vals = b3[fc].values.astype(float)
        bad  = ~np.isfinite(vals)
        vals[bad] = col_med[j] if np.isfinite(col_med[j]) else 0.0
        X_b3_raw[:, j] = vals

X_b3 = scaler.transform(X_b3_raw)

# ── green-to-red prediction ───────────────────────────────────────────────────
pred_b3 = model.predict(X_b3[prod_b3])
actual_b3 = delay_b3[prod_b3]

r2_b3  = r2_score(actual_b3, pred_b3)
rho_b3 = spearmanr(actual_b3, pred_b3).statistic
r_b3   = float(np.corrcoef(actual_b3, pred_b3)[0, 1])
mae_b3 = float(np.mean(np.abs(actual_b3 - pred_b3)))

print(f"\n── Transfer: XGBoost A2+A3 → B3 green-to-red ──")
print(f"n={prod_b3.sum()}  R²={r2_b3:.3f}  r={r_b3:.3f}  ρ={rho_b3:.3f}")
print(f"MAE={mae_b3:.0f} min ({mae_b3/60:.1f} h)")
print(f"\n(Reference — ElasticNet A2+A3 → B3: R²=-1.747  r=0.128  ρ=0.113)")
print(f"(Reference — XGBoost A2+A3 CV:      R²=0.041   ρ=0.285)")

# ── plot ──────────────────────────────────────────────────────────────────────
actual_h = actual_b3 / 60
pred_h   = pred_b3   / 60
lim = [min(actual_h.min(), pred_h.min()) * 0.9,
       max(actual_h.max(), pred_h.max()) * 1.05]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("XGBoost (A2+A3) → B3 transfer", fontsize=11, fontweight="bold")

# left: green-to-red
ax = axes[0]
ax.scatter(actual_h, pred_h, color="steelblue", alpha=0.5, s=18)
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual GFP→mCherry delay (h)")
ax.set_ylabel("Predicted delay (h)")
ax.set_title(f"Green-to-red  (n={prod_b3.sum()})\nR²={r2_b3:.3f}   r={r_b3:.3f}   ρ={rho_b3:.3f}")

# right: comparison bar chart
ax = axes[1]
methods  = ["EN\nA2+A3 CV", "XGB\nA2+A3 CV", "EN\nA2+A3→B3", "XGB\nA2+A3→B3"]
r2_vals  = [0.101, 0.041, -1.747, r2_b3]
rho_vals = [0.322, 0.285,  0.113, rho_b3]
x = np.arange(len(methods))
w = 0.35
ax.bar(x - w/2, r2_vals,  width=w, label="R²",          color="steelblue", alpha=0.8)
ax.bar(x + w/2, rho_vals, width=w, label="Spearman ρ",  color="tomato",    alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Model comparison")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "xgb_transfer_B3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved figures/B3/xgb_transfer_B3.png")

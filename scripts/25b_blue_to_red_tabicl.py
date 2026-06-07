"""
25b_blue_to_red_tabicl.py

TabICL regression for BFP→mCherry (blue-to-red) delay.
Uses the same 45-feature extended dataset as 19_elasticnet_productive_only.R.
Also runs green-to-red for direct comparison.

Run with: /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12 25b_blue_to_red_tabicl.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from tabicl import TabICLRegressor
except ImportError:
    raise ImportError(
        "tabicl not found — run with:\n"
        "/home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12"
    )

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
EXT_DIR     = BASE / "results" / "elasticnet_extended2"
RESULTS_DIR = BASE / "results" / "blue_to_red"
FIG_DIR     = BASE / "figures" / "combined"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
N_EST   = 8
SEED    = 42

# ── load 45-feature extended dataset ─────────────────────────────────────────
ext_df = pd.read_csv(EXT_DIR / "model_df_extended2.csv")
print(f"Extended dataset: {len(ext_df)} rows × {ext_df.shape[1]} cols")

META_COLS = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue"}
EXTRAS_18 = {
    "cell_aspect_start", "cell_aspect_mean", "bfp_nuc_frac_start",
    "nuc_ratio_start", "nuc_ratio_end",
    "bf_ctrst_start", "bf_ctrst_end", "bf_ctrst_slope",
}
feat_cols = [c for c in ext_df.columns if c not in META_COLS and c not in EXTRAS_18]
print(f"Features: {len(feat_cols)}")

cat_df = pd.read_csv(EXPORT_DIR / "category_df.csv")
df = ext_df.merge(cat_df[["Track.ID", "category"]], on="Track.ID", how="left")

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]

prod_mask = np.isfinite(df["delay_green_to_red"].values.astype(float))
cat_mask  = df["category"].isin(["early", "medium", "late"])
df_prod   = df[prod_mask & cat_mask].reset_index(drop=True)
print(f"Productive cells (early/mid/late): {len(df_prod)}  "
      f"({df_prod['category'].value_counts().to_dict()})")

y_g2r  = df_prod["delay_green_to_red"].values.astype(float)
y_b2r  = df_prod["delay_blue_to_red"].values.astype(float)
strata = df_prod["category"].values

# ── feature matrix (median impute) ────────────────────────────────────────────
X_raw = df_prod[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# ── TabICL regression helper ──────────────────────────────────────────────────
def run_tabicl(X, y, strata, label):
    print(f"\n── TabICL Regression [{label}] ──")
    print(f"  n={len(y)} | {X.shape[1]} features | {N_FOLDS}-fold stratified CV "
          f"| n_estimators={N_EST}")
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_preds = np.zeros(len(y))
    for fold, (tr, te) in enumerate(skf.split(X, strata)):
        scaler = StandardScaler().fit(X[tr])
        X_tr   = scaler.transform(X[tr])
        X_te   = scaler.transform(X[te])
        reg    = TabICLRegressor(n_estimators=N_EST, random_state=SEED)
        reg.fit(X_tr, y[tr])
        cv_preds[te] = reg.predict(X_te)
        r2_f = r2_score(y[te], cv_preds[te])
        r_f  = float(pearsonr(y[te], cv_preds[te])[0])
        print(f"  Fold {fold+1}: n_test={len(te):3d}  R²={r2_f:.3f}  r={r_f:.3f}")
    r2  = r2_score(y, cv_preds)
    r   = float(pearsonr(y, cv_preds)[0])
    rho = float(spearmanr(y, cv_preds)[0])
    print(f"  CV overall  R²={r2:.3f}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")
    return cv_preds, r2, r, rho

preds_g2r, r2_g2r, r_g2r, rho_g2r = run_tabicl(X_raw, y_g2r, strata, "Green→Red")
preds_b2r, r2_b2r, r_b2r, rho_b2r = run_tabicl(X_raw, y_b2r, strata, "Blue→Red")

# ── save ─────────────────────────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "Track.ID":          df_prod["Track.ID"],
    "category":          strata,
    "y_green_to_red":    y_g2r.round(2),
    "y_blue_to_red":     y_b2r.round(2),
    "tabicl_pred_g2r":   preds_g2r.round(2),
    "tabicl_pred_b2r":   preds_b2r.round(2),
})
pred_df.to_csv(RESULTS_DIR / "per_cell_predictions_tabicl.csv", index=False)

metrics_df = pd.DataFrame([
    {"method": "TabICL", "target": "green_to_red", "R2": round(r2_g2r, 4),
     "pearson_r": round(r_g2r, 4), "spearman_r": round(rho_g2r, 4)},
    {"method": "TabICL", "target": "blue_to_red",  "R2": round(r2_b2r, 4),
     "pearson_r": round(r_b2r, 4), "spearman_r": round(rho_b2r, 4)},
])
metrics_df.to_csv(RESULTS_DIR / "metrics_tabicl.csv", index=False)
print(f"\nSaved results → {RESULTS_DIR}")

# ── figures ───────────────────────────────────────────────────────────────────
colors = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"TabICL Regression — A2+A3  (n={len(df_prod)} productive cells)\n"
    f"{N_FOLDS}-fold stratified CV  |  n_estimators={N_EST}  |  "
    f"{len(feat_cols)} features",
    fontsize=12, fontweight="bold"
)

for ax, y_true, y_pred, title in [
    (axes[0], y_g2r, preds_g2r,
     f"Green→Red (reference)\nR²={r2_g2r:.3f}  r={r_g2r:.3f}  ρ={rho_g2r:.3f}"),
    (axes[1], y_b2r, preds_b2r,
     f"Blue→Red\nR²={r2_b2r:.3f}  r={r_b2r:.3f}  ρ={rho_b2r:.3f}"),
]:
    for cat, grp in df_prod.groupby("category"):
        idx = grp.index.tolist()
        ax.scatter(y_true[idx] / 60, y_pred[idx] / 60,
                   color=colors.get(cat, "grey"), alpha=0.45, s=14, label=cat)
    lo = min(y_true.min(), y_pred.min()) / 60 * 0.92
    hi = max(y_true.max(), y_pred.max()) / 60 * 1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Actual delay (h)")
    ax.set_ylabel("CV predicted delay (h)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)

plt.tight_layout()
fig.savefig(FIG_DIR / "blue_to_red_tabicl.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → figures/combined/blue_to_red_tabicl.png")

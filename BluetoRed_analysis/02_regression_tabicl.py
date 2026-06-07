"""
02_regression_tabicl.py
TabICL regression for delay_blue_to_red (BFP→mCherry).
A2+A3, 45-feature extended set, productive-only filter (no half-movie filter).
Includes permutation test (500 perms, train-label shuffle).
Run with: /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12
"""

import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
EXT_DIR    = BASE / "results" / "elasticnet_extended2"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
FIG_DIR    = ANA_DIR / "figures"
SEED, N_PERM, N_EST, N_FOLDS = 42, 500, 8, 5

# ── data — 45-feature extended set ────────────────────────────────────────────
ext_df = pd.read_csv(EXT_DIR / "model_df_extended2.csv")
filt_df = pd.read_csv(EXPORT_DIR / "model_df.csv")[
    ["Track.ID", "abs_gfp_onset_min", "movie_half_min"]]
df = ext_df.merge(filt_df, on="Track.ID", how="left")

META_COLS = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
             "abs_gfp_onset_min","movie_half_min"}
EXTRAS_18 = {"cell_aspect_start","cell_aspect_mean","bfp_nuc_frac_start",
             "nuc_ratio_start","nuc_ratio_end",
             "bf_ctrst_start","bf_ctrst_end","bf_ctrst_slope"}
feat_cols = [c for c in ext_df.columns if c not in META_COLS and c not in EXTRAS_18]

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]
# productive-only: both delays observed; NO half-movie filter for regression
mask = np.isfinite(df["delay_blue_to_red"].values.astype(float))
df = df[mask].reset_index(drop=True)

y      = df["delay_blue_to_red"].values.astype(float)
strata = np.array(pd.qcut(y, q=3, labels=["early","medium","late"]).astype(str))

X_raw  = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

print(f"n={len(y)} cells | {len(feat_cols)} features", flush=True)

cat_colors = {"early":"#e74c3c","medium":"#f39c12","late":"#2980b9"}

# ── TabICL CV ─────────────────────────────────────────────────────────────────
print(f"\n═══ TabICL ({N_FOLDS}-fold, n_estimators={N_EST}) ═══", flush=True)
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof     = np.zeros(len(y))

for fold, (tr, te) in enumerate(skf.split(X_raw, strata)):
    sc  = StandardScaler().fit(X_raw[tr])
    reg = TabICLRegressor(n_estimators=N_EST, random_state=SEED)
    reg.fit(sc.transform(X_raw[tr]), y[tr])
    oof[te] = reg.predict(sc.transform(X_raw[te]))
    r_f = float(pearsonr(y[te], oof[te])[0])
    print(f"  Fold {fold+1}: n_test={len(te)}  r={r_f:.3f}", flush=True)

r2  = r2_score(y, oof)
r   = float(pearsonr(y, oof)[0])
rho = float(spearmanr(y, oof)[0])
print(f"  CV: R²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}", flush=True)

# scatter
fig, ax = plt.subplots(figsize=(5, 5))
for cat in ["early","medium","late"]:
    m = strata == cat
    ax.scatter(y[m]/60, oof[m]/60, color=cat_colors[cat],
               alpha=0.5, s=16, label=cat, edgecolors="none")
lo = min(y.min(), oof.min())/60*0.92
hi = max(y.max(), oof.max())/60*1.06
ax.plot([lo,hi],[lo,hi],"k--",lw=0.8,alpha=0.5)
ax.set_xlabel("Actual delay BFP→mCherry (h)")
ax.set_ylabel("CV predicted (h)")
ax.set_title(f"TabICL — delay BFP→mCherry\nR²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}  (n={len(y)})")
ax.legend(fontsize=8, frameon=False)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"reg_tabicl_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── permutation test ──────────────────────────────────────────────────────────
print(f"\n  Running permutation test ({N_PERM} perms)...", flush=True)
rng = np.random.default_rng(SEED)
null_rhos = []
for i in range(N_PERM):
    null_oof = np.zeros(len(y))
    for tr, te in skf.split(X_raw, strata):
        y_shuf = y[tr].copy(); rng.shuffle(y_shuf)
        sc  = StandardScaler().fit(X_raw[tr])
        reg = TabICLRegressor(n_estimators=N_EST, random_state=SEED)
        reg.fit(sc.transform(X_raw[tr]), y_shuf)
        null_oof[te] = reg.predict(sc.transform(X_raw[te]))
    null_rhos.append(float(spearmanr(y, null_oof)[0]))
    if (i+1) % 50 == 0:
        print(f"  perm {i+1}/{N_PERM}", flush=True)

null_rhos = np.array(null_rhos)
p = (null_rhos >= rho).mean()

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(null_rhos, bins=40, color="#95a5a6", edgecolor="white", label="Null (shuffled)")
ax.axvline(rho, color="#e74c3c", lw=2, label=f"Observed ρ={rho:.3f}")
ax.set_xlabel("Spearman ρ"); ax.set_ylabel("Count")
ax.set_title(f"TabICL permutation test\ndelay BFP→mCherry  p={p:.3f}")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"reg_tabicl_permtest.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Permutation p={p:.3f}", flush=True)

# ── save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "Track.ID": df["Track.ID"], "dataset": df["dataset"],
    "y_true": y.round(2), "tabicl_pred": oof.round(2), "strata": strata,
}).to_csv(RES_DIR/"reg_tabicl_predictions.csv", index=False)

pd.DataFrame([{"method":"TabICL","R2":round(r2,4),"pearson_r":round(r,4),
               "spearman_rho":round(rho,4),"perm_p":round(p,4),
               "n":len(y),"n_feat":len(feat_cols)}
             ]).to_csv(RES_DIR/"reg_tabicl_metrics.csv", index=False)
np.save(RES_DIR/"reg_tabicl_null_rhos.npy", null_rhos)

print(f"\nTabICL regression: R²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}  p={p:.3f}")
print("Done.", flush=True)

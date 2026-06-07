"""
04_classify_tabicl.py
TabICL classification: early vs medium+late based on delay_blue_to_red GMM cutoff.
497 cells (A2+A3), 33 features, half-movie filter.
Includes permutation test (500 perms, train-label shuffle).
Run with: /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12
"""

import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import norm as sp_norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from tabicl import TabICLClassifier
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
FIG_DIR    = ANA_DIR / "figures"
SEED, N_PERM, N_EST, N_FOLDS = 42, 500, 8, 5

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
f16 = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
df = df.merge(f16, on="Track.ID", how="left")

NON_FEAT = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
            "green_onset_min","track_start_min","abs_gfp_onset_min",
            "movie_half_min","y","gfp_snr_mean","bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]
# productive-only: outcome confirmed observed; NO half-movie filter
mask = np.isfinite(df["delay_blue_to_red"].values.astype(float))
df = df[mask].reset_index(drop=True)

delays = df["delay_blue_to_red"].values.astype(float)

X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# ── GMM cutoff ────────────────────────────────────────────────────────────────
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=10)
gmm.fit(delays.reshape(-1, 1))
order = np.argsort(gmm.means_.ravel())
mu    = gmm.means_.ravel()[order]
sig   = np.sqrt(gmm.covariances_.ravel()[order])
pro   = gmm.weights_[order]

x_grid   = np.arange(0, delays.max() + 1, 1.0)
dens_mat = np.column_stack([pro[i] * sp_norm.pdf(x_grid, mu[i], sig[i]) for i in range(3)])
cls_pred = dens_mat.argmax(axis=1)
idx_early = np.where(cls_pred == 0)[0]
cutoff1   = x_grid[idx_early[-1]] if len(idx_early) > 0 else mu[0] + sig[0]

y_cls   = (delays <= cutoff1).astype(int)
n_early = y_cls.sum(); n_rest = (y_cls==0).sum()
print(f"GMM cutoff: {cutoff1:.0f} min  |  early={n_early}  med+late={n_rest}")
print(f"n={len(y_cls)} cells | {len(feat_cols)} features | "
      f"n_estimators={N_EST}", flush=True)

# ── TabICL CV ─────────────────────────────────────────────────────────────────
print(f"\n═══ TabICL ({N_FOLDS}-fold) ═══", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(y_cls))

for fold, (tr, te) in enumerate(skf.split(X_raw, y_cls)):
    sc  = StandardScaler().fit(X_raw[tr])
    clf = TabICLClassifier(n_estimators=N_EST, random_state=SEED)
    clf.fit(sc.transform(X_raw[tr]), y_cls[tr])
    oof[te] = clf.predict_proba(sc.transform(X_raw[te]))[:, 1]
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_cls[te], oof[te]):.3f}", flush=True)

auc  = roc_auc_score(y_cls, oof)
fpr_c, tpr_c, thresholds_c = roc_curve(y_cls, oof)
thresh = float(thresholds_c[np.argmax(tpr_c - fpr_c)])
pred = (oof >= thresh).astype(int)
tp = int(((pred==1)&(y_cls==1)).sum()); fn = int(((pred==0)&(y_cls==1)).sum())
tn = int(((pred==0)&(y_cls==0)).sum()); fp = int(((pred==1)&(y_cls==0)).sum())
sens = tp/(tp+fn) if tp+fn>0 else 0
spec = tn/(tn+fp) if tn+fp>0 else 0
bal  = balanced_accuracy_score(y_cls, pred)
rho  = float(spearmanr(delays, oof)[0])
print(f"  CV: AUC={auc:.3f}  thresh={thresh:.3f}  Sens={sens:.3f}  "
      f"Spec={spec:.3f}  BalAcc={bal:.3f}  ρ={rho:.3f}", flush=True)

# ROC
fpr, tpr, _ = roc_curve(y_cls, oof)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, color="#e67e22", label=f"TabICL  AUC={auc:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random")
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title(f"TabICL — early vs med+late (BFP→mCherry)\n(n={len(y_cls)}, {n_early} early)")
ax.legend(fontsize=9, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"cls_tabicl_roc.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ── permutation test ──────────────────────────────────────────────────────────
print(f"\n  Running permutation test ({N_PERM} perms)...", flush=True)
rng = np.random.default_rng(SEED)
null_rhos = []
for i in range(N_PERM):
    null_oof = np.zeros(len(y_cls))
    for tr, te in skf.split(X_raw, y_cls):
        y_shuf = y_cls[tr].copy(); rng.shuffle(y_shuf)
        sc  = StandardScaler().fit(X_raw[tr])
        clf = TabICLClassifier(n_estimators=N_EST, random_state=SEED)
        clf.fit(sc.transform(X_raw[tr]), y_shuf)
        null_oof[te] = clf.predict_proba(sc.transform(X_raw[te]))[:, 1]
    null_rhos.append(float(spearmanr(delays, null_oof)[0]))
    if (i+1) % 50 == 0: print(f"  perm {i+1}/{N_PERM}", flush=True)

null_rhos = np.array(null_rhos)
# rho is negative (early=high prob, long delay=low prob); test one-sided <=
p = (null_rhos <= rho).mean()

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(null_rhos, bins=40, color="#95a5a6", edgecolor="white", label="Null (shuffled)")
ax.axvline(rho, color="#e74c3c", lw=2, label=f"Observed ρ={rho:.3f}")
ax.set_xlabel("Spearman ρ(delay, proba)"); ax.set_ylabel("Count")
ax.set_title(f"TabICL permutation test\nearly vs med+late  p={p:.3f}")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"cls_tabicl_permtest.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  Permutation p={p:.3f}", flush=True)

# ── save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "Track.ID": df["Track.ID"], "dataset": df["dataset"],
    "delay_b2r": delays.round(2), "y_true": y_cls, "tabicl_proba": oof.round(4),
}).to_csv(RES_DIR/"cls_tabicl_predictions.csv", index=False)

pd.DataFrame([{"method":"TabICL","AUC":round(auc,4),"sensitivity":round(sens,4),
               "specificity":round(spec,4),"bal_acc":round(bal,4),
               "spearman_rho":round(rho,4),"perm_p":round(p,4),
               "threshold":round(thresh,4),
               "n":len(y_cls),"n_early":n_early,"gmm_cutoff_min":round(cutoff1,0)}
             ]).to_csv(RES_DIR/"cls_tabicl_metrics.csv", index=False)
np.save(RES_DIR/"cls_tabicl_null_rhos.npy", null_rhos)

print(f"\nTabICL classification: AUC={auc:.3f}  ρ={rho:.3f}  p={p:.3f}")
print("Done.", flush=True)

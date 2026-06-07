"""
03c_classify_en_only.py
ElasticNet-only classification (early vs med+late, delay_blue_to_red).
Standalone — runs fast with N_PERM=200.
"""

import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import norm as sp_norm
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
FIG_DIR    = ANA_DIR / "figures"
SEED, N_PERM, N_FOLDS = 42, 200, 5

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
f16 = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
df = df.merge(f16, on="Track.ID", how="left")

NON_FEAT = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
            "green_onset_min","track_start_min","abs_gfp_onset_min",
            "movie_half_min","y","gfp_snr_mean","bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]
mask = np.isfinite(df["delay_blue_to_red"].values.astype(float))
df = df[mask].reset_index(drop=True)

delays = df["delay_blue_to_red"].values.astype(float)
X_raw  = df[feat_cols].values.astype(float)
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
idx_med   = np.where(cls_pred == 1)[0]
cutoff2   = x_grid[idx_med[-1]]   if len(idx_med)   > 0 else mu[1] + sig[1]

y_cls   = (delays <= cutoff1).astype(int)
n_early = y_cls.sum(); n_rest = (y_cls == 0).sum()
print(f"GMM cutoffs: {cutoff1:.0f} min | {cutoff2:.0f} min")
print(f"n={len(y_cls)}  early={n_early}  med+late={n_rest}")
print(f"{len(feat_cols)} features", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def youden_threshold(y_true, proba):
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr - fpr)])

def perm_fig(obs_rho, null_rhos, title, path):
    # one-sided <=: obs_rho is negative
    p = (null_rhos <= obs_rho).mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(null_rhos, bins=40, color="#95a5a6", edgecolor="white", label="Null (shuffled)")
    ax.axvline(obs_rho, color="#e74c3c", lw=2, label=f"Observed ρ={obs_rho:.3f}")
    ax.set_xlabel("Spearman ρ"); ax.set_ylabel("Count")
    ax.set_title(f"{title}\np={p:.3f}  (N={len(null_rhos)} perms)")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return p

outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
rng       = np.random.default_rng(SEED)

# ── ElasticNet CV ─────────────────────────────────────────────────────────────
print("\n═══ ElasticNet (LogisticRegressionCV) ═══", flush=True)
LR_KWARGS = dict(l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0], penalty="elasticnet",
                 solver="saga", max_iter=2000, random_state=SEED, n_jobs=-1)
oof_en = np.zeros(len(y_cls))
best_C, best_l1 = [], []

for fold, (tr, te) in enumerate(outer_skf.split(X_raw, y_cls)):
    sc = StandardScaler().fit(X_raw[tr])
    m  = LogisticRegressionCV(**{**LR_KWARGS, "cv": N_FOLDS})
    m.fit(sc.transform(X_raw[tr]), y_cls[tr])
    oof_en[te] = m.predict_proba(sc.transform(X_raw[te]))[:, 1]
    best_C.append(m.C_[0]); best_l1.append(m.l1_ratio_[0])
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_cls[te], oof_en[te]):.3f}", flush=True)

auc_en    = roc_auc_score(y_cls, oof_en)
thresh_en = youden_threshold(y_cls, oof_en)
pred_en   = (oof_en >= thresh_en).astype(int)
tp = int(((pred_en==1)&(y_cls==1)).sum()); fn = int(((pred_en==0)&(y_cls==1)).sum())
tn = int(((pred_en==0)&(y_cls==0)).sum()); fp = int(((pred_en==1)&(y_cls==0)).sum())
sens_en = tp/(tp+fn) if tp+fn>0 else 0
spec_en = tn/(tn+fp) if tn+fp>0 else 0
bal_en  = balanced_accuracy_score(y_cls, pred_en)
rho_en  = float(spearmanr(delays, oof_en)[0])
print(f"  ElasticNet: AUC={auc_en:.3f}  thresh={thresh_en:.3f}  Sens={sens_en:.3f}  "
      f"Spec={spec_en:.3f}  BalAcc={bal_en:.3f}  ρ={rho_en:.3f}", flush=True)

# ROC
fpr_en, tpr_en, _ = roc_curve(y_cls, oof_en)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_en, tpr_en, lw=2, color="#27ae60", label=f"ElasticNet  AUC={auc_en:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random")
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title(f"ElasticNet — early vs med+late (BFP→mCherry)\n(n={len(y_cls)}, {n_early} early)")
ax.legend(fontsize=9, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"cls_en_roc.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# permutation test — use fixed C/l1_ratio (mean of CV results) to avoid nested CV per perm
mean_C  = float(np.mean(best_C))
mean_l1 = float(np.mean(best_l1))
print(f"  Running permutation test ({N_PERM} perms, fixed C={mean_C:.4f} l1={mean_l1:.2f})...", flush=True)

null_en = []
for i in range(N_PERM):
    null_oof = np.zeros(len(y_cls))
    for tr, te in outer_skf.split(X_raw, y_cls):
        y_shuf = y_cls[tr].copy(); rng.shuffle(y_shuf)
        if y_shuf.sum() == 0 or y_shuf.sum() == len(y_shuf):
            continue
        sc = StandardScaler().fit(X_raw[tr])
        m  = LogisticRegression(C=mean_C, l1_ratio=mean_l1, penalty="elasticnet",
                                solver="saga", max_iter=2000, random_state=SEED, n_jobs=-1)
        m.fit(sc.transform(X_raw[tr]), y_shuf)
        null_oof[te] = m.predict_proba(sc.transform(X_raw[te]))[:, 1]
    null_en.append(float(spearmanr(delays, null_oof)[0]))
    if (i+1) % 50 == 0: print(f"  perm {i+1}/{N_PERM}", flush=True)
null_en = np.array(null_en)
p_en = perm_fig(rho_en, null_en, "ElasticNet permutation test\nearly vs med+late",
                FIG_DIR/"cls_en_permtest.png")
print(f"  Permutation p={p_en:.3f}", flush=True)

# ── save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "Track.ID": df["Track.ID"], "dataset": df["dataset"],
    "delay_b2r": delays.round(2), "y_true": y_cls, "en_proba": oof_en.round(4),
}).to_csv(RES_DIR/"cls_en_predictions.csv", index=False)

pd.DataFrame([
    {"method":"ElasticNet","AUC":round(auc_en,4),"sensitivity":round(sens_en,4),
     "specificity":round(spec_en,4),"bal_acc":round(bal_en,4),
     "spearman_rho":round(rho_en,4),"perm_p":round(p_en,4),
     "threshold":round(thresh_en,4),
     "n":len(y_cls),"n_early":n_early,"gmm_cutoff_min":round(cutoff1,0)},
]).to_csv(RES_DIR/"cls_en_metrics.csv", index=False)

np.save(RES_DIR/"cls_en_null_rhos.npy", null_en)

print(f"\n── Summary ──")
print(f"  ElasticNet: AUC={auc_en:.3f}  thresh={thresh_en:.3f}  "
      f"Sens={sens_en:.3f}  Spec={spec_en:.3f}  BalAcc={bal_en:.3f}  "
      f"ρ={rho_en:.3f}  p={p_en:.3f}")
print("\nDone.", flush=True)

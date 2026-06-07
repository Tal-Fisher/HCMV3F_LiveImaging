"""
03b_classify_xgb_only.py
XGBoost-only classification (early vs med+late, delay_blue_to_red).
Standalone version of the XGBoost section of 03_classify_en_xgb.py —
runs in parallel with EN permutation test. Saves to cls_xgb_metrics.csv.
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
from xgboost import XGBClassifier
from collections import Counter
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
FIG_DIR    = ANA_DIR / "figures"
SEED, N_PERM, N_FOLDS, N_TRIALS = 42, 500, 5, 50

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
idx_med   = np.where(cls_pred == 1)[0]
cutoff2   = x_grid[idx_med[-1]]   if len(idx_med)   > 0 else mu[1] + sig[1]

y_cls   = (delays <= cutoff1).astype(int)
n_early = y_cls.sum(); n_rest = (y_cls == 0).sum()
spw     = n_rest / n_early
print(f"GMM cutoffs: {cutoff1:.0f} min | {cutoff2:.0f} min")
print(f"n={len(y_cls)}  early={n_early}  med+late={n_rest}  scale_pos_weight={spw:.2f}")
print(f"{len(feat_cols)} features", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def youden_threshold(y_true, proba):
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr - fpr)])

def perm_fig(obs_rho, null_rhos, title, path):
    # obs_rho is negative (early=high prob, long delay=low prob); test one-sided <=
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

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST (Optuna)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n═══ XGBoost (Optuna {N_TRIALS} trials/fold) ═══", flush=True)

def suggest_params(trial):
    return dict(
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma            = trial.suggest_float("gamma", 0.0, 1.0),
    )

oof_xgb     = np.zeros(len(y_cls))
best_params = []

for fold, (tr, te) in enumerate(outer_skf.split(X_raw, y_cls)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", flush=True)
    sc    = StandardScaler().fit(X_raw[tr])
    Xtr_s = sc.transform(X_raw[tr]); Xte_s = sc.transform(X_raw[te])

    n_val    = max(10, len(tr) // 5)
    inner_tr = tr[:-n_val]; inner_va = tr[-n_val:]
    Xi_tr    = sc.transform(X_raw[inner_tr]); Xi_va = sc.transform(X_raw[inner_va])

    def obj(trial):
        p = suggest_params(trial)
        m = XGBClassifier(**p, n_estimators=600, early_stopping_rounds=40,
                          scale_pos_weight=spw, eval_metric="auc",
                          random_state=SEED, verbosity=0, n_jobs=4)
        m.fit(Xi_tr, y_cls[inner_tr], eval_set=[(Xi_va, y_cls[inner_va])], verbose=False)
        return roc_auc_score(y_cls[inner_va], m.predict_proba(Xi_va)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED+fold))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params; best_params.append(bp)

    final = XGBClassifier(**bp, n_estimators=600, early_stopping_rounds=40,
                          scale_pos_weight=spw, eval_metric="auc",
                          random_state=SEED, verbosity=0, n_jobs=4)
    final.fit(Xtr_s, y_cls[tr], eval_set=[(Xte_s, y_cls[te])], verbose=False)
    oof_xgb[te] = final.predict_proba(Xte_s)[:, 1]
    print(f"    AUC={roc_auc_score(y_cls[te], oof_xgb[te]):.3f}  best_params={bp}", flush=True)

auc_xgb  = roc_auc_score(y_cls, oof_xgb)
thresh_xgb = youden_threshold(y_cls, oof_xgb)
pred_xgb = (oof_xgb >= thresh_xgb).astype(int)
tp = int(((pred_xgb==1)&(y_cls==1)).sum()); fn = int(((pred_xgb==0)&(y_cls==1)).sum())
tn = int(((pred_xgb==0)&(y_cls==0)).sum()); fp = int(((pred_xgb==1)&(y_cls==0)).sum())
sens_xgb = tp/(tp+fn) if tp+fn>0 else 0
spec_xgb = tn/(tn+fp) if tn+fp>0 else 0
bal_xgb  = balanced_accuracy_score(y_cls, pred_xgb)
rho_xgb  = float(spearmanr(delays, oof_xgb)[0])
print(f"  XGBoost: AUC={auc_xgb:.3f}  thresh={thresh_xgb:.3f}  Sens={sens_xgb:.3f}  "
      f"Spec={spec_xgb:.3f}  BalAcc={bal_xgb:.3f}  ρ={rho_xgb:.3f}", flush=True)

# ROC figure
fpr_xgb, tpr_xgb, _ = roc_curve(y_cls, oof_xgb)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_xgb, tpr_xgb, lw=2, color="#8e44ad", label=f"XGBoost  AUC={auc_xgb:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random")
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title(f"XGBoost — early vs med+late (BFP→mCherry)\n(n={len(y_cls)}, {n_early} early)")
ax.legend(fontsize=9, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIG_DIR/"cls_xgb_roc.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# permutation test
print("  Running permutation test...", flush=True)
avg_bp = {k: Counter(str(d[k]) for d in best_params).most_common(1)[0][0]
          for k in best_params[0]}
avg_bp = {k: (int(v) if v.isdigit() else float(v)) for k, v in avg_bp.items()}

null_xgb = []
for i in range(N_PERM):
    null_oof = np.zeros(len(y_cls))
    for tr, te in outer_skf.split(X_raw, y_cls):
        y_shuf = y_cls[tr].copy(); rng.shuffle(y_shuf)
        if y_shuf.sum() == 0 or y_shuf.sum() == len(y_shuf):
            continue
        sc    = StandardScaler().fit(X_raw[tr])
        spw_p = float((y_shuf==0).sum()) / max(1, y_shuf.sum())
        m = XGBClassifier(**avg_bp, n_estimators=100, scale_pos_weight=spw_p,
                          eval_metric="auc", random_state=SEED, verbosity=0, n_jobs=4)
        m.fit(sc.transform(X_raw[tr]), y_shuf)
        null_oof[te] = m.predict_proba(sc.transform(X_raw[te]))[:, 1]
    null_xgb.append(float(spearmanr(delays, null_oof)[0]))
    if (i+1) % 100 == 0: print(f"  perm {i+1}/{N_PERM}", flush=True)
null_xgb = np.array(null_xgb)
p_xgb = perm_fig(rho_xgb, null_xgb, "XGBoost permutation test\nearly vs med+late",
                 FIG_DIR/"cls_xgb_permtest.png")
print(f"  Permutation p={p_xgb:.3f}", flush=True)

# ── save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "Track.ID": df["Track.ID"], "dataset": df["dataset"],
    "delay_b2r": delays.round(2), "y_true": y_cls, "xgb_proba": oof_xgb.round(4),
}).to_csv(RES_DIR/"cls_xgb_predictions.csv", index=False)

pd.DataFrame([
    {"method":"XGBoost","AUC":round(auc_xgb,4),"sensitivity":round(sens_xgb,4),
     "specificity":round(spec_xgb,4),"bal_acc":round(bal_xgb,4),
     "spearman_rho":round(rho_xgb,4),"perm_p":round(p_xgb,4),
     "threshold":round(thresh_xgb,4),
     "n":len(y_cls),"n_early":n_early,"gmm_cutoff_min":round(cutoff1,0)},
]).to_csv(RES_DIR/"cls_xgb_metrics.csv", index=False)

pd.DataFrame(best_params).to_csv(RES_DIR/"cls_xgb_best_params.csv", index=False)
np.save(RES_DIR/"cls_xgb_null_rhos.npy", null_xgb)

print(f"\n── Summary ──")
print(f"  XGBoost: AUC={auc_xgb:.3f}  thresh={thresh_xgb:.3f}  "
      f"Sens={sens_xgb:.3f}  Spec={spec_xgb:.3f}  BalAcc={bal_xgb:.3f}  "
      f"ρ={rho_xgb:.3f}  p={p_xgb:.3f}")
print(f"  GMM cutoff early/med: {cutoff1:.0f} min | med/late: {cutoff2:.0f} min")
print("\nDone.", flush=True)

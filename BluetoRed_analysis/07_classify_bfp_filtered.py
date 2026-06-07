"""
07_classify_bfp_filtered.py
Re-run EN + XGBoost + TabICL classification with late-bloomers filter.
Removes cells with BFP onset >10 frames after GFP onset (delay_green_to_blue > 10 frames),
suspected tracking artifacts (57/525 productive cells, 10.9%).
Reuses XGBoost best params from the unfiltered run (saves ~20 min of Optuna).
Saves to results_bfp_filtered/ and produces a before/after confusion matrix figure.

Run with TabICL env:
  /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12 07_classify_bfp_filtered.py
"""

import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, norm as sp_norm
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                             balanced_accuracy_score, ConfusionMatrixDisplay,
                             confusion_matrix)
from xgboost import XGBClassifier
from tabicl import TabICLClassifier
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results_bfp_filtered"
FIG_DIR    = ANA_DIR / "figures"
RES_DIR.mkdir(exist_ok=True)
SEED, N_FOLDS, N_EST = 42, 5, 8

# ── data + BFP half-movie filter ──────────────────────────────────────────────
df = pd.read_csv(EXPORT_DIR / "model_df.csv")
f16 = pd.read_csv(EXPORT_DIR / "frame16_features.csv")
df = df.merge(f16, on="Track.ID", how="left")

NON_FEAT = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
            "green_onset_min","track_start_min","abs_gfp_onset_min",
            "movie_half_min","y","gfp_snr_mean","bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

df["delay_blue_to_red"] = df["delay_green_to_red"] - df["delay_green_to_blue"]

# productive-only + late-bloomers filter (BFP onset <= 10 frames after GFP)
FRAME_MIN = 15.667  # minutes per frame for A2/A3
MAX_G2B_FRAMES = 10
mask = (np.isfinite(df["delay_blue_to_red"].values.astype(float)) &
        (df["delay_green_to_blue"].values <= MAX_G2B_FRAMES * FRAME_MIN))
df = df[mask].reset_index(drop=True)

delays = df["delay_blue_to_red"].values.astype(float)
print(f"After late-bloomers filter (BFP onset ≤{MAX_G2B_FRAMES} frames after GFP): "
      f"{len(df)} cells (removed {525 - len(df)} late-bloomer cells)")

X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

# ── GMM cutoff (refit on filtered data) ───────────────────────────────────────
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
n_early = int(y_cls.sum()); n_rest = int((y_cls==0).sum())
spw     = n_rest / n_early
print(f"GMM cutoff: {cutoff1:.0f} min  |  early={n_early}  med+late={n_rest}  spw={spw:.2f}")
print(f"{len(feat_cols)} features", flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def clf_metrics(y_true, proba, label):
    auc  = roc_auc_score(y_true, proba)
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum()); fn = int(((pred==0)&(y_true==1)).sum())
    tn = int(((pred==0)&(y_true==0)).sum()); fp = int(((pred==1)&(y_true==0)).sum())
    sens = tp/(tp+fn) if tp+fn>0 else 0
    spec = tn/(tn+fp) if tn+fp>0 else 0
    bal  = balanced_accuracy_score(y_true, pred)
    rho  = float(spearmanr(delays, proba)[0])
    print(f"  {label}: AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  "
          f"BalAcc={bal:.3f}  ρ={rho:.3f}")
    return dict(auc=auc, sens=sens, spec=spec, bal_acc=bal, rho=rho,
                tp=tp, fn=fn, tn=tn, fp=fp)

# ══════════════════════════════════════════════════════════════════════════════
# ELASTIC NET
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ ElasticNet ═══", flush=True)
LR_KWARGS = dict(penalty="elasticnet", solver="saga",
                 l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                 Cs=np.logspace(-3, 1, 20),
                 cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED+1),
                 class_weight="balanced", scoring="roc_auc",
                 max_iter=2000, random_state=SEED, n_jobs=-1)

oof_en = np.zeros(len(y_cls))
for fold, (tr, te) in enumerate(skf.split(X_raw, y_cls)):
    sc = StandardScaler().fit(X_raw[tr])
    m  = LogisticRegressionCV(**LR_KWARGS)
    m.fit(sc.transform(X_raw[tr]), y_cls[tr])
    oof_en[te] = m.predict_proba(sc.transform(X_raw[te]))[:, 1]
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_cls[te], oof_en[te]):.3f}", flush=True)

met_en = clf_metrics(y_cls, oof_en, "ElasticNet")

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST — reuse best params from unfiltered run
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ XGBoost (reusing best params from unfiltered run) ═══", flush=True)
bp_df = pd.read_csv(ANA_DIR / "results" / "cls_xgb_best_params.csv")
avg_bp = {
    "max_depth":        int(round(bp_df["max_depth"].mean())),
    "learning_rate":    float(bp_df["learning_rate"].mean()),
    "subsample":        float(bp_df["subsample"].mean()),
    "colsample_bytree": float(bp_df["colsample_bytree"].mean()),
    "min_child_weight": int(round(bp_df["min_child_weight"].mean())),
    "reg_alpha":        float(bp_df["reg_alpha"].mean()),
    "reg_lambda":       float(bp_df["reg_lambda"].mean()),
    "gamma":            float(bp_df["gamma"].mean()),
}
print(f"  avg params: {avg_bp}")

oof_xgb = np.zeros(len(y_cls))
for fold, (tr, te) in enumerate(skf.split(X_raw, y_cls)):
    sc = StandardScaler().fit(X_raw[tr])
    m  = XGBClassifier(**avg_bp, n_estimators=300,
                       scale_pos_weight=spw, eval_metric="auc",
                       random_state=SEED, verbosity=0, n_jobs=4)
    m.fit(sc.transform(X_raw[tr]), y_cls[tr])
    oof_xgb[te] = m.predict_proba(sc.transform(X_raw[te]))[:, 1]
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_cls[te], oof_xgb[te]):.3f}", flush=True)

met_xgb = clf_metrics(y_cls, oof_xgb, "XGBoost")

# ══════════════════════════════════════════════════════════════════════════════
# TABICL
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TabICL ═══", flush=True)
oof_tab = np.zeros(len(y_cls))
for fold, (tr, te) in enumerate(skf.split(X_raw, y_cls)):
    sc  = StandardScaler().fit(X_raw[tr])
    clf = TabICLClassifier(n_estimators=N_EST, random_state=SEED)
    clf.fit(sc.transform(X_raw[tr]), y_cls[tr])
    oof_tab[te] = clf.predict_proba(sc.transform(X_raw[te]))[:, 1]
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_cls[te], oof_tab[te]):.3f}", flush=True)

met_tab = clf_metrics(y_cls, oof_tab, "TabICL")

# ── save predictions + metrics ────────────────────────────────────────────────
pd.DataFrame({
    "Track.ID": df["Track.ID"], "dataset": df["dataset"],
    "delay_b2r": delays.round(2), "y_true": y_cls,
    "en_proba": oof_en.round(4), "xgb_proba": oof_xgb.round(4),
    "tabicl_proba": oof_tab.round(4),
}).to_csv(RES_DIR / "cls_predictions.csv", index=False)

rows = []
for name, m in [("ElasticNet", met_en), ("XGBoost", met_xgb), ("TabICL", met_tab)]:
    rows.append({"method": name, "n": len(y_cls), "n_early": n_early,
                 "gmm_cutoff_min": round(cutoff1, 0), **{k: round(v, 4) for k, v in m.items()
                 if k not in ("tp","fn","tn","fp")}})
pd.DataFrame(rows).to_csv(RES_DIR / "cls_metrics.csv", index=False)
print(f"\nSaved results to {RES_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — side-by-side confusion matrices before vs after filter
# ══════════════════════════════════════════════════════════════════════════════
prev_en  = pd.read_csv(ANA_DIR / "results" / "cls_en_xgb_predictions.csv")
prev_tab = pd.read_csv(ANA_DIR / "results" / "cls_tabicl_predictions.csv")
prev     = prev_en.merge(prev_tab[["Track.ID","tabicl_proba"]], on="Track.ID")

MODELS = [
    ("ElasticNet", prev["en_proba"].values,  oof_en,  "#27ae60"),
    ("XGBoost",    prev["xgb_proba"].values, oof_xgb, "#8e44ad"),
    ("TabICL",     prev["tabicl_proba"].values, oof_tab, "#e67e22"),
]
CLASS_LABELS = ["Med+Late", "Early"]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
fig.suptitle(
    "Confusion matrices: BFP→mCherry early vs med+late\n"
    f"Top: unfiltered (n=525)  |  Bottom: late-bloomers filter ≤{MAX_G2B_FRAMES} frames (n={len(df)})",
    fontsize=12, fontweight="bold"
)

for col, (name, proba_before, proba_after, color) in enumerate(MODELS):
    for row, (y_true_r, proba, n_cells) in enumerate([
        (prev["y_true"].values, proba_before, len(prev)),
        (y_cls,                 proba_after,  len(y_cls)),
    ]):
        ax = axes[row, col]
        y_pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true_r, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax.text(j, i + 0.28, f"({100*cm[i,j]/total:.1f}%)",
                        ha="center", va="center", fontsize=9, color="black")

        tp = cm[1,1]; fn = cm[1,0]; tn = cm[0,0]; fp = cm[0,1]
        sens = tp/(tp+fn) if tp+fn>0 else 0
        spec = tn/(tn+fp) if tn+fp>0 else 0
        bal  = (sens+spec)/2
        auc  = roc_auc_score(y_true_r, proba)

        label = ("Unfiltered" if row == 0 else "Late-bloomers filtered")
        ax.set_title(
            f"{name} — {label} (n={n_cells})\n"
            f"AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  BalAcc={bal:.3f}",
            fontsize=9, fontweight="bold", color=color
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(1.5)

plt.tight_layout()
out = FIG_DIR / "confusion_matrices_bfp_filter_comparison.png"
fig.savefig(str(out), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ── print summary comparison ──────────────────────────────────────────────────
print("\n── Metric comparison (unfiltered → BFP-filtered) ──")
prev_metrics = pd.read_csv(ANA_DIR / "results" / "cls_en_xgb_metrics.csv")
prev_tab_m   = pd.read_csv(ANA_DIR / "results" / "cls_tabicl_metrics.csv")
prev_all     = pd.concat([prev_metrics, prev_tab_m], ignore_index=True)

new_metrics  = pd.read_csv(RES_DIR / "cls_metrics.csv")

for name in ["ElasticNet", "XGBoost", "TabICL"]:
    old = prev_all[prev_all["method"] == name].iloc[0]
    new = new_metrics[new_metrics["method"] == name].iloc[0]
    print(f"  {name:<12}  AUC: {old['AUC']:.3f} → {new['auc']:.3f}  "
          f"BalAcc: {old['bal_acc']:.3f} → {new['bal_acc']:.3f}  "
          f"Sens: {old['sensitivity']:.3f} → {new['sens']:.3f}")

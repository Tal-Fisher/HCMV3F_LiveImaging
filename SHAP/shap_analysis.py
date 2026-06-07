"""
shap_analysis.py — SHAP explanations for XGBoost (regression + binary)

SHAP values are computed fold-by-fold on held-out test cells using the exact
hyperparameters saved during the original CV run. This means every cell's SHAP
values come from a model that never saw that cell during training — the gold
standard for out-of-sample explanation.

Outputs (all in SHAP/):
  Regression:
    shap_beeswarm_regression.png   — global feature importance + direction
    shap_dependence_regression.png — SHAP value vs feature value for top 6 features
    shap_waterfall_regression.png  — per-cell breakdown for 3 selected cells
  Binary:
    shap_beeswarm_binary.png
    shap_dependence_binary.png
    shap_waterfall_binary.png
  methodology.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr
from pathlib import Path

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
RES_DIR = BASE / "results" / "xgboost"
OUT_DIR = BASE / "SHAP"
SEED    = 42
N_FOLDS = 3

# ── preprocessing (identical to 11_xgboost.py) ────────────────────────────────
df = pd.read_csv(BASE / "cache" / "python_export" / "model_df.csv")
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)

NON_FEAT = {"Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
            "green_onset_min", "track_start_min", "abs_gfp_onset_min",
            "movie_half_min", "y", "gfp_snr_mean", "bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]
print(f"Cells: {len(df)}  Features: {len(feat_cols)}", flush=True)

X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

scaler = StandardScaler()
X_all  = scaler.fit_transform(X_raw)

delay = df["delay_green_to_red"].values.astype(float)
y_bin = np.isfinite(delay).astype(int)
prod_mask = np.isfinite(delay)

# ── load saved hyperparameters ────────────────────────────────────────────────
params_reg = pd.read_csv(RES_DIR / "best_params_regression.csv")
params_bin = pd.read_csv(RES_DIR / "best_params_binary.csv")
FIXED = {"random_state": SEED, "verbosity": 0, "n_jobs": 1}

def row_to_params(row):
    skip = {"fold", "random_state", "verbosity", "n_jobs", "n_estimators"}
    p = {k: v for k, v in row.items() if k not in skip}
    p["max_depth"]        = int(p["max_depth"])
    p["min_child_weight"] = int(p["min_child_weight"])
    p.update(FIXED)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# CV + SHAP computation
# ══════════════════════════════════════════════════════════════════════════════
def run_shap_cv(task):
    is_reg = (task == "regression")
    if is_reg:
        X, y  = X_all[prod_mask], delay[prod_mask]
        pdf   = params_reg
        tertile = pd.qcut(y, q=3, labels=False)
        skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, tertile))
        X_orig = X_raw[prod_mask]
    else:
        X, y  = X_all, y_bin
        pdf   = params_bin
        skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y))
        X_orig = X_raw

    n, nf = len(X), len(feat_cols)
    shap_vals   = np.full((n, nf), np.nan)
    base_vals   = np.full(n, np.nan)
    predictions = np.full(n, np.nan)

    for fold, (tr, te) in enumerate(splits):
        print(f"  [{task}] fold {fold+1}/{N_FOLDS}...", flush=True)
        row   = pdf[pdf["fold"] == fold + 1].iloc[0]
        p     = row_to_params(row)
        n_est = int(row["n_estimators"])

        if is_reg:
            model = XGBRegressor(**p, n_estimators=n_est, eval_metric="rmse")
            model.fit(X[tr], y[tr], verbose=False)
            predictions[te] = model.predict(X[te])
        else:
            spw   = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            model = XGBClassifier(**p, n_estimators=n_est,
                                  scale_pos_weight=spw, eval_metric="auc")
            model.fit(X[tr], y[tr], verbose=False)
            predictions[te] = model.predict_proba(X[te])[:, 1]

        explainer        = shap.TreeExplainer(model)
        sv               = explainer.shap_values(X[te])
        shap_vals[te]    = sv
        bv               = explainer.expected_value
        base_vals[te]    = float(bv[1] if hasattr(bv, '__len__') else bv)
        print(f"    done — base={base_vals[te[0]]:.2f}", flush=True)

    return dict(shap_vals=shap_vals, base_vals=base_vals,
                predictions=predictions, X_orig=X_orig, y=y)


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════
def save_beeswarm(sv, X_orig, feat_names, title, path):
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X_orig, feature_names=feat_names,
                      plot_type="dot", max_display=20, show=False)
    plt.title(title, fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}", flush=True)


def save_dependence(sv, X_orig, feat_names, title_prefix, path, n_top=6):
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_top]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{title_prefix} — SHAP dependence plots (top {n_top} features)",
                 fontsize=11, fontweight="bold")

    for ax, fi in zip(axes.flat, top_idx):
        x_vals = X_orig[:, fi]
        s_vals = sv[:, fi]
        # colour by feature with strongest interaction
        corrs = [abs(np.corrcoef(s_vals, X_orig[:, j])[0, 1])
                 if j != fi else 0 for j in range(len(feat_names))]
        ci    = int(np.argmax(corrs))
        sc    = ax.scatter(x_vals, s_vals, c=X_orig[:, ci],
                           cmap="coolwarm", alpha=0.55, s=12)
        plt.colorbar(sc, ax=ax, label=feat_names[ci], pad=0.02)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(feat_names[fi], fontsize=8)
        ax.set_ylabel("SHAP value", fontsize=8)
        ax.set_title(feat_names[fi], fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}", flush=True)


def save_waterfall(sv, base_vals, X_orig, feat_names,
                   cell_indices, cell_labels, title_prefix, path):
    mean_abs = np.abs(sv).mean(axis=0)
    top15    = np.argsort(mean_abs)[::-1][:15]

    fig, axes = plt.subplots(1, len(cell_indices),
                             figsize=(5 * len(cell_indices), 7))
    if len(cell_indices) == 1:
        axes = [axes]
    fig.suptitle(f"{title_prefix} — SHAP waterfall (individual cells)",
                 fontsize=11, fontweight="bold")

    for ax, ci, label in zip(axes, cell_indices, cell_labels):
        s  = sv[ci, top15]
        fv = X_orig[ci, top15]
        fn = [feat_names[i] for i in top15]
        bv = float(base_vals[ci])

        # sort by |SHAP| ascending so largest bar is at top
        order  = np.argsort(np.abs(s))
        s, fv, fn = s[order], fv[order], [fn[i] for i in order]

        cumulative = bv + np.concatenate([[0], np.cumsum(s)])
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in s]
        y_pos  = np.arange(len(s))

        for i, (start, end, col) in enumerate(
                zip(cumulative[:-1], cumulative[1:], colors)):
            ax.barh(i, end - start, left=start, color=col, alpha=0.82, height=0.6)

        pred = bv + s.sum()
        ax.axvline(bv,   color="gray",  lw=1.2, linestyle="--", alpha=0.7,
                   label=f"Base = {bv:.1f}")
        ax.axvline(pred, color="black", lw=1.5,
                   label=f"Pred = {pred:.1f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{n}  [{v:.3g}]" for n, v in zip(fn, fv)], fontsize=7)
        ax.set_xlabel("Model output", fontsize=8)
        ax.set_title(label, fontsize=8.5, fontweight="bold")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Regression ──", flush=True)
reg = run_shap_cv("regression")

r2_  = r2_score(reg["y"], reg["predictions"])
rho_ = spearmanr(reg["y"], reg["predictions"])[0]
print(f"  Reproduced: R²={r2_:.3f}  ρ={rho_:.3f}", flush=True)

save_beeswarm(
    reg["shap_vals"], reg["X_orig"], feat_cols,
    f"XGBoost Regression — SHAP beeswarm\n"
    f"(n={len(reg['y'])} productive cells, 3-fold CV out-of-sample, "
    f"R²={r2_:.3f}, ρ={rho_:.3f})",
    OUT_DIR / "shap_beeswarm_regression.png"
)
save_dependence(
    reg["shap_vals"], reg["X_orig"], feat_cols,
    "Regression", OUT_DIR / "shap_dependence_regression.png"
)

y_r, p_r = reg["y"], reg["predictions"]
save_waterfall(
    reg["shap_vals"], reg["base_vals"], reg["X_orig"], feat_cols,
    [int(np.argmin(y_r)), int(np.argmax(y_r)), int(np.argmax(np.abs(y_r - p_r)))],
    [f"Shortest delay\ntrue={y_r[np.argmin(y_r)]/60:.1f}h  pred={p_r[np.argmin(y_r)]/60:.1f}h",
     f"Longest delay\ntrue={y_r[np.argmax(y_r)]/60:.1f}h  pred={p_r[np.argmax(y_r)]/60:.1f}h",
     f"Worst prediction\ntrue={y_r[np.argmax(np.abs(y_r-p_r))]/60:.1f}h  "
     f"pred={p_r[np.argmax(np.abs(y_r-p_r))]/60:.1f}h"],
    "Regression", OUT_DIR / "shap_waterfall_regression.png"
)

# ══════════════════════════════════════════════════════════════════════════════
# BINARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Binary ──", flush=True)
bn = run_shap_cv("binary")

auc_ = roc_auc_score(bn["y"], bn["predictions"])
print(f"  Reproduced: AUC={auc_:.3f}", flush=True)

save_beeswarm(
    bn["shap_vals"], bn["X_orig"], feat_cols,
    f"XGBoost Binary — SHAP beeswarm\n"
    f"(n={len(bn['y'])} cells, 3-fold CV out-of-sample, AUC={auc_:.3f})",
    OUT_DIR / "shap_beeswarm_binary.png"
)
save_dependence(
    bn["shap_vals"], bn["X_orig"], feat_cols,
    "Binary", OUT_DIR / "shap_dependence_binary.png"
)

y_b, p_b = bn["y"], bn["predictions"]
tp = np.where(y_b == 1)[0]
tn = np.where(y_b == 0)[0]
c_tp = int(tp[np.argmax(p_b[tp])])
c_tn = int(tn[np.argmin(p_b[tn])])
c_fp = int(tn[np.argmax(p_b[tn])])
save_waterfall(
    bn["shap_vals"], bn["base_vals"], bn["X_orig"], feat_cols,
    [c_tp, c_tn, c_fp],
    [f"Most confident TP\n(true=prod, P={p_b[c_tp]:.2f})",
     f"Most confident TN\n(true=non-prod, P={p_b[c_tn]:.2f})",
     f"Biggest false positive\n(true=non-prod, P={p_b[c_fp]:.2f})"],
    "Binary", OUT_DIR / "shap_waterfall_binary.png"
)

# ══════════════════════════════════════════════════════════════════════════════
# Methodology text
# ══════════════════════════════════════════════════════════════════════════════
text = f"""SHAP Analysis — HCMV Live Imaging XGBoost Models
=================================================
Generated by: SHAP/shap_analysis.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. WHAT IS SHAP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHAP (SHapley Additive exPlanations) decomposes each model prediction into
additive contributions from every feature, grounded in cooperative game theory.
For a given cell, SHAP answers: "How much did each feature push the prediction
above or below the baseline (average prediction across all training cells)?"

For tree-based models (XGBoost), shap.TreeExplainer computes exact SHAP values
in polynomial time — no approximations, no sampling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. HOW SHAP VALUES WERE COMPUTED (FOLD-BY-FOLD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To ensure SHAP values reflect out-of-sample explanations:

  For each of the 3 CV folds:
    1. Reload the exact hyperparameters saved from the original CV run
       (results/xgboost/best_params_regression.csv / best_params_binary.csv)
    2. Retrain the XGBoost model on the training fold
    3. Apply shap.TreeExplainer to the trained model
    4. Compute SHAP values ONLY on the held-out test fold cells

  Every cell's SHAP values come from a model that never saw that cell during
  training — identical to how predictions were evaluated in the CV.
  Final SHAP arrays concatenate all held-out fold SHAP values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. DATA AND FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dataset:      A2 + A3, first-half filter (abs_gfp_onset_min ≤ movie_half_min)
  Total cells:  809 (after filter)
  Features:     29 cell-intrinsic features from 16-frame window at GFP onset
  Preprocessing: median imputation → StandardScaler z-normalisation
  Excluded:     Track.ID, dataset, delay columns, timing columns,
                gfp_snr_mean, bf_snr_mean (100% missing)

  Regression: productive cells only (n=497, finite delay_green_to_red)
  Binary:     all 809 cells (productive=1, non-productive=0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. MODEL PERFORMANCE (REPRODUCED IN THIS SCRIPT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Regression: R²={r2_:.3f}   Spearman ρ={rho_:.3f}
  Binary:     AUC={auc_:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. XGBOOST HYPERPARAMETERS (PER FOLD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CV:    StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
  Search: Optuna TPESampler, 50 trials/fold (inner 80/20 split)
  n_estimators: early stopping on inner split, scaled for full training fold

  Regression parameters (one row per fold):
{params_reg.to_string(index=False)}

  Binary parameters (one row per fold):
{params_bin.to_string(index=False)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. PLOTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Beeswarm (shap_beeswarm_*.png):
    Each dot = one cell. Y-axis = feature ranked by mean |SHAP|.
    X-axis = SHAP value (right = pushes prediction UP, left = DOWN).
    Color = original (unscaled) feature value (red=high, blue=low).

  Dependence (shap_dependence_*.png):
    Top 6 features by mean |SHAP|. X-axis = raw feature value.
    Y-axis = SHAP value. Color = the feature with the strongest
    correlation with that feature's SHAP values (proxy for interaction).

  Waterfall (shap_waterfall_*.png):
    Decomposes predictions for 3 individual cells into per-feature contributions.
    Each bar = SHAP value for one feature (red=positive, blue=negative).
    Dashed vertical line = base value (mean prediction on training fold).
    Solid vertical line = final prediction (base + sum of SHAP values).
    Feature labels show the raw (unscaled) feature value in brackets.
    Regression: shortest delay, longest delay, worst-predicted cell.
    Binary: most confident TP, most confident TN, biggest false positive.
"""

(OUT_DIR / "methodology.txt").write_text(text)
print("\nSaved methodology.txt", flush=True)
print("All done.", flush=True)

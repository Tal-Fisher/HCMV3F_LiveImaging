"""
01_regression_en_xgb.py
ElasticNet + XGBoost regression for delay_blue_to_red (BFP→mCherry).
A2+A3, 45-feature extended set, productive-only filter (no half-movie filter).
Includes permutation test (500 perms, train-label shuffle).
"""

import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"
EXT_DIR    = BASE / "results" / "elasticnet_extended2"
XGB_DIR    = BASE / "results" / "xgboost"
ANA_DIR    = Path("/home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis")
RES_DIR    = ANA_DIR / "results"
FIG_DIR    = ANA_DIR / "figures"
SEED, N_PERM, N_TRIALS, N_FOLDS_XGB = 42, 500, 50, 5

# ── data — 45-feature extended set ────────────────────────────────────────────
ext_df = pd.read_csv(EXT_DIR / "model_df_extended2.csv")
# bring in half-movie filter columns from model_df.csv
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

print(f"n={len(y)} cells | {len(feat_cols)} features | "
      f"delay_b2r: {y.min():.0f}–{y.max():.0f} min (median {np.median(y):.0f})")

cat_colors = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}

# ── permutation test helper ────────────────────────────────────────────────────
def permutation_test(X, y, strata, cv_fn, n_perm, seed):
    rng = np.random.default_rng(seed)
    null_rhos = []
    for i in range(n_perm):
        null_preds = np.zeros(len(y))
        for tr, te in cv_fn(X, y, strata):
            y_shuf = y[tr].copy(); rng.shuffle(y_shuf)
            sc = StandardScaler().fit(X[tr])
            null_preds[te] = _fit_predict(sc.transform(X[tr]), y_shuf,
                                          sc.transform(X[te]))
        null_rhos.append(float(spearmanr(y, null_preds)[0]))
        if (i+1) % 100 == 0:
            print(f"  perm {i+1}/{n_perm}", flush=True)
    return np.array(null_rhos)

# placeholder — replaced per model
_fit_predict = None

def perm_figure(obs_rho, null_rhos, title, path):
    p = (null_rhos >= obs_rho).mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(null_rhos, bins=40, color="#95a5a6", edgecolor="white", label="Null (shuffled)")
    ax.axvline(obs_rho, color="#e74c3c", lw=2, label=f"Observed ρ={obs_rho:.3f}")
    ax.set_xlabel("Spearman ρ"); ax.set_ylabel("Count")
    ax.set_title(f"{title}\np={p:.3f}  (N={len(null_rhos)} perms)")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return p

def scatter_fig(y_true, y_pred, strata, title, path):
    r2 = r2_score(y_true, y_pred)
    r  = float(pearsonr(y_true, y_pred)[0])
    rho = float(spearmanr(y_true, y_pred)[0])
    fig, ax = plt.subplots(figsize=(5, 5))
    for cat in ["early","medium","late"]:
        m = strata == cat
        ax.scatter(y_true[m]/60, y_pred[m]/60, color=cat_colors[cat],
                   alpha=0.5, s=16, label=cat, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())/60*0.92
    hi = max(y_true.max(), y_pred.max())/60*1.06
    ax.plot([lo,hi],[lo,hi],"k--",lw=0.8,alpha=0.5)
    ax.set_xlabel("Actual delay BFP→mCherry (h)")
    ax.set_ylabel("CV predicted (h)")
    ax.set_title(f"{title}\nR²={r2:.3f}  r={r:.3f}  ρ={rho:.3f}  (n={len(y_true)})")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return r2, r, rho

# ══════════════════════════════════════════════════════════════════════════════
# ELASTIC NET
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ ElasticNet ═══", flush=True)
outer_kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
oof_en = np.zeros(len(y))
y_mean, y_std = y.mean(), y.std()
y_s = (y - y_mean) / y_std

for fold, (tr, te) in enumerate(outer_kf.split(X_raw)):
    sc = StandardScaler().fit(X_raw[tr])
    en = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=SEED,
                      max_iter=2000, n_alphas=100, tol=1e-4, n_jobs=-1)
    en.fit(sc.transform(X_raw[tr]), y_s[tr])
    oof_en[te] = en.predict(sc.transform(X_raw[te])) * y_std + y_mean

r2_en, r_en, rho_en = scatter_fig(y, oof_en, strata,
    "ElasticNet — delay BFP→mCherry",
    FIG_DIR/"reg_en_scatter.png")
print(f"  CV: R²={r2_en:.3f}  r={r_en:.3f}  ρ={rho_en:.3f}", flush=True)

# permutation test
print("  Running permutation test...", flush=True)
def en_cv_splits(X, y, strata):
    for tr, te in KFold(n_splits=10, shuffle=True, random_state=SEED).split(X):
        yield tr, te

def _fit_predict_en(Xtr, ytr, Xte):
    en = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=SEED,
                      max_iter=2000, n_alphas=100, tol=1e-4, n_jobs=1)
    ys, ym = ytr.mean(), ytr.std() or 1.0
    en.fit(Xtr, (ytr-ys)/ym)
    return en.predict(Xte)*ym + ys

_fit_predict = _fit_predict_en
null_en = permutation_test(X_raw, y, strata, en_cv_splits, N_PERM, SEED)
p_en = perm_figure(rho_en, null_en, "ElasticNet permutation test\ndelay BFP→mCherry",
                   FIG_DIR/"reg_en_permtest.png")
print(f"  Permutation p={p_en:.3f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST (Optuna)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n═══ XGBoost (Optuna {N_TRIALS} trials/fold, {N_FOLDS_XGB}-fold) ═══", flush=True)

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

outer_skf   = StratifiedKFold(n_splits=N_FOLDS_XGB, shuffle=True, random_state=SEED)
oof_xgb     = np.zeros(len(y))
imp_folds   = []
best_params = []

for fold, (tr, te) in enumerate(outer_skf.split(X_raw, strata)):
    print(f"  Fold {fold+1}/{N_FOLDS_XGB}...", flush=True)
    sc    = StandardScaler().fit(X_raw[tr])
    Xtr_s = sc.transform(X_raw[tr]); Xte_s = sc.transform(X_raw[te])

    n_val    = max(10, len(tr) // 5)
    inner_tr = tr[:-n_val]; inner_va = tr[-n_val:]
    Xi_tr    = sc.transform(X_raw[inner_tr]); Xi_va = sc.transform(X_raw[inner_va])

    def obj(trial):
        p = suggest_params(trial)
        m = XGBRegressor(**p, n_estimators=600, early_stopping_rounds=40,
                         eval_metric="rmse", random_state=SEED, verbosity=0, n_jobs=4)
        m.fit(Xi_tr, y[inner_tr], eval_set=[(Xi_va, y[inner_va])], verbose=False)
        return float(spearmanr(y[inner_va], m.predict(Xi_va))[0])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED+fold))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params; best_params.append(bp)

    final = XGBRegressor(**bp, n_estimators=600, early_stopping_rounds=40,
                         eval_metric="rmse", random_state=SEED, verbosity=0, n_jobs=4)
    final.fit(Xtr_s, y[tr], eval_set=[(Xte_s, y[te])], verbose=False)
    oof_xgb[te] = final.predict(Xte_s)
    imp_folds.append(final.feature_importances_)
    print(f"    r={float(pearsonr(y[te], oof_xgb[te])[0]):.3f}  best_params={bp}", flush=True)

# average best params (used for permutation test)
avg_p = {}
for k in best_params[0]:
    vals = [d[k] for d in best_params]
    avg_p[k] = int(round(sum(vals)/len(vals))) if isinstance(vals[0], int) else sum(vals)/len(vals)
avg_p.update({"random_state": SEED, "verbosity": 0, "n_jobs": 4})
print(f"  Averaged Optuna params: {avg_p}", flush=True)

# save per-fold best params
bp_rows = [{**d, "fold": i+1} for i, d in enumerate(best_params)]
pd.DataFrame(bp_rows).to_csv(RES_DIR/"best_params_b2r_regression.csv", index=False)

r2_xgb, r_xgb, rho_xgb = scatter_fig(y, oof_xgb, strata,
    "XGBoost (Optuna) — delay BFP→mCherry",
    FIG_DIR/"reg_xgb_scatter.png")
print(f"  CV: R²={r2_xgb:.3f}  r={r_xgb:.3f}  ρ={rho_xgb:.3f}", flush=True)

# feature importance
mean_imp = np.array(imp_folds).mean(axis=0)
top_idx  = np.argsort(mean_imp)[-15:]
fig, ax  = plt.subplots(figsize=(6, 5))
ax.barh(range(15), mean_imp[top_idx], color="#2980b9")
ax.set_yticks(range(15))
ax.set_yticklabels([feat_cols[i] for i in top_idx], fontsize=8)
ax.set_xlabel("Mean gain"); ax.set_title("XGBoost feature importance\ndelay BFP→mCherry")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); fig.savefig(str(FIG_DIR/"reg_xgb_importance.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# permutation test (uses averaged Optuna params — no re-tuning per perm)
print("  Running permutation test...", flush=True)
def xgb_cv_splits(X, y, strata):
    for tr, te in StratifiedKFold(n_splits=N_FOLDS_XGB, shuffle=True, random_state=SEED).split(X, strata):
        yield tr, te

def _fit_predict_xgb(Xtr, ytr, Xte):
    mdl = XGBRegressor(**avg_p, n_estimators=100)
    mdl.fit(Xtr, ytr, verbose=False)
    return mdl.predict(Xte)

_fit_predict = _fit_predict_xgb
null_xgb = permutation_test(X_raw, y, strata, xgb_cv_splits, N_PERM, SEED)
p_xgb = perm_figure(rho_xgb, null_xgb, "XGBoost permutation test\ndelay BFP→mCherry",
                    FIG_DIR/"reg_xgb_permtest.png")
print(f"  Permutation p={p_xgb:.3f}", flush=True)

# ── save results ───────────────────────────────────────────────────────────────
pd.DataFrame({
    "Track.ID": df["Track.ID"],
    "dataset":  df["dataset"],
    "y_true":   y.round(2),
    "en_pred":  oof_en.round(2),
    "xgb_pred": oof_xgb.round(2),
    "strata":   strata,
}).to_csv(RES_DIR/"reg_en_xgb_predictions.csv", index=False)

pd.DataFrame([
    {"method":"ElasticNet", "R2":round(r2_en,4), "pearson_r":round(r_en,4),
     "spearman_rho":round(rho_en,4), "perm_p":round(p_en,4), "n":len(y), "n_feat":len(feat_cols)},
    {"method":"XGBoost",   "R2":round(r2_xgb,4), "pearson_r":round(r_xgb,4),
     "spearman_rho":round(rho_xgb,4), "perm_p":round(p_xgb,4), "n":len(y), "n_feat":len(feat_cols)},
]).to_csv(RES_DIR/"reg_en_xgb_metrics.csv", index=False)

np.save(RES_DIR/"reg_en_null_rhos.npy", null_en)
np.save(RES_DIR/"reg_xgb_null_rhos.npy", null_xgb)

print("\n── Summary ──")
print(f"  ElasticNet:  R²={r2_en:.3f}  r={r_en:.3f}  ρ={rho_en:.3f}  p={p_en:.3f}")
print(f"  XGBoost:     R²={r2_xgb:.3f}  r={r_xgb:.3f}  ρ={rho_xgb:.3f}  p={p_xgb:.3f}")
print("\nDone.", flush=True)

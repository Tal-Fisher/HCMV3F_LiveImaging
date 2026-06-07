"""
06_bf_regression_g2r.py
ElasticNet regression: BF Cellpose embeddings (10 frames before GFP onset)
→ predict green-to-red delay.  A2 only (only movie with BF embeddings).
Productive cells only (actually turned red).
Both raw 256-dim and top-20 dims (selected by |beta| from full-data fit).
Comparison baseline: tabular ElasticNet r=0.323 (A2+A3).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging/BrightFieldEmbedding")
LIVEIMG = Path("/home/labs/ginossar/talfis/LiveImaging")
RES_DIR = BASE / "results"
FIG_DIR = BASE / "figures"
SEED    = 42

# ── load embeddings ────────────────────────────────────────────────────────────
print("Loading BF embeddings (relaxed)...", flush=True)
d = np.load(str(BASE / "embeddings" / "A2_bf_embeddings_m10_relaxed.npz"))
emb_ids   = d["gfp_track_ids"].astype(int)   # (782,)
embeddings = d["embeddings"].astype(np.float64)  # (782, 256)
id_to_row  = {tid: i for i, tid in enumerate(emb_ids)}
print(f"  {embeddings.shape[0]} cells × {embeddings.shape[1]} dims")

# ── load outcomes — productive A2 cells only ──────────────────────────────────
print("Loading outcomes...", flush=True)
model_df  = pd.read_csv(LIVEIMG / "cache" / "python_export" / "model_df.csv")
cat_df    = pd.read_csv(LIVEIMG / "cache" / "python_export" / "category_df.csv")

df = model_df[model_df["dataset"] == "A2"].copy()
df = df.merge(cat_df[["Track.ID", "productive"]], on="Track.ID", how="left")
df = df[df["productive"] == True].copy()
df["track_id"] = df["Track.ID"].str.replace("A2_", "", regex=False).astype(int)

# keep only cells that have BF embeddings
df = df[df["track_id"].isin(id_to_row)].sort_values("track_id").reset_index(drop=True)
rows = [id_to_row[tid] for tid in df["track_id"]]
X    = embeddings[rows]                             # (n, 256)
y    = df["delay_green_to_red"].values.astype(float)

print(f"  n={len(y)} productive A2 cells with BF embeddings")
print(f"  Delay range: {y.min():.0f}–{y.max():.0f} min  (median {np.median(y):.0f})")

# ── CV helper ─────────────────────────────────────────────────────────────────
def run_cv(X, y, label, ref_r=None):
    kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0],
                          alphas=np.logspace(-2, 3, 30),
                          cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
        en.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = en.predict(sc.transform(X[te]))
    r2  = r2_score(y, oof)
    r   = float(pearsonr(y, oof)[0])
    rho = float(spearmanr(y, oof)[0])
    mae = mean_absolute_error(y, oof)
    ref = f"  (ref tabular r=0.323)" if ref_r is None else f"  ref={ref_r:.3f}"
    print(f"  {label}: R²={r2:.3f}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}  MAE={mae:.0f} min{ref}")
    return oof, r2, r, rho, mae

def fit_full(X, y):
    sc = StandardScaler()
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0],
                      alphas=np.logspace(-2, 3, 30),
                      cv=5, max_iter=5000, n_jobs=-1, random_state=SEED)
    en.fit(sc.fit_transform(X), y)
    nz = int((en.coef_ != 0).sum())
    print(f"  Full-data fit: alpha={en.alpha_:.5f}  l1={en.l1_ratio_:.2f}  non-zero={nz}/{X.shape[1]}")
    return en.coef_

# ══════════════════════════════════════════════════════════════════════════════
# RAW 256-DIM
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ Raw 256-dim ═══", flush=True)
oof_256, r2_256, r_256, rho_256, mae_256 = run_cv(X, y, "5-fold CV")
betas_256 = fit_full(X, y)

# ── top-20 selection by |beta| ─────────────────────────────────────────────────
top20_dims = np.argsort(np.abs(betas_256))[-20:][::-1].tolist()
print(f"\n  Top-20 dims (by |beta|): {top20_dims}")

# ══════════════════════════════════════════════════════════════════════════════
# TOP-20 DIMS
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ Top-20 dims ═══", flush=True)
X_t20 = X[:, top20_dims]
oof_t20, r2_t20, r_t20, rho_t20, mae_t20 = run_cv(X_t20, y, "5-fold CV")
betas_t20 = fit_full(X_t20, y)

# ── save metrics ──────────────────────────────────────────────────────────────
pd.DataFrame([
    dict(task="BF_raw256_g2r",  n_dims=256, n=len(y),
         cv_r2=round(r2_256,4), cv_r=round(r_256,4),
         cv_rho=round(rho_256,4), cv_mae=round(mae_256,1),
         n_nonzero=int((betas_256!=0).sum()), ref_tabular_r=0.323),
    dict(task="BF_top20_g2r",   n_dims=20,  n=len(y),
         cv_r2=round(r2_t20,4),  cv_r=round(r_t20,4),
         cv_rho=round(rho_t20,4),  cv_mae=round(mae_t20,1),
         n_nonzero=int((betas_t20!=0).sum()), ref_tabular_r=0.323),
]).to_csv(RES_DIR / "bf_regression_g2r_metrics.csv", index=False)
print(f"\nSaved results/bf_regression_g2r_metrics.csv")

# ── scatter plot ───────────────────────────────────────────────────────────────
cat_map = dict(zip(cat_df["Track.ID"], cat_df["category"]))
cats    = df["Track.ID"].map(cat_map).values
colors  = {"early": "#e74c3c", "medium": "#f39c12", "late": "#2980b9"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"BF embeddings → green-to-red delay  |  A2, n={len(y)} productive cells\n"
    f"10 frames before GFP onset  |  ElasticNet 5-fold CV",
    fontsize=11, fontweight="bold"
)

for ax, oof, r2, r, rho, label in [
    (axes[0], oof_256, r2_256, r_256, rho_256, f"Raw 256-dim\nR²={r2_256:.3f}  r={r_256:.3f}  ρ={rho_256:.3f}"),
    (axes[1], oof_t20, r2_t20, r_t20, rho_t20, f"Top-20 dims\nR²={r2_t20:.3f}  r={r_t20:.3f}  ρ={rho_t20:.3f}"),
]:
    for cat in ["early", "medium", "late"]:
        m = cats == cat
        ax.scatter(y[m]/60, oof[m]/60, c=colors[cat], label=cat,
                   alpha=0.55, s=20, edgecolors="none")
    lo = min(y.min(), oof.min()) / 60 * 0.92
    hi = max(y.max(), oof.max()) / 60 * 1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Actual delay (h)")
    ax.set_ylabel("CV predicted (h)")
    ax.set_title(label)
    ax.legend(fontsize=8, frameon=False)
    # reference lines
    ax.text(0.05, 0.95, f"Tabular ref: r=0.323",
            transform=ax.transAxes, fontsize=8, va="top", color="#7f8c8d")

plt.tight_layout()
out = FIG_DIR / "bf_regression_g2r_scatter.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out.name}")

# ── beta bar chart (top-20) ────────────────────────────────────────────────────
order = np.argsort(np.abs(betas_t20))[::-1]
fig, ax = plt.subplots(figsize=(6, 5))
vals = betas_t20[order]
labs = [f"dim_{top20_dims[i]}" for i in order]
cols = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
ax.barh(range(20), vals, color=cols, height=0.7)
ax.set_yticks(range(20))
ax.set_yticklabels(labs, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Beta (scaled)")
ax.set_title(f"BF embeddings: top-20 dim betas\ngreen-to-red delay  ({(betas_t20!=0).sum()} non-zero)")
plt.tight_layout()
fig.savefig(str(FIG_DIR / "bf_regression_g2r_betas.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved bf_regression_g2r_betas.png")

print("\n── Summary ──")
print(f"  {'Model':<20}  {'R²':>6}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'MAE (min)':>10}")
print(f"  {'BF raw 256-dim':<20}  {r2_256:>6.3f}  {r_256:>10.3f}  {rho_256:>11.3f}  {mae_256:>10.0f}")
print(f"  {'BF top-20 dims':<20}  {r2_t20:>6.3f}  {r_t20:>10.3f}  {rho_t20:>11.3f}  {mae_t20:>10.0f}")
print(f"  {'Tabular EN (ref)':<20}  {'0.104':>6}  {'0.323':>10}  {'—':>11}  {'—':>10}")
print("Done.", flush=True)

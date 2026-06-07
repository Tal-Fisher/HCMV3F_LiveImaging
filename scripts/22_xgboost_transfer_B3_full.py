"""
22_xgboost_transfer_B3_full.py

Transfer XGBoost early-vs-rest model to B3 movie with full feature set.

  - Base 29 features: from model_df_B3.csv (R-pipeline processed, correct bleedthrough + nuclei)
  - abs_gfp_onset_min: first frame time of each track from raw spots → first-half filter
  - Frame-16 features: from raw spots with per-track alpha correction
  - Proximity features: dist_nearest, n_within_100 at GFP onset from raw spots
  - GMM: 3-component fitted to B3 delays → B3-specific early cutoff
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, roc_curve
from xgboost import XGBClassifier

BASE        = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR  = BASE / "cache" / "python_export"
RESULTS_DIR = BASE / "results" / "xgboost_early"
FIG_DIR     = BASE / "figures" / "combined"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_BLEED = 3.1
N_FRAMES    = 16
THRESHOLD   = 0.10

# ── load saved model ────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "feat_cols.json") as f:
    feat_cols = json.load(f)
scaler_mean  = np.load(RESULTS_DIR / "scaler_mean.npy")
scaler_scale = np.load(RESULTS_DIR / "scaler_scale.npy")
model = XGBClassifier()
model.load_model(str(RESULTS_DIR / "xgb_early_vs_rest.json"))
print(f"Model loaded: {len(feat_cols)} features")

# ── load raw B3 spots ───────────────────────────────────────────────────────────
spots = pd.read_csv(BASE / "CompleteImage" / "B3_Merged_spots.csv", low_memory=False)
spots.rename(columns={
    "Track ID": "track_id", "T (sec)": "t_sec", "Frame": "frame",
    "X": "x", "Y": "y",
    "Mean ch1": "ch1", "Mean ch2": "ch2",
}, inplace=True)
for col in ["ch1", "ch2", "x", "y"]:
    spots[col] = pd.to_numeric(spots[col], errors="coerce")
spots = spots[spots["track_id"].notna()].copy()
spots["track_id"] = spots["track_id"].astype(int)
spots["t_min"]    = spots["t_sec"] / 60.0
spots = spots.sort_values(["track_id", "t_min"]).reset_index(drop=True)

movie_half_min = spots["t_min"].max() / 2.0
print(f"B3 movie: {spots['t_min'].max():.0f} min total  |  half={movie_half_min:.0f} min")

# ── abs_gfp_onset_min per track (first frame time) ─────────────────────────────
onset_info = (spots.groupby("track_id")
              .agg(abs_gfp_onset_min=("t_min", "first"),
                   onset_x=("x", "first"),
                   onset_y=("y", "first"))
              .reset_index())
onset_info["movie_half_min"] = movie_half_min

# ── load model_df_B3 (R-processed 29 base features + delay labels) ─────────────
b3 = pd.read_csv(EXPORT_DIR / "model_df_B3.csv")
# Track.ID in b3 is integer-like; map to int
b3["track_id"] = pd.to_numeric(b3["Track.ID"].astype(str).str.extract(r"(\d+)")[0]).astype(int)

# merge onset info
b3 = b3.merge(onset_info, on="track_id", how="left")

# ── first-half filter ──────────────────────────────────────────────────────────
before = len(b3)
b3 = b3[b3["abs_gfp_onset_min"] <= b3["movie_half_min"]].reset_index(drop=True)
print(f"After first-half filter: {len(b3)}/{before} cells")

# ── frame-16 features from raw spots ──────────────────────────────────────────
f16_records = []
for _, row in b3.iterrows():
    tid   = row["track_id"]
    track = spots[spots["track_id"] == tid].sort_values("t_min").reset_index(drop=True)
    if len(track) < 2:
        f16_records.append({"track_id": tid,
                             "gfp_at_f16": np.nan, "bfp_at_f16": np.nan,
                             "gfp_delta_mean": np.nan, "bfp_delta_mean": np.nan})
        continue
    # per-track alpha from first 5 frames
    first5 = track.head(5)
    valid  = first5[(first5["ch1"] > 0.1) & (first5["ch2"] > 0)]
    alpha  = float(valid["ch2"].mean() / valid["ch1"].mean()) if len(valid) >= 2 else ALPHA_BLEED
    alpha  = np.clip(alpha, 0.5, 8.0)
    track["ch2_corr"] = track["ch2"] - alpha * track["ch1"]

    d16 = track.head(N_FRAMES)
    f16_idx = len(d16) - 1
    gfp_vals = d16["ch2_corr"].values.astype(float)
    bfp_vals = d16["ch1"].values.astype(float)
    f16_records.append({
        "track_id":       tid,
        "gfp_at_f16":     gfp_vals[f16_idx],
        "bfp_at_f16":     bfp_vals[f16_idx],
        "gfp_delta_mean": float(np.nanmean(np.diff(gfp_vals))) if len(gfp_vals) >= 2 else np.nan,
        "bfp_delta_mean": float(np.nanmean(np.diff(bfp_vals))) if len(bfp_vals) >= 2 else np.nan,
    })

f16_df = pd.DataFrame(f16_records)
b3 = b3.merge(f16_df, on="track_id", how="left")
print(f"Frame-16 features computed for {f16_df['gfp_at_f16'].notna().sum()} cells")

# ── proximity features ─────────────────────────────────────────────────────────
pos = onset_info[onset_info["track_id"].isin(b3["track_id"])].copy()
pos = pos.dropna(subset=["onset_x", "onset_y"]).reset_index(drop=True)
coords = pos[["onset_x", "onset_y"]].values.astype(float)
dists  = cdist(coords, coords)
np.fill_diagonal(dists, np.inf)
pos["dist_nearest"] = dists.min(axis=1)
pos["n_within_100"] = (dists <= 100).sum(axis=1)
b3 = b3.merge(pos[["track_id", "dist_nearest", "n_within_100"]], on="track_id", how="left")
print(f"Proximity: median dist_nearest={b3['dist_nearest'].median():.1f} px")

# ── productive cells + delay ────────────────────────────────────────────────────
delay     = b3["delay_green_to_red"].values.astype(float)
prod_mask = np.isfinite(delay)
b3_prod   = b3[prod_mask].copy().reset_index(drop=True)
delay_prod = delay[prod_mask]
print(f"Productive cells: {len(b3_prod)}")

# ── B3 GMM → early cutoff ──────────────────────────────────────────────────────
gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
gmm.fit(delay_prod.reshape(-1, 1))
order    = np.argsort(gmm.means_.ravel())
means    = gmm.means_.ravel()[order]
cut1     = float((means[0] + means[1]) / 2)
cut2     = float((means[1] + means[2]) / 2)
print(f"\nB3 GMM means: {means.round(0)}  →  cutoffs: {cut1:.0f} / {cut2:.0f} min")

y = (delay_prod <= cut1).astype(int)
print(f"B3 early (≤{cut1:.0f}): {y.sum()}  |  med+late: {(y==0).sum()}")

# ── build feature matrix ────────────────────────────────────────────────────────
X_raw = b3_prod[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.
X_b3 = (X_raw - scaler_mean) / scaler_scale

# ── predict ─────────────────────────────────────────────────────────────────────
proba = model.predict_proba(X_b3)[:, 1]
pred  = (proba >= THRESHOLD).astype(int)
auc   = roc_auc_score(y, proba)
tp    = ((pred==1)&(y==1)).sum(); fn = ((pred==0)&(y==1)).sum()
tn    = ((pred==0)&(y==0)).sum(); fp = ((pred==1)&(y==0)).sum()
sens  = tp/(tp+fn); spec = tn/(tn+fp); bal = (sens+spec)/2

print(f"\n{'='*50}")
print(f"B3 TRANSFER  (GMM cutoff={cut1:.0f} min, threshold={THRESHOLD})")
print(f"{'='*50}")
print(f"AUC          : {auc:.3f}")
print(f"Sensitivity  : {sens:.3f}  ({tp}/{tp+fn} early correctly identified)")
print(f"Specificity  : {spec:.3f}  ({tn}/{tn+fp} med+late correctly identified)")
print(f"Balanced acc : {bal:.3f}")
print(f"\nThreshold sweep:")
print(f"{'Thresh':>8}  {'Sens':>6}  {'Spec':>6}  {'BalAcc':>8}")
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    p   = (proba >= thr).astype(int)
    tp_ = ((p==1)&(y==1)).sum(); fn_ = ((p==0)&(y==1)).sum()
    tn_ = ((p==0)&(y==0)).sum(); fp_ = ((p==1)&(y==0)).sum()
    s   = tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
    sp  = tn_/(tn_+fp_) if (tn_+fp_)>0 else 0
    print(f"{thr:>8.2f}  {s:>6.3f}  {sp:>6.3f}  {(s+sp)/2:>8.3f}")
print(f"\nA2+A3 CV reference: AUC=0.713  Sens=0.727  Spec=0.636  Bal=0.681")

# ── figure ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"XGBoost transfer — B3 movie  (n={len(y)}, early={y.sum()}, med+late={(y==0).sum()})\n"
    f"B3 GMM cutoff: early ≤ {cut1:.0f} min  |  threshold={THRESHOLD}",
    fontsize=11, fontweight="bold"
)

# panel 1: delay distribution with GMM cutoffs
ax = axes[0]
ax.hist(delay_prod, bins=25, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(cut1, color="red",    ls="--", lw=1.8, label=f"B3 early ≤ {cut1:.0f} min")
ax.axvline(cut2, color="orange", ls="--", lw=1.5, label=f"B3 mid/late {cut2:.0f} min")
ax.axvline(911,  color="grey",   ls=":",  lw=1.2, label="A2+A3 cutoff 911 min")
ax.set_xlabel("GFP→mCherry delay (min)"); ax.set_ylabel("Cells")
ax.set_title("B3 delay distribution\nwith GMM cutoffs")
ax.legend(fontsize=8)

# panel 2: confusion matrix
ax = axes[1]
cm_disp = confusion_matrix(y, pred, labels=[1, 0])
cm_norm = cm_disp.astype(float) / cm_disp.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0,1]); ax.set_xticklabels(["Pred Early","Pred Med+Late"])
ax.set_yticks([0,1]); ax.set_yticklabels(["True Early","True Med+Late"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{100*cm_norm[i,j]:.0f}%\n(n={cm_disp[i,j]})",
                ha="center", va="center", fontsize=11,
                color="white" if cm_norm[i,j]>0.55 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f"Confusion matrix\nSens={sens:.2f}  Spec={spec:.2f}  AUC={auc:.3f}  Bal={bal:.3f}")

# panel 3: ROC curve vs A2+A3
ax = axes[2]
fpr, tpr, _ = roc_curve(y, proba)
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"B3 transfer  AUC={auc:.3f}")
roc_ref = pd.read_csv(RESULTS_DIR / "roc_curve.csv")
ax.plot(roc_ref["fpr"], roc_ref["tpr"], color="steelblue", lw=1.5, ls="--",
        alpha=0.7, label="A2+A3 CV  AUC=0.713")
ax.plot([0,1],[0,1],"k--",lw=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — B3 transfer vs A2+A3 CV")
ax.legend(loc="lower right", fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.tight_layout()
out = FIG_DIR / "xgboost_B3_transfer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

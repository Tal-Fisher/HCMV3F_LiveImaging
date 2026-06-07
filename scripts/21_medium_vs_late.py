"""Quick binary elastic net: medium vs late cells only."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE       = Path("/home/labs/ginossar/talfis/LiveImaging")
EXPORT_DIR = BASE / "cache" / "python_export"

CUT_EARLY_MED = 911
CUT_MED_LATE  = 2163

df = pd.read_csv(EXPORT_DIR / "model_df.csv")
df = df[df["abs_gfp_onset_min"] <= df["movie_half_min"]].reset_index(drop=True)

NON_FEAT = {"Track.ID","dataset","delay_green_to_red","delay_green_to_blue",
            "green_onset_min","track_start_min","abs_gfp_onset_min",
            "movie_half_min","y","gfp_snr_mean","bf_snr_mean"}
feat_cols = [c for c in df.columns if c not in NON_FEAT]

X_raw = df[feat_cols].values.astype(float)
col_med = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    bad = ~np.isfinite(X_raw[:, j])
    X_raw[bad, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
X_all = StandardScaler().fit_transform(X_raw)

delay = df["delay_green_to_red"].values.astype(float)
mask  = (delay > CUT_EARLY_MED) & (delay <= CUT_MED_LATE) | (delay > CUT_MED_LATE)
mask  = np.isfinite(delay) & (delay > CUT_EARLY_MED)   # medium + late only
X     = X_all[mask]
d     = delay[mask]
y     = (d > CUT_MED_LATE).astype(int)   # 1=late, 0=medium
print(f"Medium (0): {(y==0).sum()}   Late (1): {(y==1).sum()}")

cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_proba = np.zeros(len(y))
param_grid = {"C": [0.001,0.01,0.1,1,10], "l1_ratio": [0.0,0.25,0.5,0.75,1.0]}

from sklearn.model_selection import GridSearchCV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
for fold, (tr, te) in enumerate(cv.split(X, y)):
    gs = GridSearchCV(
        LogisticRegression(penalty="elasticnet", solver="saga",
                           class_weight="balanced", max_iter=2000, random_state=42),
        param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1
    )
    gs.fit(X[tr], y[tr])
    cv_proba[te] = gs.predict_proba(X[te])[:, 1]
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y[te], cv_proba[te]):.3f}")

print(f"\nCV AUC (medium vs late): {roc_auc_score(y, cv_proba):.3f}")

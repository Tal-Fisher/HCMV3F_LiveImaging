"""
check_covariates.py
Test whether TabICLForecaster uses the 9 covariates or ignores them.
We predict 10 windows twice: once with covariates, once without.
If predictions are identical, TabICL is treating the problem as univariate.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tabicl import TabICLForecaster

OUT          = Path("/home/labs/ginossar/talfis/LiveImaging/Forecast")
CONTEXT_LEN  = 16
H            = 10
MIN_PER_FRAME = 15
FEAT_NAMES   = ["GFP", "Nuc_BFP", "BF_mean", "BF_contrast",
                "Cell_area", "Nuc_area", "Speed", "Circ_cell", "Circ_nuc"]

# load a small sample of windows from saved predictions
results = pd.read_csv(OUT / "predictions_allwindows_h010.csv").head(10)
X_tensor = np.load(OUT / "tensor_X.npz")["X"]
Y_tensor = np.load(OUT / "tensor_y_mcherry.npz")["Y"]
meta     = pd.read_csv(OUT / "cell_metadata.csv").set_index("Track.ID")
cell_idx = {tid: i for i, tid in enumerate(meta.index)}

# build context_df for these 10 windows — WITH covariates
rows_with = []
rows_without = []
for _, r in results.iterrows():
    tid     = r["Track.ID"]
    w_start = int(r["w_start"])
    ci      = cell_idx.get(tid)
    if ci is None:
        continue
    ts = pd.date_range("2000-01-01", periods=CONTEXT_LEN, freq=f"{MIN_PER_FRAME}min")
    iid = f"{tid}_w{w_start}"
    for t_idx in range(CONTEXT_LEN):
        base = {"item_id": iid, "timestamp": ts[t_idx],
                "target": float(Y_tensor[ci, w_start + t_idx])}
        with_cov = {**base, **{fn: float(X_tensor[ci, w_start + t_idx, fi])
                                for fi, fn in enumerate(FEAT_NAMES)}}
        rows_with.append(with_cov)
        rows_without.append(base)

ctx_with    = pd.DataFrame(rows_with)
ctx_without = pd.DataFrame(rows_without)

print(f"context_df WITH covariates: {ctx_with.shape}  columns: {list(ctx_with.columns)}")
print(f"context_df WITHOUT covariates: {ctx_without.shape}  columns: {list(ctx_without.columns)}")

forecaster = TabICLForecaster(max_context_length=CONTEXT_LEN)

print("\nPredicting WITH covariates ...")
pred_with = forecaster.predict_df(ctx_with, prediction_length=H)

print("Predicting WITHOUT covariates ...")
pred_without = forecaster.predict_df(ctx_without, prediction_length=H)

# compare
vals_with    = pred_with.groupby(level=0)["target"].apply(list)
vals_without = pred_without.groupby(level=0)["target"].apply(list)

diffs = []
for iid in vals_with.index:
    a = np.array(vals_with[iid])
    b = np.array(vals_without[iid])
    diffs.append(np.max(np.abs(a - b)))
    print(f"  {iid}: max_diff={diffs[-1]:.6f}")

print(f"\nMax difference across all windows: {max(diffs):.6f}")
if max(diffs) < 1e-6:
    print("CONCLUSION: predictions are IDENTICAL → TabICL is ignoring covariates (univariate)")
else:
    print("CONCLUSION: predictions DIFFER → TabICL is using covariates (multivariate)")

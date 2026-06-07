BluetoRed_analysis
==================
Predicting BFP→mCherry delay (delay_blue_to_red) in HCMV-infected cells.
A2 + A3 datasets combined.

BIOLOGICAL QUESTION
-------------------
Can cell morphology and GFP kinetics at the time of GFP onset predict how
long it takes from BFP onset (nuclear capsid) to mCherry onset (late gene)?

  target (regression)     : delay_blue_to_red = delay_green_to_red − delay_green_to_blue
  target (classification) : early (delay_blue_to_red ≤ GMM cutoff1) vs medium+late

CELL FILTERS (ALL analyses)
---------------------------
  isfinite(delay_blue_to_red) only — productive cells with both delays observed.
  NO half-movie filter for any analysis.
  Rationale: isfinite(delay_blue_to_red) already guarantees the outcome was fully
  observed. The half-movie filter is redundant and removes cells that carry real
  signal (regression r drops 0.285→0.182 when applied; classification AUC also
  reduced). Result: n=451 cells (A2+A3) for regression; similar for classification.

FEATURE SETS
------------
REGRESSION (ElasticNet, XGBoost, TabICL) — 45 features, NO proximity
  Source: model_df_extended2.csv
  Excluded: Track.ID, dataset, delay_green_to_red, delay_green_to_blue (meta)
            + 8 EXTRAS (cell_aspect_start/mean, bfp_nuc_frac_start,
              nuc_ratio_start/end, bf_ctrst_start/end/slope — unreliable)
  No proximity features present in this dataset.

CLASSIFICATION (ElasticNet, XGBoost, TabICL) — 33 features, NO proximity
  Source: model_df.csv (base features) + frame16_features.csv (4 frame-16)
  Excluded: Track.ID, dataset, delay_green_to_red, delay_green_to_blue,
            green_onset_min, track_start_min, abs_gfp_onset_min,
            movie_half_min, y, gfp_snr_mean, bf_snr_mean
  Proximity features (dist_nearest, n_within_100) not present in model_df.csv.

45 regression features (from model_df_extended2.csv, after exclusions):
  gfp_corr_start, gfp_corr_mean, gfp_corr_sd, gfp_corr_slope,
  nuc_bfp_start, nuc_bfp_mean, nuc_bfp_sd, nuc_bfp_slope,
  nuc_area_mean, nuc_area_slope, nuc_circ_mean, nuc_circ_sd,
  nuc_ratio_mean, nuc_ratio_slope,
  area_start, area_mean, area_sd, area_slope,
  solidity_mean, solidity_sd, shape_idx_mean, gfp_snr_sd,
  bf_ctrst_mean, bf_ctrst_sd,
  gfp_ratio_start, gfp_ratio_mean, gfp_ratio_sd, gfp_ratio_slope, gfp_ratio_max,
  gfp_corr_end, nuc_bfp_end,
  gfp_corr_slope_early, gfp_corr_slope_late,
  nuc_bfp_slope_early, nuc_bfp_slope_late,
  area_slope_early, area_slope_late,
  circ_start, circ_end, circ_slope, circ_sd,
  perim_start, perim_end, perim_slope, perim_sd

33 classification features (from model_df.csv + frame16_features.csv):
  29 base: gfp_corr_*, nuc_bfp_*, nuc_area_*, nuc_circ_*, nuc_ratio_*,
           area_*, solidity_*, shape_idx_mean, gfp_snr_sd, bf_ctrst_*,
           gfp_ratio_*
  4 frame-16: gfp_at_f16, bfp_at_f16, gfp_delta_mean, bfp_delta_mean

CLASSIFICATION CATEGORIES
--------------------------
GMM G=3 fit on delay_blue_to_red in the 497-cell filtered set.
Bayes-optimal cutoffs separate early / medium / late.
Binary label: early (1) vs medium+late (0).
Expected ~114 early cells.

METHODS
-------
Regression:
  01_regression_en_xgb.py   — ElasticNet (10-fold CV) + XGBoost (4-fold, Optuna params)
  02_regression_tabicl.py   — TabICL (5-fold stratified CV)

Classification:
  03_classify_en_xgb.py     — ElasticNet/LogReg (5-fold) + XGBoost (5-fold, Optuna 50 trials)
  04_classify_tabicl.py     — TabICL classifier (5-fold)

Summary:
  05_summary_figures.py     — comparison bar charts across all methods

PERMUTATION TEST
----------------
All models include a permutation test:
  train-set label shuffling, N=500 permutations.
  Null distribution of Spearman ρ; p-value = fraction of null ρ ≥ observed ρ.

OUTPUTS
-------
results/  — CSV files with per-cell predictions and summary metrics
figures/  — scatter plots, ROC curves, importance plots, permutation plots
logs/     — LSF stdout/stderr

REFERENCE RESULTS (green_to_red, same cell set)
-------------------------------------------------
ElasticNet:  R²=0.126  Pearson r=0.359  Spearman ρ=0.337
XGBoost:     R²=0.083  Pearson r=0.303  Spearman ρ=~0.29
Classification (early vs rest, green_to_red):  AUC=0.683  Spearman ρ=0.229  p=0.002

PRIOR blue_to_red RESULTS (script 25, 45 feat, NO half-movie filter, ~800 cells)
----------------------------------------------------------------------------------
ElasticNet:  r=0.285  ρ=0.280
XGBoost:     r=0.323  ρ=0.322
TabICL:      r=0.382  ρ=0.362

CURRENT RESULTS (this folder — 45 feat regression, 33 feat classification,
                 NO half-movie filter, n=451 regression / n=525 classification,
                 500-perm permutation test, train-label shuffle)
---------------------------------------------------------------------------
Regression (delay_blue_to_red, n=451, 45 features):
  ElasticNet: R²=0.081  r=0.285  ρ=0.280  p<0.001
  XGBoost:    R²=0.051  r=0.340  ρ=0.329  p<0.001
  TabICL:     R²=0.110  r=0.334  ρ=0.320  p<0.001

Classification (early vs med+late, n=525, 114 early, GMM cutoff=1094 min):
  Note: Spearman ρ is negative (early=high prob, long delay=low prob); p-value
        uses one-sided test p=(null_rhos <= obs_rho).mean().
  ElasticNet: AUC=0.651  Sens=0.456  Spec=0.703  ρ=-0.266  p<0.001
  XGBoost:    AUC=0.668  Sens=0.518  Spec=0.752  ρ=-0.274  p<0.001
  TabICL:     AUC=0.678  Sens=0.088  Spec=0.981  ρ=-0.316  p<0.001

PYTHON ENVIRONMENTS
-------------------
  EN + XGBoost scripts : python3  (system)
  TabICL scripts       : /home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12

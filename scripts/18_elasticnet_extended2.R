options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

# ── config ─────────────────────────────────────────────────────────────────────
LATE_BLOOMER_CUTOFF <- 150
N_OUTER             <- 10
SEED                <- 42

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache", "combined")
fig_dir    <- file.path(base_dir, "figures", "combined")
res_dir    <- file.path(base_dir, "results", "elasticnet_extended2")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

# ── helpers ────────────────────────────────────────────────────────────────────
safe_slope <- function(x, t) {
  keep <- is.finite(x) & is.finite(t)
  if (sum(keep) < 2) return(NA_real_)
  unname(coef(lm(x[keep] ~ t[keep]))[2])
}
safe_sd    <- function(x) { if (sum(!is.na(x)) < 2) NA_real_ else sd(x, na.rm=TRUE) }
safe_first <- function(x) { v <- x[is.finite(x)]; if (!length(v)) NA_real_ else v[1] }
safe_last  <- function(x) { v <- x[is.finite(x)]; if (!length(v)) NA_real_ else v[length(v)] }

# ── per-dataset feature extraction ────────────────────────────────────────────
extract_features <- function(dataset) {
  cache_dir    <- file.path(base_dir, "cache", dataset)
  spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
  nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
  onset_df     <- readRDS(file.path(cache_dir, "onset_df.rds"))
  flagged_path <- file.path(cache_dir, "onset_flagged.rds")
  flagged_ids  <- if (file.exists(flagged_path)) readRDS(flagged_path)$Track.ID else character(0)

  n_feat <- 16L
  n_half <-  8L
  cell_ids <- setdiff(intersect(unique(spots$Track.ID), unique(nuc_assigned$Track.ID)),
                      flagged_ids)
  cat(sprintf("  %s: %d cells with nucleus\n", dataset, length(cell_ids)))

  feat_list <- lapply(cell_ids, function(tid) {
    cs_all <- spots[spots$Track.ID == tid, ]
    cs_all <- cs_all[order(cs_all$T..sec.), ]
    t0     <- cs_all$T..sec.[1]

    g_min   <- onset_df$green_onset_min[onset_df$Track.ID == tid]
    g_min   <- if (length(g_min) == 1 && !is.na(g_min)) g_min else 0
    g_sec   <- g_min * 60 + t0
    onset_i <- which(cs_all$T..sec. >= g_sec)
    onset_i <- if (length(onset_i) == 0) 1L else onset_i[1]

    r_min <- onset_df$red_onset_min[onset_df$Track.ID == tid]
    r_sec <- if (length(r_min) == 1 && !is.na(r_min)) r_min * 60 + t0 else Inf
    red_i <- which(cs_all$T..sec. >= r_sec)
    red_i <- if (length(red_i) == 0) nrow(cs_all) + 1L else red_i[1]

    cs <- cs_all[onset_i:min(onset_i + n_feat - 1L, red_i - 1L, nrow(cs_all)), ]
    if (nrow(cs) < n_feat) return(NULL)

    ns <- nuc_assigned[nuc_assigned$Track.ID == tid, ]
    ns <- ns[order(ns$T..sec.), ]

    # ── displacement features (X, Y from cell spots, before merge) ──────────
    dx         <- diff(cs$X)
    dy         <- diff(cs$Y)
    step_dists <- sqrt(dx^2 + dy^2)
    total_path <- sum(step_dists, na.rm = TRUE)
    net_disp   <- sqrt((cs$X[nrow(cs)] - cs$X[1])^2 + (cs$Y[nrow(cs)] - cs$Y[1])^2)
    conf_ratio <- if (is.finite(total_path) && total_path > 0) net_disp / total_path else NA_real_

    # ── merge cell spots + nucleus spots ────────────────────────────────────
    # El..a.r. exists in both → gets _cell / _nuc suffixes
    # Mean.ch1 exists in both → Mean.ch1_cell (whole-cell BFP), Mean.ch1_nuc (nuclear BFP)
    merged <- merge(
      cs[, c("Frame", "ch2_corrected", "Mean.ch2", "Mean.ch1", "Mean.ch3",
             "Area", "Solidity", "Shape.index", "SNR.ch2", "SNR.ch4", "Ctrst.ch4",
             "Circ.", "Perim.", "El..a.r.")],
      ns[, c("Frame", "Mean.ch1", "Area", "Circ.", "El..a.r.")],
      by = "Frame", suffixes = c("_cell", "_nuc"), all.x = TRUE)
    merged <- merged[order(merged$Frame), ]
    nf     <- seq_len(nrow(merged)) - 1L

    nuc_ratio     <- ifelse(merged$Area_cell > 0,
                            merged$Area_nuc / merged$Area_cell, NA_real_)
    gfp_bfp_ratio <- ifelse(!is.na(merged$Mean.ch1_nuc) & merged$Mean.ch1_nuc > 0,
                             merged$ch2_corrected / merged$Mean.ch1_nuc, NA_real_)

    early <- seq_len(n_half)
    late  <- seq(n_half + 1L, n_feat)

    # cytoplasmic BFP concentration (total cell signal minus nuclear signal, per cytoplasm area)
    cyto_area <- merged$Area_cell - merged$Area_nuc
    cyto_bfp  <- ifelse(cyto_area > 0,
                        (merged$Mean.ch1_cell * merged$Area_cell -
                         merged$Mean.ch1_nuc  * merged$Area_nuc) / cyto_area,
                        NA_real_)
    # fraction of total BFP signal residing in nucleus
    bfp_nuc_frac <- ifelse(merged$Mean.ch1_cell > 0 & merged$Area_cell > 0,
                            (merged$Mean.ch1_nuc * merged$Area_nuc) /
                            (merged$Mean.ch1_cell * merged$Area_cell),
                            NA_real_)

    # ── ORIGINAL FEATURES (29) ─────────────────────────────────────────────
    orig <- data.frame(
      Track.ID = tid,

      gfp_corr_start  = safe_first(merged$ch2_corrected),
      gfp_corr_mean   = mean(merged$ch2_corrected, na.rm = TRUE),
      gfp_corr_sd     = safe_sd(merged$ch2_corrected),
      gfp_corr_slope  = safe_slope(merged$ch2_corrected, nf),

      nuc_bfp_start   = safe_first(merged$Mean.ch1_nuc),
      nuc_bfp_mean    = mean(merged$Mean.ch1_nuc, na.rm = TRUE),
      nuc_bfp_sd      = safe_sd(merged$Mean.ch1_nuc),
      nuc_bfp_slope   = safe_slope(merged$Mean.ch1_nuc, nf),

      nuc_area_mean   = mean(merged$Area_nuc, na.rm = TRUE),
      nuc_area_slope  = safe_slope(merged$Area_nuc, nf),
      nuc_circ_mean   = mean(merged$Circ._nuc, na.rm = TRUE),
      nuc_circ_sd     = safe_sd(merged$Circ._nuc),

      nuc_ratio_mean  = mean(nuc_ratio, na.rm = TRUE),
      nuc_ratio_slope = safe_slope(nuc_ratio, nf),

      area_start      = safe_first(merged$Area_cell),
      area_mean       = mean(merged$Area_cell, na.rm = TRUE),
      area_sd         = safe_sd(merged$Area_cell),
      area_slope      = safe_slope(merged$Area_cell, nf),

      solidity_mean   = mean(merged$Solidity, na.rm = TRUE),
      solidity_sd     = safe_sd(merged$Solidity),
      shape_idx_mean  = mean(merged$Shape.index, na.rm = TRUE),

      gfp_snr_mean    = mean(merged$SNR.ch2, na.rm = TRUE),
      gfp_snr_sd      = safe_sd(merged$SNR.ch2),
      bf_snr_mean     = mean(merged$SNR.ch4, na.rm = TRUE),
      bf_ctrst_mean   = mean(merged$Ctrst.ch4, na.rm = TRUE),
      bf_ctrst_sd     = safe_sd(merged$Ctrst.ch4),

      gfp_ratio_start = safe_first(gfp_bfp_ratio),
      gfp_ratio_mean  = mean(gfp_bfp_ratio, na.rm = TRUE),
      gfp_ratio_sd    = safe_sd(gfp_bfp_ratio),
      gfp_ratio_slope = safe_slope(gfp_bfp_ratio, nf),
      gfp_ratio_max   = if (sum(is.finite(gfp_bfp_ratio)) > 0)
                          max(gfp_bfp_ratio, na.rm = TRUE) else NA_real_
    )

    # ── EXTENDED FEATURES (+16) ────────────────────────────────────────────
    ext <- data.frame(
      gfp_corr_end         = safe_last(merged$ch2_corrected),
      nuc_bfp_end          = safe_last(merged$Mean.ch1_nuc),

      gfp_corr_slope_early = safe_slope(merged$ch2_corrected[early], nf[early]),
      gfp_corr_slope_late  = safe_slope(merged$ch2_corrected[late],  nf[late]),

      nuc_bfp_slope_early  = safe_slope(merged$Mean.ch1_nuc[early], nf[early]),
      nuc_bfp_slope_late   = safe_slope(merged$Mean.ch1_nuc[late],  nf[late]),

      area_slope_early     = safe_slope(merged$Area_cell[early], nf[early]),
      area_slope_late      = safe_slope(merged$Area_cell[late],  nf[late]),

      circ_start  = safe_first(merged$Circ._cell),
      circ_end    = safe_last(merged$Circ._cell),
      circ_slope  = safe_slope(merged$Circ._cell, nf),
      circ_sd     = safe_sd(merged$Circ._cell),

      perim_start = safe_first(merged$Perim.),
      perim_end   = safe_last(merged$Perim.),
      perim_slope = safe_slope(merged$Perim., nf),
      perim_sd    = safe_sd(merged$Perim.)
    )

    # ── EXTENDED2 FEATURES (+19) ───────────────────────────────────────────
    gfp_mean_abs <- abs(mean(merged$ch2_corrected, na.rm = TRUE))

    ext2 <- data.frame(
      # Cell ellipse aspect ratio
      cell_aspect_start = safe_first(merged$El..a.r._cell),
      cell_aspect_mean  = mean(merged$El..a.r._cell, na.rm = TRUE),

      # Fraction of BFP signal in nucleus
      bfp_nuc_frac_start = safe_first(bfp_nuc_frac),

      # N/C ratio at start and end of window
      nuc_ratio_start = safe_first(nuc_ratio),
      nuc_ratio_end   = safe_last(nuc_ratio),

      # Brightfield contrast dynamics
      bf_ctrst_start  = safe_first(merged$Ctrst.ch4),
      bf_ctrst_end    = safe_last(merged$Ctrst.ch4),
      bf_ctrst_slope  = safe_slope(merged$Ctrst.ch4, nf)
    )

    cbind(orig, ext, ext2)
  })

  n_excl    <- sum(vapply(feat_list, is.null, logical(1)))
  feat_list <- Filter(Negate(is.null), feat_list)
  feat_df   <- do.call(rbind, feat_list)

  no_nuc  <- is.na(feat_df$nuc_bfp_mean) | is.nan(feat_df$nuc_bfp_mean)
  feat_df <- feat_df[!no_nuc, ]
  cat(sprintf("  %s: %d cells after filters (%d excl <16 frames, %d no nucleus)\n",
              dataset, nrow(feat_df), n_excl, sum(no_nuc)))

  feat_df$Track.ID <- paste0(dataset, "_", feat_df$Track.ID)
  feat_df
}

# ── extract features ───────────────────────────────────────────────────────────
cat("Extracting features (per-dataset)...\n")
feat_a2 <- extract_features("A2")
feat_a3 <- extract_features("A3")
feat_df  <- rbind(feat_a2, feat_a3)
cat(sprintf("Combined: %d cells total\n", nrow(feat_df)))

# ── load outcomes and merge ────────────────────────────────────────────────────
onset_comb   <- readRDS(file.path(cache_comb, "onset_df.rds"))
feat_df_orig <- readRDS(file.path(cache_comb, "feat_df.rds"))

model_df <- merge(feat_df,
                  onset_comb[, c("Track.ID", "delay_green_to_red", "delay_green_to_blue")],
                  by = "Track.ID", all.x = TRUE)
model_df <- model_df[!is.na(model_df$delay_green_to_blue) &
                     model_df$delay_green_to_blue <= LATE_BLOOMER_CUTOFF, ]
cat(sprintf("After late-bloomer filter: %d cells  (productive=%d)\n",
            nrow(model_df), sum(is.finite(model_df$delay_green_to_red))))

# ── define feature sets ────────────────────────────────────────────────────────
NON_FEAT <- c("Track.ID", "delay_green_to_red", "delay_green_to_blue",
              "gfp_snr_mean", "bf_snr_mean")

orig_feat_cols <- setdiff(colnames(feat_df_orig),
                          c("Track.ID", "delay_green_to_blue", "gfp_snr_mean", "bf_snr_mean"))
orig_feat_cols <- orig_feat_cols[orig_feat_cols %in% colnames(model_df)]

ext_feat_names <- c(
  "gfp_corr_end","nuc_bfp_end",
  "gfp_corr_slope_early","gfp_corr_slope_late",
  "nuc_bfp_slope_early","nuc_bfp_slope_late",
  "area_slope_early","area_slope_late",
  "circ_start","circ_end","circ_slope","circ_sd",
  "perim_start","perim_end","perim_slope","perim_sd"
)
ext_feat_cols <- c(orig_feat_cols, ext_feat_names)

ext2_feat_names <- c(
  "cell_aspect_start","cell_aspect_mean",
  "bfp_nuc_frac_start",
  "nuc_ratio_start","nuc_ratio_end",
  "bf_ctrst_start","bf_ctrst_end","bf_ctrst_slope"
)
ext2_feat_cols <- c(ext_feat_cols, ext2_feat_names)

cat(sprintf("Feature sets — original: %d  extended: %d  extended2: %d\n",
            length(orig_feat_cols), length(ext_feat_cols), length(ext2_feat_cols)))

# ── export CSV for downstream analysis ─────────────────────────────────────────
model_df$dataset <- ifelse(grepl("^A2_", model_df$Track.ID), "A2", "A3")
export_cols <- c("Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
                 ext2_feat_cols[ext2_feat_cols %in% colnames(model_df)])
write.csv(model_df[, export_cols],
          file.path(res_dir, "model_df_extended2.csv"), row.names = FALSE)
cat(sprintf("Exported model_df_extended2.csv (%d cells x %d features)\n",
            nrow(model_df), length(ext2_feat_cols)))

# ── elastic net helper ─────────────────────────────────────────────────────────
run_elasticnet <- function(df, fc, label, n_outer = N_OUTER, seed = SEED) {
  max_obs <- max(df$delay_green_to_red[is.finite(df$delay_green_to_red)], na.rm = TRUE)
  df$y    <- ifelse(is.finite(df$delay_green_to_red), df$delay_green_to_red, max_obs * 1.1)

  fc <- fc[vapply(fc, function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]

  X_raw <- as.matrix(df[, fc])
  for (j in seq_len(ncol(X_raw))) {
    med_j <- median(X_raw[, j], na.rm = TRUE)
    X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
  }
  X <- scale(X_raw)
  y <- df$y

  set.seed(seed)
  cv_en    <- cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
  en_model <- glmnet(X, y, alpha = 0.5, lambda = cv_en$lambda.min)
  coefs    <- as.matrix(coef(en_model))
  coefs_nz <- coefs[coefs[, 1] != 0 & rownames(coefs) != "(Intercept)", , drop = FALSE]
  coefs_nz <- coefs_nz[order(abs(coefs_nz[, 1]), decreasing = TRUE), , drop = FALSE]
  y_hat    <- as.numeric(predict(en_model, newx = X, s = cv_en$lambda.min))
  r2_in    <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

  set.seed(seed)
  outer_folds <- sample(rep(seq_len(n_outer), length.out = nrow(X)))
  cv_preds    <- numeric(length(y))
  for (k in seq_len(n_outer)) {
    test_idx  <- which(outer_folds == k)
    train_idx <- which(outer_folds != k)
    inner_cv  <- cv.glmnet(X[train_idx, ], y[train_idx], alpha = 0.5, nfolds = 5)
    fold_m    <- glmnet(X[train_idx, ], y[train_idx], alpha = 0.5, lambda = inner_cv$lambda.min)
    cv_preds[test_idx] <- as.numeric(predict(fold_m, newx = X[test_idx, ],
                                             s = inner_cv$lambda.min))
  }
  r2_cv  <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
  r_cv   <- cor(y, cv_preds, use = "complete.obs")

  prod_mask  <- is.finite(df$delay_green_to_red)
  r2_cv_prod <- 1 - sum((y[prod_mask] - cv_preds[prod_mask])^2) /
                    sum((y[prod_mask] - mean(y[prod_mask]))^2)
  r_cv_prod  <- cor(y[prod_mask], cv_preds[prod_mask], use = "complete.obs")

  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  n=%d  (productive=%d)  features=%d  non-zero=%d\n",
              nrow(df), sum(prod_mask), length(fc), nrow(coefs_nz)))
  cat(sprintf("  In-sample R²: %.3f\n", r2_in))
  cat(sprintf("  CV R² (all):  %.3f   r=%.3f\n", r2_cv, r_cv))
  cat(sprintf("  CV R² (prod): %.3f   r=%.3f\n", r2_cv_prod, r_cv_prod))
  cat("  Top coefficients:\n")
  print(round(head(coefs_nz, 15), 4))

  list(model = en_model, cv = cv_en, X = X, y = y, fc = fc,
       coefs_nz = coefs_nz, r2_in = r2_in,
       r2_cv = r2_cv, r_cv = r_cv, cv_preds = cv_preds,
       r2_cv_prod = r2_cv_prod, r_cv_prod = r_cv_prod, prod_mask = prod_mask)
}

# ── run all three models ───────────────────────────────────────────────────────
cat("\nRunning ElasticNet — Original (29 features)...\n")
res_orig <- run_elasticnet(model_df, orig_feat_cols,
                           sprintf("Original (%d features)", length(orig_feat_cols)))

cat("\nRunning ElasticNet — Extended (45 features)...\n")
res_ext  <- run_elasticnet(model_df, ext_feat_cols,
                           sprintf("Extended (%d features)", length(ext_feat_cols)))

cat("\nRunning ElasticNet — Extended2 (literature-derived features)...\n")
res_ext2 <- run_elasticnet(model_df, ext2_feat_cols,
                           sprintf("Extended2 (%d features)", length(ext2_feat_cols)))

# ── 3-column comparison table ──────────────────────────────────────────────────
cat("\n\n══════════════════════════════════════════════════════════════════\n")
cat("  COMPARISON: Original vs Extended vs Extended2\n")
cat("══════════════════════════════════════════════════════════════════\n")
fmt3 <- function(lbl, v1, v2, v3)
  sprintf("  %-28s  %8.3f  %8.3f  %8.3f  %+7.3f\n", lbl, v1, v2, v3, v3 - v1)
cat(sprintf("  %-28s  %8s  %8s  %8s  %7s\n",
            "Metric", "Original", "Extended", "Ext2", "Δ(2-0)"))
cat(sprintf("  %-28s  %8s  %8s  %8s  %7s\n",
            "------", "--------", "--------", "----", "------"))
cat(fmt3("CV R² (all cells)",  res_orig$r2_cv,      res_ext$r2_cv,      res_ext2$r2_cv))
cat(fmt3("CV r (all cells)",   res_orig$r_cv,       res_ext$r_cv,       res_ext2$r_cv))
cat(fmt3("CV R² (productive)", res_orig$r2_cv_prod, res_ext$r2_cv_prod, res_ext2$r2_cv_prod))
cat(fmt3("CV r (productive)",  res_orig$r_cv_prod,  res_ext$r_cv_prod,  res_ext2$r_cv_prod))
cat(fmt3("In-sample R²",       res_orig$r2_in,      res_ext$r2_in,      res_ext2$r2_in))
cat(sprintf("  %-28s  %8d  %8d  %8d\n", "Non-zero coefs",
            nrow(res_orig$coefs_nz), nrow(res_ext$coefs_nz), nrow(res_ext2$coefs_nz)))
cat(sprintf("  %-28s  %8d  %8d  %8d\n", "Features used",
            length(res_orig$fc), length(res_ext$fc), length(res_ext2$fc)))
cat("══════════════════════════════════════════════════════════════════\n")

# ── save CSVs ──────────────────────────────────────────────────────────────────
metrics_df <- data.frame(
  metric   = c("CV R² (all cells)", "CV r (all cells)",
               "CV R² (productive only)", "CV r (productive only)",
               "In-sample R²", "Non-zero coefs", "Features used"),
  original = c(res_orig$r2_cv, res_orig$r_cv, res_orig$r2_cv_prod, res_orig$r_cv_prod,
               res_orig$r2_in, nrow(res_orig$coefs_nz), length(res_orig$fc)),
  extended = c(res_ext$r2_cv,  res_ext$r_cv,  res_ext$r2_cv_prod,  res_ext$r_cv_prod,
               res_ext$r2_in,  nrow(res_ext$coefs_nz),  length(res_ext$fc)),
  extended2 = c(res_ext2$r2_cv, res_ext2$r_cv, res_ext2$r2_cv_prod, res_ext2$r_cv_prod,
                res_ext2$r2_in, nrow(res_ext2$coefs_nz), length(res_ext2$fc))
)
write.csv(metrics_df, file.path(res_dir, "metrics_comparison3.csv"), row.names = FALSE)
cat("Saved metrics_comparison3.csv\n")

all_feats <- union(union(rownames(res_orig$coefs_nz),
                         rownames(res_ext$coefs_nz)),
                   rownames(res_ext2$coefs_nz))
coef_df <- data.frame(
  feature      = all_feats,
  coef_original = 0, coef_extended = 0, coef_extended2 = 0,
  is_ext  = all_feats %in% ext_feat_names,
  is_ext2 = all_feats %in% ext2_feat_names
)
coef_df$coef_original[match(rownames(res_orig$coefs_nz), all_feats)] <- res_orig$coefs_nz[, 1]
coef_df$coef_extended[match(rownames(res_ext$coefs_nz),  all_feats)] <- res_ext$coefs_nz[, 1]
coef_df$coef_extended2[match(rownames(res_ext2$coefs_nz), all_feats)] <- res_ext2$coefs_nz[, 1]
coef_df <- coef_df[order(abs(coef_df$coef_extended2), decreasing = TRUE), ]
write.csv(coef_df, file.path(res_dir, "coefficients_comparison3.csv"), row.names = FALSE)
cat("Saved coefficients_comparison3.csv\n")

# ── save RDS ───────────────────────────────────────────────────────────────────
saveRDS(list(orig  = res_orig,  ext  = res_ext,  ext2 = res_ext2,
             fc_orig = orig_feat_cols, fc_ext = ext_feat_cols, fc_ext2 = ext2_feat_cols),
        file.path(res_dir, "en_extended2_results.rds"))
cat("Saved en_extended2_results.rds\n")

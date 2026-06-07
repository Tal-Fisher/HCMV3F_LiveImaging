options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

# ── config ─────────────────────────────────────────────────────────────────────
# Set EXTENDED = FALSE to reproduce the original 29-feature result
EXTENDED            <- TRUE
LATE_BLOOMER_CUTOFF <- 150   # minutes; cells with delay_green_to_blue > this removed
N_OUTER             <- 10
SEED                <- 42

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache", "combined")
fig_dir    <- file.path(base_dir, "figures", "combined")
res_dir    <- file.path(base_dir, "results", "elasticnet_extended")
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

# ── per-dataset feature extraction (mirrors 04_features.R) ────────────────────
# Run per-dataset with per-dataset caches (integer Track.IDs) then combine,
# exactly as the original pipeline does in 07_combine.R.
extract_features <- function(dataset) {
  cache_dir <- file.path(base_dir, "cache", dataset)
  spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
  nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
  onset_df     <- readRDS(file.path(cache_dir, "onset_df.rds"))
  flagged_path <- file.path(cache_dir, "onset_flagged.rds")
  flagged_ids  <- if (file.exists(flagged_path)) readRDS(flagged_path)$Track.ID else character(0)

  n_feat <- 16
  n_half <-  8
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

    r_min  <- onset_df$red_onset_min[onset_df$Track.ID == tid]
    r_sec  <- if (length(r_min) == 1 && !is.na(r_min)) r_min * 60 + t0 else Inf
    red_i  <- which(cs_all$T..sec. >= r_sec)
    red_i  <- if (length(red_i) == 0) nrow(cs_all) + 1L else red_i[1]
    cs <- cs_all[onset_i:min(onset_i + n_feat - 1L, red_i - 1L, nrow(cs_all)), ]
    if (nrow(cs) < n_feat) return(NULL)

    ns <- nuc_assigned[nuc_assigned$Track.ID == tid, ]
    ns <- ns[order(ns$T..sec.), ]

    merged <- merge(
      cs[, c("Frame","ch2_corrected","Mean.ch2","Mean.ch1","Mean.ch3",
             "Area","Solidity","Shape.index","SNR.ch2","SNR.ch4","Ctrst.ch4",
             "Circ.","Perim.")],
      ns[, c("Frame","Mean.ch1","Area","Circ.")],
      by="Frame", suffixes=c("_cell","_nuc"), all.x=TRUE)
    merged <- merged[order(merged$Frame), ]
    nf     <- seq_len(nrow(merged)) - 1

    nuc_ratio     <- ifelse(merged$Area_cell > 0,
                            merged$Area_nuc / merged$Area_cell, NA_real_)
    gfp_bfp_ratio <- ifelse(!is.na(merged$Mean.ch1_nuc) & merged$Mean.ch1_nuc > 0,
                             merged$ch2_corrected / merged$Mean.ch1_nuc, NA_real_)

    early <- seq_len(n_half)
    late  <- seq(n_half + 1L, n_feat)

    # ── ORIGINAL FEATURES (29) ────────────────────────────────────────────────
    orig <- data.frame(
      Track.ID = tid,

      gfp_corr_start = safe_first(merged$ch2_corrected),
      gfp_corr_mean  = mean(merged$ch2_corrected, na.rm=TRUE),
      gfp_corr_sd    = safe_sd(merged$ch2_corrected),
      gfp_corr_slope = safe_slope(merged$ch2_corrected, nf),

      nuc_bfp_start  = safe_first(merged$Mean.ch1_nuc),
      nuc_bfp_mean   = mean(merged$Mean.ch1_nuc,  na.rm=TRUE),
      nuc_bfp_sd     = safe_sd(merged$Mean.ch1_nuc),
      nuc_bfp_slope  = safe_slope(merged$Mean.ch1_nuc, nf),

      nuc_area_mean  = mean(merged$Area_nuc,  na.rm=TRUE),
      nuc_area_slope = safe_slope(merged$Area_nuc, nf),
      nuc_circ_mean  = mean(merged$Circ._nuc, na.rm=TRUE),
      nuc_circ_sd    = safe_sd(merged$Circ._nuc),

      nuc_ratio_mean  = mean(nuc_ratio, na.rm=TRUE),
      nuc_ratio_slope = safe_slope(nuc_ratio, nf),

      area_start = safe_first(merged$Area_cell),
      area_mean  = mean(merged$Area_cell, na.rm=TRUE),
      area_sd    = safe_sd(merged$Area_cell),
      area_slope = safe_slope(merged$Area_cell, nf),

      solidity_mean  = mean(merged$Solidity,    na.rm=TRUE),
      solidity_sd    = safe_sd(merged$Solidity),
      shape_idx_mean = mean(merged$Shape.index, na.rm=TRUE),

      gfp_snr_mean  = mean(merged$SNR.ch2,   na.rm=TRUE),
      gfp_snr_sd    = safe_sd(merged$SNR.ch2),
      bf_snr_mean   = mean(merged$SNR.ch4,   na.rm=TRUE),
      bf_ctrst_mean = mean(merged$Ctrst.ch4, na.rm=TRUE),
      bf_ctrst_sd   = safe_sd(merged$Ctrst.ch4),

      gfp_ratio_start = safe_first(gfp_bfp_ratio),
      gfp_ratio_mean  = mean(gfp_bfp_ratio,  na.rm=TRUE),
      gfp_ratio_sd    = safe_sd(gfp_bfp_ratio),
      gfp_ratio_slope = safe_slope(gfp_bfp_ratio, nf),
      gfp_ratio_max   = if (sum(is.finite(gfp_bfp_ratio)) > 0) max(gfp_bfp_ratio, na.rm=TRUE) else NA_real_
    )

    if (!EXTENDED) return(orig)

    # ── EXTENDED FEATURES (+16) ──────────────────────────────────────────────
    ext <- data.frame(
      # GFP and BFP endpoint (frame 16)
      gfp_corr_end      = safe_last(merged$ch2_corrected),
      nuc_bfp_end       = safe_last(merged$Mean.ch1_nuc),

      # GFP slope: early (frames 1-8) and late (frames 9-16)
      gfp_corr_slope_early = safe_slope(merged$ch2_corrected[early], nf[early]),
      gfp_corr_slope_late  = safe_slope(merged$ch2_corrected[late],  nf[late]),

      # BFP slope: early and late
      nuc_bfp_slope_early  = safe_slope(merged$Mean.ch1_nuc[early], nf[early]),
      nuc_bfp_slope_late   = safe_slope(merged$Mean.ch1_nuc[late],  nf[late]),

      # Cell area slope: early and late
      area_slope_early  = safe_slope(merged$Area_cell[early], nf[early]),
      area_slope_late   = safe_slope(merged$Area_cell[late],  nf[late]),

      # Cell circularity
      circ_start = safe_first(merged$Circ._cell),
      circ_end   = safe_last(merged$Circ._cell),
      circ_slope = safe_slope(merged$Circ._cell, nf),
      circ_sd    = safe_sd(merged$Circ._cell),

      # Cell perimeter
      perim_start = safe_first(merged$Perim.),
      perim_end   = safe_last(merged$Perim.),
      perim_slope = safe_slope(merged$Perim., nf),
      perim_sd    = safe_sd(merged$Perim.)
    )

    cbind(orig, ext)
  })

  n_excl    <- sum(vapply(feat_list, is.null, logical(1)))
  feat_list <- Filter(Negate(is.null), feat_list)
  feat_df   <- do.call(rbind, feat_list)

  # same nucleus filter as original pipeline
  no_nuc  <- is.na(feat_df$nuc_bfp_mean) | is.nan(feat_df$nuc_bfp_mean)
  feat_df <- feat_df[!no_nuc, ]
  cat(sprintf("  %s: %d cells after filters (%d excl <16 frames, %d no nucleus)\n",
              dataset, nrow(feat_df), n_excl, sum(no_nuc)))

  # add dataset prefix to Track.ID (mirrors 07_combine.R)
  feat_df$Track.ID <- paste0(dataset, "_", feat_df$Track.ID)
  feat_df
}

# ── run per dataset and combine ────────────────────────────────────────────────
cat("Extracting features (per-dataset, then combining)...\n")
feat_a2 <- extract_features("A2")
feat_a3 <- extract_features("A3")
feat_df <- rbind(feat_a2, feat_a3)
cat(sprintf("Combined: %d cells total\n", nrow(feat_df)))

# ── load combined outcomes ─────────────────────────────────────────────────────
onset_comb <- readRDS(file.path(cache_comb, "onset_df.rds"))
feat_df_orig_ref <- readRDS(file.path(cache_comb, "feat_df.rds"))  # for orig feat cols

model_df <- merge(feat_df,
                  onset_comb[, c("Track.ID","delay_green_to_red","delay_green_to_blue")],
                  by="Track.ID", all.x=TRUE)

# ── late-bloomer filter (matches 15_elasticnet_no_late_bloomers.R) ─────────────
model_df <- model_df[!is.na(model_df$delay_green_to_blue) &
                     model_df$delay_green_to_blue <= LATE_BLOOMER_CUTOFF, ]
cat(sprintf("After late-bloomer filter: %d cells  (productive=%d)\n",
            nrow(model_df), sum(is.finite(model_df$delay_green_to_red))))

# ── feature columns ────────────────────────────────────────────────────────────
NON_FEAT  <- c("Track.ID","delay_green_to_red","delay_green_to_blue",
               "gfp_snr_mean","bf_snr_mean")
feat_cols <- setdiff(colnames(feat_df), NON_FEAT)

# ── export extended feature matrix as CSV (for Python correlation analysis) ────
model_df$dataset <- ifelse(grepl("^A2_", model_df$Track.ID), "A2", "A3")
export_cols <- c("Track.ID", "dataset", "delay_green_to_red", "delay_green_to_blue",
                 feat_cols)
write.csv(model_df[, export_cols],
          file.path(res_dir, "model_df_extended.csv"),
          row.names=FALSE)
cat(sprintf("Exported model_df_extended.csv (%d cells x %d features)\n",
            nrow(model_df), length(feat_cols)))

orig_feat_cols <- setdiff(colnames(feat_df_orig_ref), c("Track.ID","delay_green_to_blue",
                                                         "gfp_snr_mean","bf_snr_mean"))
orig_feat_cols <- orig_feat_cols[orig_feat_cols %in% colnames(model_df)]

cat(sprintf("Feature cols — original: %d   extended: %d\n",
            length(orig_feat_cols), length(feat_cols)))

# ── elastic net helper ─────────────────────────────────────────────────────────
run_elasticnet <- function(df, fc, label, n_outer=N_OUTER, seed=SEED) {
  max_obs <- max(df$delay_green_to_red[is.finite(df$delay_green_to_red)], na.rm=TRUE)
  df$y    <- ifelse(is.finite(df$delay_green_to_red), df$delay_green_to_red, max_obs * 1.1)

  fc <- fc[vapply(fc, function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]

  X_raw <- as.matrix(df[, fc])
  for (j in seq_len(ncol(X_raw))) {
    med_j <- median(X_raw[, j], na.rm=TRUE)
    X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
  }
  X <- scale(X_raw)
  y <- df$y

  set.seed(seed)
  cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
  en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)
  coefs    <- as.matrix(coef(en_model))
  coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
  coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]
  y_hat    <- as.numeric(predict(en_model, newx=X, s=cv_en$lambda.min))
  r2_in    <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

  set.seed(seed)
  outer_folds <- sample(rep(seq_len(n_outer), length.out=nrow(X)))
  cv_preds    <- numeric(length(y))
  for (k in seq_len(n_outer)) {
    test_idx  <- which(outer_folds == k)
    train_idx <- which(outer_folds != k)
    inner_cv  <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0.5, nfolds=5)
    fold_m    <- glmnet(X[train_idx,], y[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
    cv_preds[test_idx] <- as.numeric(predict(fold_m, newx=X[test_idx,], s=inner_cv$lambda.min))
  }
  r2_cv  <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
  r_cv   <- cor(y, cv_preds, use="complete.obs")

  prod_mask  <- is.finite(df$delay_green_to_red)
  r2_cv_prod <- 1 - sum((y[prod_mask] - cv_preds[prod_mask])^2) /
                    sum((y[prod_mask] - mean(y[prod_mask]))^2)
  r_cv_prod  <- cor(y[prod_mask], cv_preds[prod_mask], use="complete.obs")

  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  n=%d  (productive=%d)  features=%d\n",
              nrow(df), sum(prod_mask), length(fc)))
  cat(sprintf("  In-sample R²: %.3f\n", r2_in))
  cat(sprintf("  CV R² (all):  %.3f   r=%.3f\n", r2_cv, r_cv))
  cat(sprintf("  CV R² (prod): %.3f   r=%.3f\n", r2_cv_prod, r_cv_prod))
  cat(sprintf("  Non-zero coefs: %d\n", nrow(coefs_nz)))
  cat("  Coefficients:\n")
  print(round(coefs_nz, 4))

  list(model=en_model, cv=cv_en, X=X, y=y, fc=fc, coefs_nz=coefs_nz,
       r2_in=r2_in, r2_cv=r2_cv, r_cv=r_cv, cv_preds=cv_preds,
       r2_cv_prod=r2_cv_prod, r_cv_prod=r_cv_prod, prod_mask=prod_mask)
}

# ── run original and extended ──────────────────────────────────────────────────
cat("\nRunning ElasticNet (original features)...\n")
res_orig <- run_elasticnet(model_df, orig_feat_cols,
                           sprintf("Original features (n=%d)", length(orig_feat_cols)))

if (EXTENDED) {
  cat("\nRunning ElasticNet (extended features)...\n")
  res_ext <- run_elasticnet(model_df, feat_cols,
                            sprintf("Extended features (n=%d)", length(feat_cols)))
}

# ── comparison table ───────────────────────────────────────────────────────────
if (EXTENDED) {
  cat("\n\n══════════════════════════════════════════════════\n")
  cat("  COMPARISON: Original vs Extended Features\n")
  cat("══════════════════════════════════════════════════\n")
  fmt <- function(lbl, v1, v2)
    sprintf("  %-28s  %8.3f  %8.3f  %+8.3f\n", lbl, v1, v2, v2 - v1)
  cat(sprintf("  %-28s  %8s  %8s  %8s\n", "Metric", "Original", "Extended", "Delta"))
  cat(sprintf("  %-28s  %8s  %8s  %8s\n", "------", "--------", "--------", "-----"))
  cat(fmt("CV R² (all cells)",   res_orig$r2_cv,      res_ext$r2_cv))
  cat(fmt("CV r (all cells)",    res_orig$r_cv,       res_ext$r_cv))
  cat(fmt("CV R² (productive)",  res_orig$r2_cv_prod, res_ext$r2_cv_prod))
  cat(fmt("CV r (productive)",   res_orig$r_cv_prod,  res_ext$r_cv_prod))
  cat(fmt("In-sample R²",        res_orig$r2_in,      res_ext$r2_in))
  cat(sprintf("  %-28s  %8d  %8d\n", "Non-zero coefs",
              nrow(res_orig$coefs_nz), nrow(res_ext$coefs_nz)))
  cat(sprintf("  %-28s  %8d  %8d\n", "Features used",
              length(res_orig$fc), length(res_ext$fc)))
  cat("══════════════════════════════════════════════════\n")
}

# ── figures ────────────────────────────────────────────────────────────────────
res_list  <- if (EXTENDED) list(orig=res_orig, ext=res_ext) else list(orig=res_orig)
res_names <- if (EXTENDED) c("Original (29)", "Extended (45)") else "Original (29)"

png(file.path(fig_dir, "elasticnet_extended_pred_vs_actual.png"),
    width=if (EXTENDED) 1200 else 700, height=600, type="cairo")
par(mfrow=c(1, length(res_list)), mar=c(4,4,3,1))
for (i in seq_along(res_list)) {
  r <- res_list[[i]]
  plot(r$y, r$cv_preds, pch=16, cex=0.5,
       col=ifelse(r$prod_mask, rgb(0,0.4,0.8,0.4), rgb(0.7,0.7,0.7,0.4)),
       xlab="Actual delay (min;  grey=censored)",
       ylab="Predicted delay (min)",
       main=sprintf("%s\nCV R²=%.3f  CV r=%.3f\nprod-only CV r=%.3f",
                    res_names[i], r$r2_cv, r$r_cv, r$r_cv_prod))
  abline(a=0, b=1, col="red", lty=2)
}
dev.off()
cat("\nSaved figures/combined/elasticnet_extended_pred_vs_actual.png\n")

if (EXTENDED) {
  all_names <- union(rownames(res_orig$coefs_nz), rownames(res_ext$coefs_nz))
  cmat <- matrix(0, nrow=length(all_names), ncol=2,
                 dimnames=list(all_names, c("Original","Extended")))
  cmat[rownames(res_orig$coefs_nz), "Original"] <- res_orig$coefs_nz[,1]
  cmat[rownames(res_ext$coefs_nz),  "Extended"] <- res_ext$coefs_nz[,1]
  ord    <- order(abs(cmat[,"Extended"]), decreasing=TRUE)
  cmat   <- cmat[ord, , drop=FALSE]
  n_show <- min(25, nrow(cmat))

  png(file.path(fig_dir, "elasticnet_extended_coef_comparison.png"),
      width=1100, height=700, type="cairo")
  par(mar=c(5,14,3,2))
  barplot(t(cmat[n_show:1, , drop=FALSE]),
          beside=TRUE, horiz=TRUE, las=1,
          col=c("steelblue","tomato"),
          main=sprintf("ElasticNet coefs: Original vs Extended  (top %d by |extended|)", n_show),
          xlab="Coefficient (z-scored features)")
  legend("bottomright", bty="n", legend=c("Original","Extended"),
         fill=c("steelblue","tomato"))
  abline(v=0, col="grey50")
  dev.off()
  cat("Saved figures/combined/elasticnet_extended_coef_comparison.png\n")
}

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(list(orig=res_orig, ext=if (EXTENDED) res_ext else NULL,
             feat_cols_orig=orig_feat_cols,
             feat_cols_ext=if (EXTENDED) feat_cols else NULL),
        file.path(res_dir, "en_extended_results.rds"))
cat("Saved results/elasticnet_extended/en_extended_results.rds\n")

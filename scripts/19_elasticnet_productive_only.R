options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

# Compares three models on productive cells only (no imputation for non-productive):
#   A) Extended (45 feat), all 743 cells, with non-productive imputation  [ref from script 16]
#   B) Extended (45 feat), productive only (451 cells), no imputation
#   C) Original (29 feat), productive only (451 cells), no imputation

LATE_BLOOMER_CUTOFF <- 150
N_OUTER             <- 10
SEED                <- 42

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache", "combined")
fig_dir    <- file.path(base_dir, "figures", "combined")
res_dir    <- file.path(base_dir, "results", "elasticnet_productive")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

safe_slope <- function(x, t) {
  keep <- is.finite(x) & is.finite(t)
  if (sum(keep) < 2) return(NA_real_)
  unname(coef(lm(x[keep] ~ t[keep]))[2])
}
safe_sd    <- function(x) { if (sum(!is.na(x)) < 2) NA_real_ else sd(x, na.rm=TRUE) }
safe_first <- function(x) { v <- x[is.finite(x)]; if (!length(v)) NA_real_ else v[1] }
safe_last  <- function(x) { v <- x[is.finite(x)]; if (!length(v)) NA_real_ else v[length(v)] }

extract_features <- function(dataset) {
  cache_dir    <- file.path(base_dir, "cache", dataset)
  spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
  nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
  onset_df     <- readRDS(file.path(cache_dir, "onset_df.rds"))
  flagged_path <- file.path(cache_dir, "onset_flagged.rds")
  flagged_ids  <- if (file.exists(flagged_path)) readRDS(flagged_path)$Track.ID else character(0)

  n_feat <- 16
  n_half <-  8
  cell_ids <- setdiff(intersect(unique(spots$Track.ID), unique(nuc_assigned$Track.ID)),
                      flagged_ids)

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

    ext <- data.frame(
      gfp_corr_end         = safe_last(merged$ch2_corrected),
      nuc_bfp_end          = safe_last(merged$Mean.ch1_nuc),
      gfp_corr_slope_early = safe_slope(merged$ch2_corrected[early], nf[early]),
      gfp_corr_slope_late  = safe_slope(merged$ch2_corrected[late],  nf[late]),
      nuc_bfp_slope_early  = safe_slope(merged$Mean.ch1_nuc[early], nf[early]),
      nuc_bfp_slope_late   = safe_slope(merged$Mean.ch1_nuc[late],  nf[late]),
      area_slope_early     = safe_slope(merged$Area_cell[early], nf[early]),
      area_slope_late      = safe_slope(merged$Area_cell[late],  nf[late]),
      circ_start = safe_first(merged$Circ._cell),
      circ_end   = safe_last(merged$Circ._cell),
      circ_slope = safe_slope(merged$Circ._cell, nf),
      circ_sd    = safe_sd(merged$Circ._cell),
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
  no_nuc    <- is.na(feat_df$nuc_bfp_mean) | is.nan(feat_df$nuc_bfp_mean)
  feat_df   <- feat_df[!no_nuc, ]
  feat_df$Track.ID <- paste0(dataset, "_", feat_df$Track.ID)
  feat_df
}

cat("Extracting features...\n")
feat_a2 <- extract_features("A2")
feat_a3 <- extract_features("A3")
feat_df <- rbind(feat_a2, feat_a3)
cat(sprintf("Combined: %d cells\n", nrow(feat_df)))

onset_comb    <- readRDS(file.path(cache_comb, "onset_df.rds"))
feat_df_orig_ref <- readRDS(file.path(cache_comb, "feat_df.rds"))

model_df <- merge(feat_df,
                  onset_comb[, c("Track.ID","delay_green_to_red","delay_green_to_blue")],
                  by="Track.ID", all.x=TRUE)

# late-bloomer filter
model_df <- model_df[!is.na(model_df$delay_green_to_blue) &
                     model_df$delay_green_to_blue <= LATE_BLOOMER_CUTOFF, ]
cat(sprintf("After late-bloomer filter: %d cells  (productive=%d)\n",
            nrow(model_df), sum(is.finite(model_df$delay_green_to_red))))

NON_FEAT  <- c("Track.ID","delay_green_to_red","delay_green_to_blue",
               "gfp_snr_mean","bf_snr_mean")
feat_cols <- setdiff(colnames(feat_df), NON_FEAT)

orig_feat_cols <- setdiff(colnames(feat_df_orig_ref),
                          c("Track.ID","delay_green_to_blue","gfp_snr_mean","bf_snr_mean"))
orig_feat_cols <- orig_feat_cols[orig_feat_cols %in% colnames(model_df)]

# productive-only subset (no imputation)
model_df_prod <- model_df[is.finite(model_df$delay_green_to_red), ]
cat(sprintf("Productive-only subset: %d cells\n", nrow(model_df_prod)))

# ── elastic net: with imputation (all cells) ───────────────────────────────────
run_en_imputed <- function(df, fc, label) {
  max_obs <- max(df$delay_green_to_red[is.finite(df$delay_green_to_red)], na.rm=TRUE)
  df$y    <- ifelse(is.finite(df$delay_green_to_red), df$delay_green_to_red, max_obs * 1.1)
  fc <- fc[vapply(fc, function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]
  X_raw <- as.matrix(df[, fc])
  for (j in seq_len(ncol(X_raw))) {
    med_j <- median(X_raw[,j], na.rm=TRUE)
    X_raw[is.na(X_raw[,j]),j] <- if (is.finite(med_j)) med_j else 0
  }
  X <- scale(X_raw); y <- df$y; n <- length(y)
  set.seed(SEED)
  outer_folds <- sample(rep(seq_len(N_OUTER), length.out=n))
  cv_preds <- numeric(n)
  for (k in seq_len(N_OUTER)) {
    test_idx <- which(outer_folds == k); train_idx <- which(outer_folds != k)
    inner_cv <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0.5, nfolds=5)
    fold_m   <- glmnet(X[train_idx,], y[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
    cv_preds[test_idx] <- as.numeric(predict(fold_m, newx=X[test_idx,], s=inner_cv$lambda.min))
  }
  r2_cv <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
  r_cv  <- cor(y, cv_preds, use="complete.obs")
  prod_mask  <- is.finite(df$delay_green_to_red)
  r2_cv_prod <- 1 - sum((y[prod_mask] - cv_preds[prod_mask])^2) /
                    sum((y[prod_mask] - mean(y[prod_mask]))^2)
  r_cv_prod  <- cor(y[prod_mask], cv_preds[prod_mask], use="complete.obs")
  set.seed(SEED)
  cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
  en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)
  coefs    <- as.matrix(coef(en_model))
  coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
  coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]
  cat(sprintf("\n=== %s  [all cells, with imputation] ===\n", label))
  cat(sprintf("  n=%d  (productive=%d)  features=%d\n", n, sum(prod_mask), length(fc)))
  cat(sprintf("  CV R² (all):  %.3f   r=%.3f\n", r2_cv, r_cv))
  cat(sprintf("  CV R² (prod): %.3f   r=%.3f\n", r2_cv_prod, r_cv_prod))
  cat(sprintf("  Non-zero coefs: %d\n", nrow(coefs_nz)))
  list(r2_cv=r2_cv, r_cv=r_cv, r2_cv_prod=r2_cv_prod, r_cv_prod=r_cv_prod,
       coefs_nz=coefs_nz, n=n, n_prod=sum(prod_mask), n_feat=length(fc),
       y=y, cv_preds=cv_preds, prod_mask=prod_mask)
}

# ── elastic net: productive only (no imputation) ───────────────────────────────
run_en_productive <- function(df, fc, label) {
  df <- df[is.finite(df$delay_green_to_red), ]  # productive cells only
  df$y <- df$delay_green_to_red
  fc <- fc[vapply(fc, function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]
  X_raw <- as.matrix(df[, fc])
  for (j in seq_len(ncol(X_raw))) {
    med_j <- median(X_raw[,j], na.rm=TRUE)
    X_raw[is.na(X_raw[,j]),j] <- if (is.finite(med_j)) med_j else 0
  }
  X <- scale(X_raw); y <- df$y; n <- length(y)
  set.seed(SEED)
  outer_folds <- sample(rep(seq_len(N_OUTER), length.out=n))
  cv_preds <- numeric(n)
  for (k in seq_len(N_OUTER)) {
    test_idx <- which(outer_folds == k); train_idx <- which(outer_folds != k)
    inner_cv <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0.5, nfolds=5)
    fold_m   <- glmnet(X[train_idx,], y[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
    cv_preds[test_idx] <- as.numeric(predict(fold_m, newx=X[test_idx,], s=inner_cv$lambda.min))
  }
  r2_cv <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
  r_cv  <- cor(y, cv_preds, use="complete.obs")
  set.seed(SEED)
  cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
  en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)
  coefs    <- as.matrix(coef(en_model))
  coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
  coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]
  y_hat_insample <- as.numeric(predict(en_model, newx=X, s=cv_en$lambda.min))
  r2_insample    <- 1 - sum((y - y_hat_insample)^2) / sum((y - mean(y))^2)
  r_insample     <- cor(y, y_hat_insample, use="complete.obs")
  cat(sprintf("\n=== %s  [productive only, no imputation] ===\n", label))
  cat(sprintf("  n=%d  features=%d\n", n, length(fc)))
  cat(sprintf("  In-sample R²: %.3f   r=%.3f\n", r2_insample, r_insample))
  cat(sprintf("  CV R²:        %.3f   r=%.3f\n", r2_cv, r_cv))
  cat(sprintf("  Overfit gap:  %.3f\n", r2_insample - r2_cv))
  cat(sprintf("  lambda.min:   %.4f\n", cv_en$lambda.min))
  cat(sprintf("  Non-zero coefs: %d\n", nrow(coefs_nz)))
  cat("  Coefficients:\n"); print(round(coefs_nz, 4))
  list(r2_cv=r2_cv, r_cv=r_cv, r2_cv_prod=r2_cv, r_cv_prod=r_cv,
       r2_insample=r2_insample, r_insample=r_insample,
       coefs_nz=coefs_nz, n=n, n_feat=length(fc),
       y=y, cv_preds=cv_preds, prod_mask=rep(TRUE, n))
}

cat("\n--- Model A: Extended (45 feat), all cells, with imputation ---\n")
res_A <- run_en_imputed(model_df, feat_cols, "Extended (45 feat)")

cat("\n--- Model B: Extended (45 feat), productive only ---\n")
res_B <- run_en_productive(model_df, feat_cols, "Extended (45 feat)")

cat("\n--- Model C: Original (29 feat), productive only ---\n")
res_C <- run_en_productive(model_df, orig_feat_cols, "Original (29 feat)")

# ── comparison table ───────────────────────────────────────────────────────────
cat("\n\n══════════════════════════════════════════════════════════════════\n")
cat("  COMPARISON\n")
cat("══════════════════════════════════════════════════════════════════\n")
fmt3 <- function(lbl, vA, vB, vC)
  sprintf("  %-26s  %8.3f  %8.3f  %8.3f\n", lbl, vA, vB, vC)
cat(sprintf("  %-26s  %8s  %8s  %8s\n",
    "Metric", "A:Ext/all", "B:Ext/prod", "C:Orig/prod"))
cat(sprintf("  %-26s  %8s  %8s  %8s\n",
    "------", "---------", "----------", "-----------"))
cat(fmt3("n cells", res_A$n, res_B$n, res_C$n))
cat(fmt3("n features", res_A$n_feat, res_B$n_feat, res_C$n_feat))
cat(fmt3("CV R² (all/prod)", res_A$r2_cv, res_B$r2_cv, res_C$r2_cv))
cat(fmt3("CV r  (all/prod)", res_A$r_cv,  res_B$r_cv,  res_C$r_cv))
cat(fmt3("CV R² (prod only)", res_A$r2_cv_prod, res_B$r2_cv, res_C$r2_cv))
cat(fmt3("CV r  (prod only)", res_A$r_cv_prod,  res_B$r_cv,  res_C$r_cv))
cat(sprintf("  %-26s  %8d  %8d  %8d\n",
    "Non-zero coefs", nrow(res_A$coefs_nz), nrow(res_B$coefs_nz), nrow(res_C$coefs_nz)))
cat("══════════════════════════════════════════════════════════════════\n")

# ── save metrics ───────────────────────────────────────────────────────────────
metrics_df <- data.frame(
  metric = c("n_cells","n_features","CV_R2","CV_r","nonzero_coefs"),
  A_ext_all    = c(res_A$n, res_A$n_feat, res_A$r2_cv, res_A$r_cv, nrow(res_A$coefs_nz)),
  B_ext_prod   = c(res_B$n, res_B$n_feat, res_B$r2_cv, res_B$r_cv, nrow(res_B$coefs_nz)),
  C_orig_prod  = c(res_C$n, res_C$n_feat, res_C$r2_cv, res_C$r_cv, nrow(res_C$coefs_nz))
)
write.csv(metrics_df, file.path(res_dir, "metrics_comparison_productive.csv"), row.names=FALSE)

# ── pred-vs-actual figure ──────────────────────────────────────────────────────
png(file.path(fig_dir, "elasticnet_productive_pred_vs_actual.png"),
    width=1500, height=600, type="cairo")
par(mfrow=c(1,3), mar=c(4,4,3,1))
panels <- list(res_A, res_B, res_C)
labels <- c("A: Extended/all\n(743 cells, imputed)",
            "B: Extended/prod-only\n(451 cells, no imputation)",
            "C: Original/prod-only\n(451 cells, no imputation)")
for (i in seq_along(panels)) {
  r <- panels[[i]]
  col_pts <- if (!is.null(r$prod_mask) && !all(r$prod_mask))
               ifelse(r$prod_mask, rgb(0,0.4,0.8,0.4), rgb(0.7,0.7,0.7,0.4))
             else rgb(0,0.4,0.8,0.4)
  plot(r$y, r$cv_preds, pch=16, cex=0.5, col=col_pts,
       xlab="Actual delay (min)", ylab="Predicted delay (min)",
       main=sprintf("%s\nCV R²=%.3f  CV r=%.3f", labels[i], r$r2_cv, r$r_cv))
  abline(a=0, b=1, col="red", lty=2)
}
dev.off()
cat("\nSaved figures/combined/elasticnet_productive_pred_vs_actual.png\n")

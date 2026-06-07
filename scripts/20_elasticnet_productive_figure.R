options(bitmapType = "png")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

# Best elastic net model: extended features (45), productive cells only, no late bloomers.
# Produces a 2-panel figure: CV scatter + coefficient bar chart.

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
  n_feat <- 16; n_half <- 8
  cell_ids <- setdiff(intersect(unique(spots$Track.ID), unique(nuc_assigned$Track.ID)), flagged_ids)

  feat_list <- lapply(cell_ids, function(tid) {
    cs_all <- spots[spots$Track.ID == tid, ]; cs_all <- cs_all[order(cs_all$T..sec.), ]
    t0     <- cs_all$T..sec.[1]
    g_min  <- onset_df$green_onset_min[onset_df$Track.ID == tid]
    g_min  <- if (length(g_min) == 1 && !is.na(g_min)) g_min else 0
    g_sec  <- g_min * 60 + t0
    onset_i <- which(cs_all$T..sec. >= g_sec); onset_i <- if (!length(onset_i)) 1L else onset_i[1]
    r_min  <- onset_df$red_onset_min[onset_df$Track.ID == tid]
    r_sec  <- if (length(r_min) == 1 && !is.na(r_min)) r_min * 60 + t0 else Inf
    red_i  <- which(cs_all$T..sec. >= r_sec); red_i <- if (!length(red_i)) nrow(cs_all) + 1L else red_i[1]
    cs <- cs_all[onset_i:min(onset_i + n_feat - 1L, red_i - 1L, nrow(cs_all)), ]
    if (nrow(cs) < n_feat) return(NULL)
    ns <- nuc_assigned[nuc_assigned$Track.ID == tid, ]; ns <- ns[order(ns$T..sec.), ]
    merged <- merge(
      cs[, c("Frame","ch2_corrected","Mean.ch2","Mean.ch1","Mean.ch3",
             "Area","Solidity","Shape.index","SNR.ch2","SNR.ch4","Ctrst.ch4","Circ.","Perim.")],
      ns[, c("Frame","Mean.ch1","Area","Circ.")],
      by="Frame", suffixes=c("_cell","_nuc"), all.x=TRUE)
    merged <- merged[order(merged$Frame), ]; nf <- seq_len(nrow(merged)) - 1
    nuc_ratio     <- ifelse(merged$Area_cell > 0, merged$Area_nuc / merged$Area_cell, NA_real_)
    gfp_bfp_ratio <- ifelse(!is.na(merged$Mean.ch1_nuc) & merged$Mean.ch1_nuc > 0,
                             merged$ch2_corrected / merged$Mean.ch1_nuc, NA_real_)
    early <- seq_len(n_half); late <- seq(n_half + 1L, n_feat)
    cbind(
      data.frame(Track.ID=tid, dataset=dataset,
        gfp_corr_start=safe_first(merged$ch2_corrected),
        gfp_corr_mean=mean(merged$ch2_corrected,na.rm=TRUE),
        gfp_corr_sd=safe_sd(merged$ch2_corrected),
        gfp_corr_slope=safe_slope(merged$ch2_corrected,nf),
        nuc_bfp_start=safe_first(merged$Mean.ch1_nuc),
        nuc_bfp_mean=mean(merged$Mean.ch1_nuc,na.rm=TRUE),
        nuc_bfp_sd=safe_sd(merged$Mean.ch1_nuc),
        nuc_bfp_slope=safe_slope(merged$Mean.ch1_nuc,nf),
        nuc_area_mean=mean(merged$Area_nuc,na.rm=TRUE),
        nuc_area_slope=safe_slope(merged$Area_nuc,nf),
        nuc_circ_mean=mean(merged$Circ._nuc,na.rm=TRUE),
        nuc_circ_sd=safe_sd(merged$Circ._nuc),
        nuc_ratio_mean=mean(nuc_ratio,na.rm=TRUE),
        nuc_ratio_slope=safe_slope(nuc_ratio,nf),
        area_start=safe_first(merged$Area_cell),
        area_mean=mean(merged$Area_cell,na.rm=TRUE),
        area_sd=safe_sd(merged$Area_cell),
        area_slope=safe_slope(merged$Area_cell,nf),
        solidity_mean=mean(merged$Solidity,na.rm=TRUE),
        solidity_sd=safe_sd(merged$Solidity),
        shape_idx_mean=mean(merged$Shape.index,na.rm=TRUE),
        gfp_snr_mean=mean(merged$SNR.ch2,na.rm=TRUE),
        gfp_snr_sd=safe_sd(merged$SNR.ch2),
        bf_snr_mean=mean(merged$SNR.ch4,na.rm=TRUE),
        bf_ctrst_mean=mean(merged$Ctrst.ch4,na.rm=TRUE),
        bf_ctrst_sd=safe_sd(merged$Ctrst.ch4),
        gfp_ratio_start=safe_first(gfp_bfp_ratio),
        gfp_ratio_mean=mean(gfp_bfp_ratio,na.rm=TRUE),
        gfp_ratio_sd=safe_sd(gfp_bfp_ratio),
        gfp_ratio_slope=safe_slope(gfp_bfp_ratio,nf),
        gfp_ratio_max=if(sum(is.finite(gfp_bfp_ratio))>0) max(gfp_bfp_ratio,na.rm=TRUE) else NA_real_),
      data.frame(
        gfp_corr_end=safe_last(merged$ch2_corrected),
        nuc_bfp_end=safe_last(merged$Mean.ch1_nuc),
        gfp_corr_slope_early=safe_slope(merged$ch2_corrected[early],nf[early]),
        gfp_corr_slope_late=safe_slope(merged$ch2_corrected[late],nf[late]),
        nuc_bfp_slope_early=safe_slope(merged$Mean.ch1_nuc[early],nf[early]),
        nuc_bfp_slope_late=safe_slope(merged$Mean.ch1_nuc[late],nf[late]),
        area_slope_early=safe_slope(merged$Area_cell[early],nf[early]),
        area_slope_late=safe_slope(merged$Area_cell[late],nf[late]),
        circ_start=safe_first(merged$Circ._cell),
        circ_end=safe_last(merged$Circ._cell),
        circ_slope=safe_slope(merged$Circ._cell,nf),
        circ_sd=safe_sd(merged$Circ._cell),
        perim_start=safe_first(merged$Perim.),
        perim_end=safe_last(merged$Perim.),
        perim_slope=safe_slope(merged$Perim.,nf),
        perim_sd=safe_sd(merged$Perim.))
    )
  })
  feat_list <- Filter(Negate(is.null), feat_list)
  feat_df   <- do.call(rbind, feat_list)
  no_nuc    <- is.na(feat_df$nuc_bfp_mean) | is.nan(feat_df$nuc_bfp_mean)
  feat_df   <- feat_df[!no_nuc, ]
  feat_df$Track.ID <- paste0(dataset, "_", feat_df$Track.ID)
  feat_df
}

cat("Extracting features...\n")
feat_df <- rbind(extract_features("A2"), extract_features("A3"))

onset_comb <- readRDS(file.path(cache_comb, "onset_df.rds"))
model_df <- merge(feat_df,
                  onset_comb[, c("Track.ID","delay_green_to_red","delay_green_to_blue")],
                  by="Track.ID", all.x=TRUE)
model_df <- model_df[!is.na(model_df$delay_green_to_blue) &
                     model_df$delay_green_to_blue <= LATE_BLOOMER_CUTOFF, ]

# productive only
df <- model_df[is.finite(model_df$delay_green_to_red), ]
df$y <- df$delay_green_to_red
cat(sprintf("n = %d productive cells (after late-bloomer filter)\n", nrow(df)))

NON_FEAT  <- c("Track.ID","dataset","delay_green_to_red","delay_green_to_blue","y",
               "gfp_snr_mean","bf_snr_mean")
feat_cols <- setdiff(colnames(df), NON_FEAT)
feat_cols <- feat_cols[vapply(feat_cols, function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]
cat(sprintf("Features: %d\n", length(feat_cols)))

X_raw <- as.matrix(df[, feat_cols])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[,j], na.rm=TRUE)
  X_raw[is.na(X_raw[,j]),j] <- if (is.finite(med_j)) med_j else 0
}
X <- scale(X_raw); y <- df$y; n <- length(y)

# ── nested CV ─────────────────────────────────────────────────────────────────
set.seed(SEED)
outer_folds <- sample(rep(seq_len(N_OUTER), length.out=n))
cv_preds    <- numeric(n)
for (k in seq_len(N_OUTER)) {
  test_idx  <- which(outer_folds == k)
  train_idx <- which(outer_folds != k)
  inner_cv  <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0.5, nfolds=5)
  fold_m    <- glmnet(X[train_idx,], y[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
  cv_preds[test_idx] <- as.numeric(predict(fold_m, newx=X[test_idx,], s=inner_cv$lambda.min))
}
r2_cv <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
r_cv  <- cor(y, cv_preds, use="complete.obs")
cat(sprintf("CV R² = %.4f   CV r = %.4f\n", r2_cv, r_cv))

# ── full-data model for coefficients ──────────────────────────────────────────
set.seed(SEED)
cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)
coefs    <- as.matrix(coef(en_model))
coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]
cat(sprintf("Non-zero coefficients: %d\n", nrow(coefs_nz)))

# save results
saveRDS(list(cv_preds=cv_preds, y=y, r2_cv=r2_cv, r_cv=r_cv,
             coefs_nz=coefs_nz, feat_cols=feat_cols, dataset=df$dataset),
        file.path(res_dir, "en_productive_results.rds"))
write.csv(data.frame(feature=rownames(coefs_nz), coefficient=coefs_nz[,1]),
          file.path(res_dir, "coefficients_productive.csv"), row.names=FALSE)
write.csv(data.frame(dataset=df$dataset, actual_min=y, predicted_min=cv_preds),
          file.path(res_dir, "cv_predictions_productive.csv"), row.names=FALSE)
cat(sprintf("r2_cv=%.4f\nr_cv=%.4f\nn=%d\n", r2_cv, r_cv, n),
    file=file.path(res_dir, "cv_metrics.txt"))

# ── pretty feature labels ──────────────────────────────────────────────────────
label_map <- c(
  nuc_bfp_start        = "Nuclear BFP (start)",
  nuc_bfp_mean         = "Nuclear BFP (mean)",
  nuc_bfp_end          = "Nuclear BFP (end)",
  nuc_bfp_sd           = "Nuclear BFP (SD)",
  nuc_bfp_slope        = "Nuclear BFP slope",
  nuc_bfp_slope_early  = "Nuclear BFP slope (early)",
  nuc_bfp_slope_late   = "Nuclear BFP slope (late)",
  nuc_area_mean        = "Nuclear area (mean)",
  nuc_area_slope       = "Nuclear area slope",
  nuc_circ_mean        = "Nuclear circularity (mean)",
  nuc_circ_sd          = "Nuclear circularity (SD)",
  nuc_ratio_mean       = "Nuc/Cell area ratio (mean)",
  nuc_ratio_slope      = "Nuc/Cell area ratio slope",
  gfp_corr_start       = "GFP intensity (start)",
  gfp_corr_mean        = "GFP intensity (mean)",
  gfp_corr_end         = "GFP intensity (end)",
  gfp_corr_sd          = "GFP intensity (SD)",
  gfp_corr_slope       = "GFP intensity slope",
  gfp_corr_slope_early = "GFP slope (early)",
  gfp_corr_slope_late  = "GFP slope (late)",
  gfp_ratio_start      = "GFP/BFP ratio (start)",
  gfp_ratio_mean       = "GFP/BFP ratio (mean)",
  gfp_ratio_sd         = "GFP/BFP ratio (SD)",
  gfp_ratio_slope      = "GFP/BFP ratio slope",
  gfp_ratio_max        = "GFP/BFP ratio (max)",
  gfp_snr_sd           = "GFP SNR (SD)",
  area_start           = "Cell area (start)",
  area_mean            = "Cell area (mean)",
  area_sd              = "Cell area (SD)",
  area_slope           = "Cell area slope",
  area_slope_early     = "Cell area slope (early)",
  area_slope_late      = "Cell area slope (late)",
  solidity_mean        = "Cell solidity (mean)",
  solidity_sd          = "Cell solidity (SD)",
  shape_idx_mean       = "Shape index (mean)",
  circ_start           = "Cell circularity (start)",
  circ_end             = "Cell circularity (end)",
  circ_slope           = "Cell circularity slope",
  circ_sd              = "Cell circularity (SD)",
  perim_start          = "Cell perimeter (start)",
  perim_end            = "Cell perimeter (end)",
  perim_slope          = "Cell perimeter slope",
  perim_sd             = "Cell perimeter (SD)",
  bf_ctrst_mean        = "BF contrast (mean)",
  bf_ctrst_sd          = "BF contrast (SD)"
)
pretty_names <- function(nms) {
  sapply(nms, function(n) if (!is.na(label_map[n])) label_map[n] else n)
}

# ── figure ────────────────────────────────────────────────────────────────────
out_png <- file.path(fig_dir, "elasticnet_productive_best_model.png")
pdf(sub("\\.png$", ".pdf", out_png), width=13, height=6)

layout(matrix(c(1,2), 1, 2), widths=c(1, 1.15))

# ── Panel 1: CV predicted vs actual ──────────────────────────────────────────
par(mar=c(5, 5, 4, 2))
col_ds <- ifelse(df$dataset == "A2", rgb(0.12, 0.47, 0.71, 0.55),
                                     rgb(0.84, 0.37, 0.00, 0.55))

# axis limits
lims <- range(c(y, cv_preds), na.rm=TRUE)
lims <- c(floor(lims[1]/200)*200, ceiling(lims[2]/200)*200)

plot(y / 60, cv_preds / 60, pch=19, cex=0.75, col=col_ds,
     xlim=lims/60, ylim=lims/60,
     xlab="Actual GFP→mCherry delay (h)",
     ylab="Predicted GFP→mCherry delay (h)",
     main="ElasticNet: productive cells only\n(extended features, 10-fold nested CV)",
     cex.main=1.0, cex.lab=0.95, las=1)
abline(a=0, b=1, col="grey40", lty=2, lwd=1.5)

# annotations
legend("topleft", bty="n",
       legend=c(sprintf("R² = %.3f", r2_cv),
                sprintf("r  = %.3f", r_cv),
                sprintf("n  = %d cells", n)),
       cex=1.0)
legend("bottomright", bty="n",
       legend=c("A2","A3"),
       pch=19, col=c(rgb(0.12,0.47,0.71), rgb(0.84,0.37,0.00)), cex=0.9)

# ── Panel 2: coefficient bar chart ───────────────────────────────────────────
par(mar=c(5, 12, 4, 3))
coef_vals  <- coefs_nz[, 1]
coef_names <- pretty_names(rownames(coefs_nz))
n_show     <- length(coef_vals)   # show all non-zero

# order: largest |coef| at top
ord     <- order(abs(coef_vals))
cv      <- coef_vals[ord]
cn      <- coef_names[ord]
bar_col <- ifelse(cv > 0, rgb(0.84, 0.37, 0.00, 0.85), rgb(0.12, 0.47, 0.71, 0.85))

xmax <- max(abs(cv)) * 1.25
bp   <- barplot(cv, horiz=TRUE, names.arg=cn, las=1,
                col=bar_col, border=NA,
                xlim=c(-xmax, xmax),
                xlab="Coefficient (z-scored features)",
                main=sprintf("Non-zero coefficients (n=%d)", n_show),
                cex.names=0.72, cex.main=1.0, cex.lab=0.95)
abline(v=0, col="grey40", lwd=1.2)

# sign legend
legend("bottomright", bty="n",
       legend=c("Positive (longer delay)", "Negative (shorter delay)"),
       fill=c(rgb(0.84,0.37,0.00,0.85), rgb(0.12,0.47,0.71,0.85)), cex=0.8)

dev.off()
cat(sprintf("Saved %s\n", out_png))

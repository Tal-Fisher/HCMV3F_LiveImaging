options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

data_dir <- "/home/labs/ginossar/talfis/LiveImaging/CompleteImage"
out_dir  <- "/home/labs/ginossar/talfis/LiveImaging"

# ── helpers ────────────────────────────────────────────────────────────────────
roll_mean_k <- function(x, k) {
  n <- length(x); half <- floor(k / 2)
  vapply(seq_len(n), function(i) {
    v <- x[max(1,i-half):min(n,i+half)]
    if (all(is.na(v))) NA_real_ else mean(v, na.rm=TRUE)
  }, numeric(1))
}
safe_slope <- function(x, t) {
  keep <- is.finite(x) & is.finite(t)
  if (sum(keep) < 2) return(NA_real_)
  unname(coef(lm(x[keep] ~ t[keep]))[2])
}
safe_sd   <- function(x) { if (sum(!is.na(x)) < 2) NA_real_ else sd(x, na.rm=TRUE) }
safe_first <- function(x) { x <- x[is.finite(x)]; if (!length(x)) NA_real_ else x[1] }
detect_onset <- function(x, thresh, n_consec) {
  pos <- !is.na(x) & x > thresh
  if (!any(pos)) return(NA_integer_)
  for (i in n_consec:length(pos))
    if (all(pos[(i-n_consec+1):i])) return(i - n_consec + 1L)
  NA_integer_
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
cat("Loading data...\n")
spots <- read.csv(file.path(data_dir, "A2_Merged_spots.csv"))
nuc   <- read.csv(file.path(data_dir, "A2_nuclei_spots.csv"))

spots <- spots[!is.na(spots$Track.ID), ]
nuc   <- nuc[!is.na(nuc$Track.ID), ]
spots <- spots[order(spots$Track.ID, spots$T..sec.), ]
nuc   <- nuc[order(nuc$Track.ID, nuc$T..sec.), ]

# ── filter split tracks ───────────────────────────────────────────────────────
first_gfp <- tapply(spots$Mean.ch2, spots$Track.ID, function(x) x[1])
keep_ids  <- as.integer(names(first_gfp)[first_gfp < 2.5])
spots     <- spots[spots$Track.ID %in% keep_ids, ]

# ── bleedthrough correction ───────────────────────────────────────────────────
n_baseline  <- 5
baseline_df <- do.call(rbind, lapply(split(spots, spots$Track.ID), function(df)
  head(df[order(df$T..sec.), ], n_baseline)))
baseline_df <- baseline_df[is.finite(baseline_df$Mean.ch1) & baseline_df$Mean.ch1 > 0 &
                            is.finite(baseline_df$Mean.ch2), ]
alpha <- unname(coef(lm(Mean.ch2 ~ Mean.ch1 - 1, data = baseline_df))[1])
cat(sprintf("Bleedthrough alpha: %.5f\n", alpha))
spots$ch2_corrected <- spots$Mean.ch2 - alpha * spots$Mean.ch1

# ── remove cells with GFP already on at track start ──────────────────────────
first_corr <- vapply(split(spots, spots$Track.ID),
                     function(df) df$ch2_corrected[1], numeric(1))
clean_ids  <- as.integer(names(first_corr)[first_corr < 1.5])
spots      <- spots[spots$Track.ID %in% clean_ids, ]
cat(sprintf("Cells after early-GFP filter: %d\n", length(unique(spots$Track.ID))))

# ══════════════════════════════════════════════════════════════════════════════
# NUCLEUS ASSIGNMENT  (nearest nucleus per frame → majority vote)
# ══════════════════════════════════════════════════════════════════════════════
cat("Assigning nuclei...\n")
max_dist <- 100

all_frames <- intersect(unique(spots$Frame), unique(nuc$Frame))

# frame-wise nearest nucleus for each cell
frame_matches <- do.call(rbind, lapply(all_frames, function(fr) {
  cs <- spots[spots$Frame == fr, c("Track.ID","X","Y")]
  ns <- nuc[nuc$Frame == fr,   c("Track.ID","X","Y")]
  if (!nrow(cs) || !nrow(ns)) return(NULL)

  # pairwise distances
  pairs <- merge(
    data.frame(cell_id=cs$Track.ID, cx=cs$X, cy=cs$Y),
    data.frame(nuc_id =ns$Track.ID, nx=ns$X, ny=ns$Y),
    by=NULL
  )
  pairs$dist <- sqrt((pairs$cx-pairs$nx)^2 + (pairs$cy-pairs$ny)^2)
  pairs      <- pairs[pairs$dist <= max_dist, ]
  if (!nrow(pairs)) return(NULL)

  # nearest nucleus per cell in this frame
  pairs[ave(pairs$dist, pairs$cell_id, FUN=min) == pairs$dist,
        c("cell_id","nuc_id","dist")]
}))

# majority-vote: for each cell, which nuc_id appears most often?
best_nuc <- do.call(rbind, lapply(split(frame_matches, frame_matches$cell_id), function(df) {
  tab   <- sort(table(df$nuc_id), decreasing=TRUE)
  n_frames_total <- nrow(df)
  best  <- as.integer(names(tab)[1])
  frac  <- tab[1] / n_frames_total
  data.frame(cell_id=df$cell_id[1], nuc_id=best,
             frac_nearest=as.numeric(frac),
             n_frames=n_frames_total)
}))

# keep only confident assignments (appeared as nearest in ≥60% of frames)
best_nuc <- best_nuc[best_nuc$frac_nearest >= 0.6 & best_nuc$n_frames >= 5, ]

# resolve: one nucleus per cell (already is, since we took the top-1 per cell)
# also resolve: one cell per nucleus
best_nuc <- best_nuc[order(-best_nuc$frac_nearest), ]
best_nuc <- best_nuc[!duplicated(best_nuc$nuc_id), ]

cat(sprintf("Cells with assigned nucleus: %d / %d\n",
            nrow(best_nuc), length(unique(spots$Track.ID))))

# relabel nuc table: nuc Track.ID → cell Track.ID
nuc_assigned <- merge(nuc, best_nuc[, c("cell_id","nuc_id")],
                      by.x="Track.ID", by.y="nuc_id", all=FALSE)
nuc_assigned$Track.ID <- nuc_assigned$cell_id
nuc_assigned$cell_id  <- NULL

# ══════════════════════════════════════════════════════════════════════════════
# DETECT RED ONSET
# ══════════════════════════════════════════════════════════════════════════════
red_thresh <- 2.25; red_n <- 5; smooth_k <- 3

onset_list <- lapply(split(spots, spots$Track.ID), function(df) {
  df  <- df[order(df$T..sec.), ]
  n   <- nrow(df)
  ch3_sm <- vapply(seq_len(n), function(i)
    mean(df$Mean.ch3[max(1, i-1):min(n, i+1)], na.rm=TRUE), numeric(1))

  pos <- !is.na(ch3_sm) & ch3_sm > red_thresh
  red_valid <- FALSE
  onset_t   <- NA_real_

  if (n >= red_n) {
    for (i in red_n:n) {
      if (all(pos[(i - red_n + 1):i])) {
        delta <- ch3_sm[i] - ch3_sm[i - red_n + 1]
        if (is.finite(delta) && delta > 0.15) {
          red_valid <- TRUE
          onset_t   <- df$T..sec.[i - red_n + 1]
          break
        }
      }
    }
  }

  data.frame(
    Track.ID           = df$Track.ID[1],
    delay_green_to_red = if (red_valid) (onset_t - df$T..sec.[1]) / 60 else Inf
  )
})
onset_df <- do.call(rbind, onset_list)
cat(sprintf("Red events: %d / %d  (median delay %.0f min)\n",
            sum(is.finite(onset_df$delay_green_to_red)),
            nrow(onset_df),
            median(onset_df$delay_green_to_red[is.finite(onset_df$delay_green_to_red)])))

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (first 16 frames, cell + nucleus)
# ══════════════════════════════════════════════════════════════════════════════
n_feat <- 16

# cells that have a nucleus assignment
cell_ids_with_nuc <- intersect(unique(spots$Track.ID), unique(nuc_assigned$Track.ID))

feat_list <- lapply(cell_ids_with_nuc, function(tid) {
  cs <- head(spots[spots$Track.ID == tid, ][order(spots$T..sec.[spots$Track.ID == tid]), ], n_feat)
  ns <- nuc_assigned[nuc_assigned$Track.ID == tid, ]
  ns <- ns[order(ns$T..sec.), ]

  # align by Frame
  merged <- merge(cs[, c("Frame","ch2_corrected","Mean.ch2","Mean.ch1","Mean.ch3",
                          "Area","P.stuck","Solidity","Shape.index",
                          "El..long.axis","SNR.ch2","SNR.ch4","Ctrst.ch4")],
                  ns[, c("Frame","Mean.ch1","Area","Circ.")],
                  by="Frame", suffixes=c("_cell","_nuc"), all.x=TRUE)
  merged  <- merged[order(merged$Frame), ]
  nf      <- seq_len(nrow(merged)) - 1

  nuc_ratio <- ifelse(merged$Area_cell > 0,
                      merged$Area_nuc / merged$Area_cell, NA_real_)

  data.frame(
    Track.ID = tid,

    # corrected GFP (cell)
    gfp_corr_start = safe_first(merged$ch2_corrected),
    gfp_corr_mean  = mean(merged$ch2_corrected, na.rm=TRUE),
    gfp_corr_sd    = safe_sd(merged$ch2_corrected),
    gfp_corr_slope = safe_slope(merged$ch2_corrected, nf),

    # nuclear BFP
    nuc_bfp_start  = safe_first(merged$Mean.ch1_nuc),
    nuc_bfp_mean   = mean(merged$Mean.ch1_nuc,  na.rm=TRUE),
    nuc_bfp_sd     = safe_sd(merged$Mean.ch1_nuc),
    nuc_bfp_slope  = safe_slope(merged$Mean.ch1_nuc, nf),

    # nucleus morphology
    nuc_area_mean  = mean(merged$Area_nuc,  na.rm=TRUE),
    nuc_area_slope = safe_slope(merged$Area_nuc, nf),
    nuc_circ_mean  = mean(merged$Circ.,     na.rm=TRUE),
    nuc_circ_sd    = safe_sd(merged$Circ.),

    # nucleus/cell area ratio
    nuc_ratio_mean  = mean(nuc_ratio, na.rm=TRUE),
    nuc_ratio_slope = safe_slope(nuc_ratio, nf),

    # cell area
    area_start     = safe_first(merged$Area_cell),
    area_mean      = mean(merged$Area_cell, na.rm=TRUE),
    area_sd        = safe_sd(merged$Area_cell),
    area_slope     = safe_slope(merged$Area_cell, nf),

    # speed proxy
    speed_mean     = mean(merged$P.stuck, na.rm=TRUE),
    speed_sd       = safe_sd(merged$P.stuck),

    # morphology
    solidity_mean  = mean(merged$Solidity,      na.rm=TRUE),
    solidity_sd    = safe_sd(merged$Solidity),
    shape_idx_mean = mean(merged$Shape.index,   na.rm=TRUE),
    long_ax_mean   = mean(merged$El..long.axis, na.rm=TRUE),
    long_ax_slope  = safe_slope(merged$El..long.axis, nf),

    # SNR / contrast
    gfp_snr_mean   = mean(merged$SNR.ch2,   na.rm=TRUE),
    gfp_snr_sd     = safe_sd(merged$SNR.ch2),
    bf_snr_mean    = mean(merged$SNR.ch4,   na.rm=TRUE),
    bf_ctrst_mean  = mean(merged$Ctrst.ch4, na.rm=TRUE),
    bf_ctrst_sd    = safe_sd(merged$Ctrst.ch4)
  )
})

feat_df  <- do.call(rbind, feat_list)
model_df <- merge(feat_df,
                  onset_df[, c("Track.ID","delay_green_to_red")],
                  by="Track.ID", all.x=TRUE)

cat(sprintf("Feature matrix: %d cells x %d features\n",
            nrow(model_df), ncol(feat_df) - 1))

# ══════════════════════════════════════════════════════════════════════════════
# ELASTICNET
# ══════════════════════════════════════════════════════════════════════════════
max_obs  <- max(model_df$delay_green_to_red[is.finite(model_df$delay_green_to_red)], na.rm=TRUE)
model_df$y <- ifelse(is.finite(model_df$delay_green_to_red),
                     model_df$delay_green_to_red, max_obs * 1.1)

feat_cols <- setdiff(colnames(feat_df), "Track.ID")
feat_cols <- feat_cols[vapply(feat_cols,
  function(c) sum(is.finite(model_df[[c]])) >= 5, logical(1))]

X_raw <- as.matrix(model_df[, feat_cols])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[,j], na.rm=TRUE)
  X_raw[is.na(X_raw[,j]), j] <- if (is.finite(med_j)) med_j else 0
}
X <- scale(X_raw)
y <- model_df$y

set.seed(42)
cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)

coefs    <- as.matrix(coef(en_model))
coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]

y_hat <- as.numeric(predict(en_model, newx=X, s=cv_en$lambda.min))
r2    <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)
r_cor <- cor(y, y_hat, use="complete.obs")
cat(sprintf("\nIn-sample  R² = %.3f   r = %.3f\n", r2, r_cor))
cat(sprintf("Non-zero coefficients: %d\n", nrow(coefs_nz)))
cat("\nCoefficients:\n"); print(round(coefs_nz, 4))

# bootstrap stability
set.seed(99)
boot_feats <- replicate(50, {
  idx <- sample(nrow(X), replace=TRUE)
  cv  <- cv.glmnet(X[idx,], y[idx], alpha=0.5, nfolds=5)
  m   <- glmnet(X[idx,], y[idx], alpha=0.5, lambda=cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc != 0)]
}, simplify=FALSE)
freq_df <- sort(table(unlist(boot_feats)), decreasing=TRUE)
freq_df <- freq_df[names(freq_df) != "(Intercept)"]

# ── PLOTS ──────────────────────────────────────────────────────────────────────
# Plot 8a: CV curve
png(file.path(out_dir, "plot8a_elasticnet_cv.png"), width=700, height=500, type="cairo")
plot(cv_en, main="ElasticNet CV  (alpha=0.5, 10-fold)")
dev.off()

# Plot 8b: Coefficients
png(file.path(out_dir, "plot8b_elasticnet_coefs.png"), width=850, height=550, type="cairo")
par(mar=c(5,12,3,2))
if (nrow(coefs_nz) > 0) {
  ord <- nrow(coefs_nz):1
  barplot(coefs_nz[ord, 1],
          names.arg = rownames(coefs_nz)[ord],
          horiz=TRUE, las=1,
          col=ifelse(coefs_nz[ord,1] > 0, "steelblue", "tomato"),
          main=sprintf("ElasticNet coefficients  (%d non-zero)   In-sample R²=%.3f",
                       nrow(coefs_nz), r2),
          xlab="Coefficient (z-scored features)")
  abline(v=0, col="grey50")
}
dev.off()

# Plot 8c: Bootstrap stability
n_show <- min(20, length(freq_df))
png(file.path(out_dir, "plot8c_bootstrap_stability.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
barplot(freq_df[n_show:1],
        names.arg=names(freq_df)[n_show:1],
        horiz=TRUE, las=1, col="steelblue",
        main="Bootstrap feature selection frequency (n=50)",
        xlab="Times selected out of 50")
abline(v=25, col="red", lty=2)
dev.off()

# Plot 8d: Predicted vs actual
png(file.path(out_dir, "plot8d_predicted_vs_actual.png"), width=600, height=600, type="cairo")
plot(y, y_hat, pch=16, cex=0.5,
     col=ifelse(is.finite(model_df$delay_green_to_red),
                rgb(0,0.4,0.8,0.4), rgb(0.7,0.7,0.7,0.4)),
     xlab="Actual delay (min;  grey=censored)",
     ylab="Predicted delay (min)",
     main=sprintf("Predicted vs actual green→red delay\nr = %.3f", r_cor))
abline(a=0, b=1, col="red", lty=2)
legend("topleft", bty="n",
       legend=c("Observed red","Censored (imputed)"),
       col=c(rgb(0,0.4,0.8), rgb(0.7,0.7,0.7)), pch=16)
dev.off()

# ══════════════════════════════════════════════════════════════════════════════
# COX ELASTICNET
# ══════════════════════════════════════════════════════════════════════════════
library(survival)
cat("\n── Cox ElasticNet ──\n")

# proper censoring: observed = delay_green_to_red, censored = track duration
track_dur <- vapply(split(spots, spots$Track.ID),
                    function(df) (max(df$T..sec.) - min(df$T..sec.)) / 60,
                    numeric(1))

cox_df <- model_df
cox_df$event     <- as.integer(is.finite(cox_df$delay_green_to_red))
cox_df$surv_time <- ifelse(
  cox_df$event == 1,
  cox_df$delay_green_to_red,
  track_dur[as.character(cox_df$Track.ID)]
)

# a handful of cells may have surv_time <= 0 due to single-frame tracks; remove
cox_df <- cox_df[is.finite(cox_df$surv_time) & cox_df$surv_time > 0, ]

cat(sprintf("Cox data: %d cells  |  %d events  |  %d censored\n",
            nrow(cox_df),
            sum(cox_df$event == 1),
            sum(cox_df$event == 0)))

# align X to cox_df rows
X_cox <- X[rownames(X) %in% rownames(cox_df), , drop=FALSE]
# rownames of X are 1:nrow(model_df); match by position via Track.ID
X_cox <- X[match(rownames(cox_df), rownames(model_df)), , drop=FALSE]

surv_y <- Surv(cox_df$surv_time, cox_df$event)

set.seed(42)
cv_cox   <- cv.glmnet(X_cox, surv_y, family="cox", alpha=0.5, nfolds=10)
cox_en   <- glmnet(X_cox, surv_y, family="cox", alpha=0.5, lambda=cv_cox$lambda.min)

cox_coefs    <- as.matrix(coef(cox_en))
cox_coefs_nz <- cox_coefs[cox_coefs[,1] != 0, , drop=FALSE]
cox_coefs_nz <- cox_coefs_nz[order(abs(cox_coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]

# concordance (C-statistic) — higher is better (0.5 = random, 1 = perfect)
risk_score <- as.numeric(predict(cox_en, newx=X_cox, s=cv_cox$lambda.min, type="link"))
conc       <- concordance(surv_y ~ risk_score)
cat(sprintf("Concordance (C-statistic): %.3f  (SE=%.3f)\n",
            conc$concordance, sqrt(conc$var)))
cat(sprintf("Non-zero Cox coefficients: %d\n", nrow(cox_coefs_nz)))
cat("\nCox coefficients (log-HR per SD):\n")
print(round(cox_coefs_nz, 4))

# bootstrap stability for Cox
set.seed(99)
boot_cox <- replicate(50, {
  idx <- sample(nrow(X_cox), replace=TRUE)
  cv  <- cv.glmnet(X_cox[idx,], surv_y[idx], family="cox", alpha=0.5, nfolds=5)
  m   <- glmnet(X_cox[idx,], surv_y[idx], family="cox", alpha=0.5, lambda=cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc != 0)]
}, simplify=FALSE)
cox_freq <- sort(table(unlist(boot_cox)), decreasing=TRUE)

# ── Plot C1: Cox CV curve ──────────────────────────────────────────────────────
png(file.path(out_dir, "plot_cox1_cv.png"), width=700, height=500, type="cairo")
plot(cv_cox,
     main=sprintf("Cox ElasticNet CV  (alpha=0.5, 10-fold)\nC-statistic=%.3f", conc$concordance))
dev.off()
cat("Saved plot_cox1_cv.png\n")

# ── Plot C2: Cox coefficients ─────────────────────────────────────────────────
png(file.path(out_dir, "plot_cox2_coefs.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
if (nrow(cox_coefs_nz) > 0) {
  ord <- nrow(cox_coefs_nz):1
  barplot(cox_coefs_nz[ord, 1],
          names.arg=rownames(cox_coefs_nz)[ord],
          horiz=TRUE, las=1,
          col=ifelse(cox_coefs_nz[ord,1] > 0, "tomato", "steelblue"),
          main=sprintf("Cox ElasticNet coefficients  (%d non-zero)\nlog-HR per SD  |  blue=faster to red, red=slower",
                       nrow(cox_coefs_nz)),
          xlab="log Hazard Ratio  (per SD of z-scored feature)")
  abline(v=0, col="grey50")
}
dev.off()
cat("Saved plot_cox2_coefs.png\n")

# ── Plot C3: Bootstrap stability ──────────────────────────────────────────────
n_show <- min(20, length(cox_freq))
png(file.path(out_dir, "plot_cox3_bootstrap.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
barplot(cox_freq[n_show:1],
        names.arg=names(cox_freq)[n_show:1],
        horiz=TRUE, las=1, col="steelblue",
        main="Cox bootstrap feature selection frequency (n=50)",
        xlab="Times selected out of 50")
abline(v=25, col="red", lty=2)
dev.off()
cat("Saved plot_cox3_bootstrap.png\n")

# ── Plot C4: KM curves — cross-validated risk groups ─────────────────────────
k_folds  <- 5
set.seed(42)
fold_ids <- sample(rep(1:k_folds, length.out=nrow(X_cox)))

cv_risk <- numeric(nrow(X_cox))
for (fold in seq_len(k_folds)) {
  test_idx  <- which(fold_ids == fold)
  train_idx <- which(fold_ids != fold)
  inner_cv  <- cv.glmnet(X_cox[train_idx,], surv_y[train_idx],
                          family="cox", alpha=0.5, nfolds=5)
  fm <- glmnet(X_cox[train_idx,], surv_y[train_idx],
                family="cox", alpha=0.5, lambda=inner_cv$lambda.min)
  cv_risk[test_idx] <- as.numeric(
    predict(fm, newx=X_cox[test_idx,], type="link"))
}

conc_cv   <- concordance(surv_y ~ cv_risk)
c_stat_cv <- max(conc_cv$concordance, 1 - conc_cv$concordance)
cat(sprintf("Cross-validated C-statistic: %.3f\n", c_stat_cv))

risk_group_cv <- ifelse(cv_risk > median(cv_risk), "High risk", "Low risk")
km_fit_cv     <- survfit(surv_y ~ risk_group_cv)
logrank       <- survdiff(surv_y ~ risk_group_cv)
p_val         <- 1 - pchisq(logrank$chisq, df=1)
cat(sprintf("Log-rank p = %.4f\n", p_val))

km_tbl  <- summary(km_fit_cv)$table
med_hi  <- km_tbl[grep("High", rownames(km_tbl)), "median"]
med_lo  <- km_tbl[grep("Low",  rownames(km_tbl)), "median"]

png(file.path(out_dir, "plot_cox4_km.png"), width=700, height=600, type="cairo")
plot(km_fit_cv,
     col=c("tomato","steelblue"), lwd=2,
     xlab="Time from green onset (min)", ylab="P(not yet red)",
     main=sprintf("KM curves — %d-fold CV risk groups  |  C=%.3f  |  log-rank p=%.3f",
                  k_folds, c_stat_cv, p_val))
legend("topright", bty="n",
       legend=c(sprintf("High risk  (n=%d, median=%.0f min)", sum(risk_group_cv=="High risk"), med_hi),
                sprintf("Low risk   (n=%d, median=%.0f min)", sum(risk_group_cv=="Low risk"),  med_lo)),
       col=c("tomato","steelblue"), lwd=2)
dev.off()
cat("Saved plot_cox4_km.png\n")

cat("\nAll done.\n")

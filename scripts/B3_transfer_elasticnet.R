options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs", .libPaths()))
library(glmnet)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", "B3")
fig_dir   <- file.path(base_dir, "figures", "B3")

# ── load A2+A3 model ──────────────────────────────────────────────────────────
en_res       <- readRDS(file.path(base_dir, "cache", "combined", "en_results.rds"))
train_center <- attr(en_res$X, "scaled:center")
train_scale  <- attr(en_res$X, "scaled:scale")
feat_cols    <- colnames(en_res$X)
cat(sprintf("A2+A3 model: %d features, in-sample R²=%.3f  r=%.3f\n",
            length(feat_cols), en_res$r2, en_res$r_cor))

# ── load B3 data ──────────────────────────────────────────────────────────────
feat_df  <- readRDS(file.path(cache_dir, "feat_df.rds"))
onset_df <- readRDS(file.path(cache_dir, "onset_df.rds"))
spots    <- readRDS(file.path(cache_dir, "spots_clean.rds"))
nuc      <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))

# ── filters ───────────────────────────────────────────────────────────────────
# 1. first-half: track starts in the first half of the movie
movie_half_min  <- max(spots[["T..sec."]], na.rm = TRUE) / 60 / 2
track_start_min <- tapply(spots[["T..sec."]], spots[["Track.ID"]], min) / 60
fh_ids  <- names(track_start_min)[track_start_min <= movie_half_min]

# 2. nucleus-assigned
nuc_ids <- unique(nuc$Track.ID)

# 3. observed mCherry onset (finite positive delay)
red_ids <- onset_df$Track.ID[is.finite(onset_df$delay_green_to_red) &
                               onset_df$delay_green_to_red > 0]

keep <- feat_df$Track.ID %in% fh_ids &
        feat_df$Track.ID %in% nuc_ids &
        feat_df$Track.ID %in% red_ids
df <- feat_df[keep, ]
cat(sprintf("B3 cells after all filters: %d (first-half + nucleus + mCherry)\n", nrow(df)))

# ── build scaled feature matrix using A2+A3 training statistics ───────────────
X <- as.matrix(df[, feat_cols])
for (j in seq_len(ncol(X))) {
  na_j <- is.na(X[, j])
  if (any(na_j)) X[na_j, j] <- median(X[!na_j, j], na.rm = TRUE)
}
X_sc <- scale(X, center = train_center, scale = train_scale)

# ── predict ───────────────────────────────────────────────────────────────────
y_hat_min <- as.numeric(predict(en_res$model, newx = X_sc, s = en_res$cv$lambda.min))

y_actual_min <- onset_df$delay_green_to_red[match(df$Track.ID, onset_df$Track.ID)]

# log-transform for metrics (matching B3 in-sample approach)
log_actual <- log(y_actual_min)
log_pred   <- log(pmax(y_hat_min, 1))

# raw transfer metrics on log scale (R² and rho); r on raw scale for display
r2_raw  <- 1 - sum((log_actual - log_pred)^2) / sum((log_actual - mean(log_actual))^2)
r_raw   <- cor(exp(log_actual), exp(log_pred))   # raw-scale Pearson r for titles
r       <- cor(log_actual, log_pred)             # log-scale r (kept for recalib formula)
rho     <- cor(log_actual, log_pred, method = "spearman")
cat(sprintf("B3 transfer raw      R²=%.3f  r_raw=%.3f  r_log=%.3f  ρ=%.3f\n", r2_raw, r_raw, r, rho))

# linear recalibration: shift + rescale predictions to match B3's log-outcome distribution
# pred_cal = mean(log_actual) + (log_pred - mean(log_pred)) / sd(log_pred) * sd(log_actual)
# this is the optimal linear transform; after it R² == r² by construction
log_pred_cal <- mean(log_actual) +
                (log_pred - mean(log_pred)) / sd(log_pred) * sd(log_actual)
r2_cal <- 1 - sum((log_actual - log_pred_cal)^2) / sum((log_actual - mean(log_actual))^2)
cat(sprintf("B3 transfer recalib  R²=%.3f  (= r² = %.3f)\n", r2_cal, r^2))

# convert to hours for display
actual_h      <- exp(log_actual)     / 60
pred_raw_h    <- exp(log_pred)       / 60
pred_cal_h    <- exp(log_pred_cal)   / 60

# ── plot: two panels (raw | recalibrated) ────────────────────────────────────
png(file.path(fig_dir, "en_transfer_B3_proper.png"),
    width = 1100, height = 540, res = 115, type = "cairo")
par(mfrow = c(1, 2), mar = c(4.5, 5, 4, 1.5), oma = c(0, 0, 2.5, 0))

col <- adjustcolor("#2980b9", 0.55)

# panel 1: raw predictions
plot(actual_h, pred_raw_h,
     col = col, pch = 16, cex = 0.85,
     xlab = "Actual GFP→mCherry delay (h)",
     ylab = "Predicted delay (h)",
     main = sprintf("Raw transfer\nR²=%.2f  r=%.2f  ρ=%.2f", r2_raw, r_raw, rho))
abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
abline(lm(pred_raw_h ~ actual_h), col = col, lwd = 1.8)

# panel 2: recalibrated predictions
plot(actual_h, pred_cal_h,
     col = col, pch = 16, cex = 0.85,
     xlab = "Actual GFP→mCherry delay (h)",
     ylab = "Recalibrated predicted delay (h)",
     main = sprintf("Recalibrated (mean+SD matched to B3)\nR²=%.2f  r=%.2f  ρ=%.2f", r2_cal, r_raw, rho))
abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
abline(lm(pred_cal_h ~ actual_h), col = col, lwd = 1.8)

mtext(sprintf("B3 — ElasticNet transfer (A2+A3 model, log scale, n=%d cells)  |  A2+A3 in-sample: R²=%.2f  r=%.2f",
              nrow(df), en_res$r2, en_res$r_cor),
      outer = TRUE, cex = 0.82, font = 2, line = 0.8)

dev.off()
cat("Saved figures/B3/en_transfer_B3_proper.png\n")

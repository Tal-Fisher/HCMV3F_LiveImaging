options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs", .libPaths()))
library(glmnet)

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"
fig_dir  <- file.path(base_dir, "figures", "B3")

feat_df  <- readRDS(file.path(base_dir, "cache", "B3", "feat_df.rds"))
onset_df <- readRDS(file.path(base_dir, "cache", "B3", "onset_df.rds"))
onset_df$delay_blue_to_red <- onset_df$red_onset_min - onset_df$blue_onset_min
en_ref    <- readRDS(file.path(base_dir, "cache", "combined", "en_results.rds"))
feat_cols <- intersect(rownames(en_ref$coefs_nz), names(feat_df))

# ── first-half filter (same as A2+A3 pipeline) ────────────────────────────────
spots    <- readRDS(file.path(base_dir, "cache", "B3", "spots_clean.rds"))
movie_max_min  <- max(spots[["T..sec."]], na.rm = TRUE) / 60   # 4320 min = 72 h
movie_half_min <- movie_max_min / 2                             # 2160 min = 36 h
track_start_min <- tapply(spots[["T..sec."]], spots[["Track.ID"]], min) / 60
first_half_ids  <- names(track_start_min)[track_start_min <= movie_half_min]
cat(sprintf("First-half filter: %d / %d tracks kept\n",
            length(first_half_ids), length(track_start_min)))
feat_df <- feat_df[feat_df$Track.ID %in% first_half_ids, ]

feat_labels <- c(
  nuc_bfp_mean    = "Nuclear BFP mean",
  nuc_bfp_start   = "Nuclear BFP start",
  nuc_bfp_sd      = "Nuclear BFP SD",
  nuc_bfp_slope   = "Nuclear BFP slope",
  nuc_area_mean   = "Nucleus area mean",
  nuc_area_slope  = "Nucleus area slope",
  nuc_circ_mean   = "Nucleus circularity",
  nuc_ratio_mean  = "Nuc/cell area ratio",
  gfp_corr_start  = "GFP start",
  gfp_corr_slope  = "GFP slope",
  gfp_corr_sd     = "GFP SD",
  area_mean       = "Cell area mean",
  solidity_sd     = "Cell solidity SD",
  gfp_snr_sd      = "GFP SNR SD"
)

COLS <- c("green-to-red" = "#27ae60", "blue-to-red" = "#2980b9")

fit_en <- function(df, col) {
  df <- df[is.finite(df[[col]]) & df[[col]] > 0, ]
  X  <- as.matrix(df[, feat_cols])
  for (j in seq_len(ncol(X))) {
    na_j <- is.na(X[, j])
    if (any(na_j)) X[na_j, j] <- median(X[!na_j, j])
  }
  X_sc <- scale(X)
  y    <- log(df[[col]])
  set.seed(42)
  cv   <- cv.glmnet(X_sc, y, alpha = 0.5, nfolds = min(10, nrow(df)))
  pred <- as.numeric(predict(cv, newx = X_sc, s = "lambda.min"))
  coef_vec <- as.numeric(coef(cv, s = "lambda.min"))[-1]
  names(coef_vec) <- feat_cols
  list(y = y, pred = pred, coef = coef_vec, df = df, col = col,
       r2  = 1 - sum((y - pred)^2) / sum((y - mean(y))^2),
       r   = cor(y, pred),
       rho = cor(y, pred, method = "spearman"))
}

res_g2r <- fit_en(merge(feat_df, onset_df[, c("Track.ID","delay_green_to_red")], by="Track.ID"),
                  "delay_green_to_red")

# ── plot: 1×2  (scatter | coef bar) ──────────────────────────────────────────
png(file.path(fig_dir, "elasticnet_B3_insample.png"),
    width = 1100, height = 500, res = 115, type = "cairo")
par(mfrow = c(1, 2), mar = c(4.5, 5, 4, 1.5), oma = c(0, 0, 2.5, 0))

res   <- res_g2r
col   <- COLS["green-to-red"]

actual_h <- exp(res$y)   / 60
pred_h   <- exp(res$pred) / 60

# scatter
plot(actual_h, pred_h,
     col  = adjustcolor(col, 0.55), pch = 16, cex = 0.85,
     ylim = c(15, 35),
     xlab = "Actual GFP→mCherry delay (h)",
     ylab = "Predicted delay (h)",
     main = sprintf("green-to-red  (n=%d)\nR²=%.2f   r=%.2f   ρ=%.2f",
                    length(res$y), res$r2, res$r, res$rho))
abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
abline(lm(pred_h ~ actual_h), col = col, lwd = 1.8)

# top 5 coefficients
cv   <- res$coef
top5 <- names(sort(abs(cv), decreasing = TRUE))[1:5]
vals <- rev(cv[top5])
labs <- rev(ifelse(is.na(feat_labels[top5]), top5, feat_labels[top5]))
bar_cols <- ifelse(vals >= 0, adjustcolor(col, 0.85), adjustcolor("grey40", 0.7))
xlim_c <- max(abs(vals)) * 1.35 * c(-1, 1)

bp <- barplot(vals, horiz = TRUE, names.arg = labs,
              las = 1, col = bar_cols, border = NA,
              xlim = xlim_c,
              xlab = "Coefficient (scaled features, log outcome)",
              main = "Top 5 features — green-to-red",
              cex.names = 0.82, cex.axis = 0.8)
abline(v = 0, col = "grey30", lwd = 1)
text(vals + sign(vals) * diff(xlim_c) * 0.03, bp,
     labels = sprintf("%.3f", vals), cex = 0.72,
     adj = ifelse(vals >= 0, 0, 1))

mtext("B3 — ElasticNet in-sample (α=0.5, CV λ, 14 EN-selected features, log outcome, first-half filter)",
      outer = TRUE, cex = 0.88, font = 2, line = 0.8)
dev.off()
cat("Saved figures/B3/elasticnet_B3_insample.png\n")

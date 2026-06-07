options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs",
            .libPaths()))
library(glmnet)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
fig_dir   <- file.path(base_dir, "figures", "B3")

# ── load A2+A3 ElasticNet model ────────────────────────────────────────────────
en_res <- readRDS(file.path(base_dir, "cache", "combined", "en_results.rds"))
cat(sprintf("A2+A3 model — in-sample R²=%.3f  r=%.3f\n", en_res$r2, en_res$r_cor))

# scaling parameters from training data
train_center <- attr(en_res$X, "scaled:center")
train_scale  <- attr(en_res$X, "scaled:scale")
feat_cols    <- names(train_center)
cat(sprintf("Model features: %d\n", length(feat_cols)))

# ── load B3 features + compute blue-to-red delay ──────────────────────────────
feat_df  <- readRDS(file.path(base_dir, "cache", "B3", "feat_df.rds"))
onset_df <- readRDS(file.path(base_dir, "cache", "B3", "onset_df.rds"))

onset_df$delay_blue_to_red <- onset_df$red_onset_min - onset_df$blue_onset_min

# keep cells with positive blue-to-red delay AND features
df <- merge(feat_df,
            onset_df[, c("Track.ID", "delay_blue_to_red", "delay_green_to_red")],
            by = "Track.ID")
df <- df[!is.na(df$delay_blue_to_red) & df$delay_blue_to_red > 0, ]
cat(sprintf("\nB3 cells with features + positive blue-to-red delay: %d\n", nrow(df)))
cat(sprintf("Blue-to-red delay: median=%.0f min (%.1f h)  range=%.0f–%.0f min\n",
            median(df$delay_blue_to_red),
            median(df$delay_blue_to_red) / 60,
            min(df$delay_blue_to_red), max(df$delay_blue_to_red)))

# ── scale B3 features using A2+A3 training parameters ─────────────────────────
feat_cols_b3 <- intersect(feat_cols, names(df))
cat(sprintf("Features available in B3: %d / %d\n", length(feat_cols_b3), length(feat_cols)))

X_b3 <- as.matrix(df[, feat_cols_b3])
# impute any NA with column median
for (j in seq_len(ncol(X_b3))) {
  na_j <- is.na(X_b3[, j])
  if (any(na_j)) X_b3[na_j, j] <- median(X_b3[!na_j, j])
}
X_b3_sc <- scale(X_b3,
                 center = train_center[feat_cols_b3],
                 scale  = train_scale[feat_cols_b3])

# ── predict blue-to-red using the green-to-red model ──────────────────────────
pred_b2r <- as.numeric(predict(en_res$model, newx = X_b3_sc, s = en_res$cv$lambda.min))
actual    <- df$delay_blue_to_red

r2  <- 1 - sum((actual - pred_b2r)^2)  / sum((actual - mean(actual))^2)
r   <- cor(actual, pred_b2r)
rho <- cor(actual, pred_b2r, method = "spearman")
mae <- mean(abs(actual - pred_b2r))
med_ae <- median(abs(actual - pred_b2r))

cat(sprintf("\n── Transfer: A2+A3 green-to-red model → B3 blue-to-red ──\n"))
cat(sprintf("n=%d  R²=%.3f  r=%.3f  ρ=%.3f\n", nrow(df), r2, r, rho))
cat(sprintf("MAE=%.0f min (%.1f h)   MedAE=%.0f min (%.1f h)\n",
            mae, mae/60, med_ae, med_ae/60))

# also compare green-to-red actual vs blue-to-red actual
gtr_sub <- df$delay_green_to_red[is.finite(df$delay_green_to_red)]
cat(sprintf("\nFor reference (same %d cells):\n", nrow(df)))
cat(sprintf("  Median green-to-red: %.0f min (%.1f h)\n",
            median(gtr_sub), median(gtr_sub)/60))
cat(sprintf("  Median blue-to-red:  %.0f min (%.1f h)\n",
            median(actual), median(actual)/60))

# ── plot ───────────────────────────────────────────────────────────────────────
actual_h <- actual     / 60
pred_h   <- pred_b2r   / 60
lim <- range(c(actual_h, pred_h), finite = TRUE)
lim <- lim + diff(lim) * c(-0.05, 0.05)

png(file.path(fig_dir, "en_transfer_blue_to_red.png"),
    width = 600, height = 560, res = 110, type = "cairo")
par(mar = c(5, 5, 4, 2))

plot(actual_h, pred_h,
     col  = adjustcolor("#8e44ad", 0.55),
     pch  = 16, cex = 0.85,
     xlim = lim, ylim = lim,
     xlab = "Actual BFP → mCherry delay (h)",
     ylab = "Predicted delay (h)\n[A2+A3 GFP→mCherry model]",
     main = sprintf("A2+A3 ElasticNet → B3 blue-to-red\n(n=%d  R²=%.2f  r=%.2f  ρ=%.2f)",
                    nrow(df), r2, r, rho))
abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
abline(lm(pred_h ~ actual_h), col = "#8e44ad", lwd = 1.8)
legend("topleft", bty = "n",
       legend = c(sprintf("MAE = %.0f min (%.1f h)", mae, mae/60),
                  sprintf("MedAE = %.0f min (%.1f h)", med_ae, med_ae/60)),
       cex = 0.85)
dev.off()
cat(sprintf("\nSaved figures/B3/en_transfer_blue_to_red.png\n"))

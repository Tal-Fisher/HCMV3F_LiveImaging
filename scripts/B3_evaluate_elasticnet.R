options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
fig_dir   <- file.path(base_dir, "figures", "B3")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── load combined A2+A3 model ─────────────────────────────────────────────────
en_res       <- readRDS(file.path(base_dir, "cache/combined/en_results.rds"))
train_center <- attr(en_res$X, "scaled:center")
train_scale  <- attr(en_res$X, "scaled:scale")
feat_cols    <- colnames(en_res$X)   # exact columns the model was trained on

# in-sample combined CV performance (for comparison)
cat(sprintf("Combined A2+A3 model  — in-sample R²=%.3f  r=%.3f\n",
            en_res$r2, en_res$r_cor))
cat(sprintf("Features used in training: %d\n\n", length(feat_cols)))

# ── load B3 feature matrix ────────────────────────────────────────────────────
model_df_b3 <- readRDS(file.path(base_dir, "cache/B3/model_df.rds"))
cat(sprintf("B3 feature matrix: %d cells x %d features\n",
            nrow(model_df_b3), ncol(model_df_b3) - 2))  # -2: Track.ID, delay

# check column alignment
missing_cols <- setdiff(feat_cols, colnames(model_df_b3))
if (length(missing_cols) > 0)
  stop(sprintf("Missing B3 feature columns: %s", paste(missing_cols, collapse=", ")))

# ── build scaled B3 feature matrix using TRAINING scaling parameters ──────────
X_b3_raw <- as.matrix(model_df_b3[, feat_cols])

# median-impute per column (using B3's own medians; column is NaN → impute 0)
for (j in seq_len(ncol(X_b3_raw))) {
  med_j <- median(X_b3_raw[, j], na.rm = TRUE)
  X_b3_raw[is.na(X_b3_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
}

# apply training center/scale (not B3's own — this is transfer, not re-fitting)
X_b3 <- scale(X_b3_raw, center = train_center, scale = train_scale)

# ── predict ───────────────────────────────────────────────────────────────────
y_hat <- as.numeric(predict(en_res$model, newx = X_b3, s = en_res$cv$lambda.min))

# outcome: finite delay = productive cell; Inf = censored (never went red)
y_actual  <- model_df_b3$delay_green_to_red
prod_mask <- is.finite(y_actual)

# for censored cells use same convention as training: max_train * 1.1
max_train <- max(en_res$y[is.finite(en_res$y)])
y_obs     <- ifelse(prod_mask, y_actual, max_train * 1.1)

# ── performance metrics ───────────────────────────────────────────────────────
r2_all   <- 1 - sum((y_obs  - y_hat)^2)          / sum((y_obs  - mean(y_obs))^2)
r_all    <- cor(y_obs,  y_hat)

yp <- y_obs[prod_mask];  yh <- y_hat[prod_mask]
r2_prod  <- 1 - sum((yp - yh)^2) / sum((yp - mean(yp))^2)
r_prod   <- cor(yp, yh)
rho_prod <- cor(yp, yh, method = "spearman")
mae_prod <- mean(abs(yp - yh))
med_ae   <- median(abs(yp - yh))
rmse_prod <- sqrt(mean((yp - yh)^2))

cat(sprintf("── B3 transfer evaluation ──────────────────────────────────\n"))
cat(sprintf("All cells  (n=%3d)  R²=%.3f   r=%.3f\n",
            nrow(model_df_b3), r2_all, r_all))
cat(sprintf("Red cells  (n=%3d)  R²=%.3f   r=%.3f   ρ=%.3f\n",
            sum(prod_mask), r2_prod, r_prod, rho_prod))
cat(sprintf("                    MAE=%.0f min   MedAE=%.0f min   RMSE=%.0f min\n",
            mae_prod, med_ae, rmse_prod))
cat(sprintf("(Reference: combined A2+A3 in-sample R²=%.3f  r=%.3f)\n",
            en_res$r2, en_res$r_cor))

# ── scatter plot ──────────────────────────────────────────────────────────────
col_red  <- rgb(0,  0.4, 0.8, 0.5)
col_grey <- rgb(0.7, 0.7, 0.7, 0.4)

ax_range <- range(c(y_obs, y_hat), na.rm = TRUE)

png(file.path(fig_dir, "en_transfer_predicted_vs_actual.png"),
    width = 680, height = 680, type = "cairo")
par(mar = c(5, 5, 5, 2))

plot(y_obs, y_hat,
     pch  = 16, cex  = 0.65,
     col  = ifelse(prod_mask, col_red, col_grey),
     xlim = ax_range, ylim = ax_range,
     xlab = "Actual delay — track start to mCherry onset (min)",
     ylab = "Predicted delay — combined A2+A3 ElasticNet model (min)",
     main = sprintf(
       "B3  —  ElasticNet transfer  (model trained on combined A2+A3)\n\
Red cells (n=%d):  R²=%.3f   r=%.3f   ρ=%.3f\n\
MAE=%.0f min   MedAE=%.0f min   RMSE=%.0f min",
       sum(prod_mask), r2_prod, r_prod, rho_prod,
       mae_prod, med_ae, rmse_prod))

abline(a = 0, b = 1, col = "red", lty = 2, lwd = 1.5)

# reference: combined training CV performance as subtitle annotation
mtext(sprintf("(A2+A3 in-sample: R²=%.3f  r=%.3f)", en_res$r2, en_res$r_cor),
      side = 1, line = 4, cex = 0.75, col = "grey40")

legend("topleft", bty = "n",
       legend = c(sprintf("Observed red  (n=%d)", sum(prod_mask)),
                  sprintf("Censored        (n=%d)", sum(!prod_mask))),
       col = c(rgb(0, 0.4, 0.8), rgb(0.6, 0.6, 0.6)),
       pch = 16, pt.cex = 0.9)
dev.off()
cat("Saved figures/B3/en_transfer_predicted_vs_actual.png\n")

# ── save results ──────────────────────────────────────────────────────────────
saveRDS(
  list(y_actual   = y_actual,
       y_obs      = y_obs,
       y_hat      = y_hat,
       prod_mask  = prod_mask,
       r2_all     = r2_all,   r_all     = r_all,
       r2_prod    = r2_prod,  r_prod    = r_prod,
       rho_prod   = rho_prod,
       mae_prod   = mae_prod, med_ae    = med_ae,  rmse_prod = rmse_prod,
       feat_cols  = feat_cols,
       train_dataset = "combined"),
  file.path(base_dir, "cache/B3/en_transfer_results.rds"))
cat("Saved cache/B3/en_transfer_results.rds\n")

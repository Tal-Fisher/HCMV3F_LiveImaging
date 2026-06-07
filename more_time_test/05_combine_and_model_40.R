options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

# Combines A2 + A3 feature matrices produced by 04_features_40.R and runs:
#   1. Spearman feature correlation matrix (heatmap, clustered)
#   2. ElasticNet regression predicting delay_green_to_red (all cells, censored)
#   3. Binary ElasticNet predicting productive vs non-productive
#
# Cells that turned red within the first 40 frames are already absent from the
# per-dataset caches — they are excluded from both models.
#
# Usage: Rscript 05_combine_and_model_40.R

datasets   <- c("A2", "A3")
base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
mtt_dir    <- file.path(base_dir, "more_time_test")
cache_comb <- file.path(mtt_dir, "cache", "combined")
fig_dir    <- file.path(mtt_dir, "figures", "combined")
dir.create(cache_comb, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir,    recursive = TRUE, showWarnings = FALSE)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND COMBINE PER-DATASET CACHES
# ══════════════════════════════════════════════════════════════════════════════
load_mtt <- function(ds, file) {
  path <- file.path(mtt_dir, "cache", ds, file)
  if (!file.exists(path))
    stop(sprintf("Missing more_time_test/cache/%s/%s — run: Rscript 04_features_40.R %s",
                 ds, file, ds))
  readRDS(path)
}

feat_list  <- lapply(datasets, load_mtt, "feat_df.rds")
model_list <- lapply(datasets, load_mtt, "model_df.rds")

for (i in seq_along(datasets)) {
  ds <- datasets[i]
  feat_list[[i]]$Track.ID  <- paste0(ds, "_", feat_list[[i]]$Track.ID)
  model_list[[i]]$Track.ID <- paste0(ds, "_", model_list[[i]]$Track.ID)
  model_list[[i]]$dataset  <- ds
}

common_feat  <- Reduce(intersect, lapply(feat_list,  names))
common_model <- Reduce(intersect, lapply(model_list, names))
feat_df  <- do.call(rbind, lapply(feat_list,  function(d) d[, common_feat,  drop = FALSE]))
model_df <- do.call(rbind, lapply(model_list, function(d) d[, common_model, drop = FALSE]))

n_prod    <- sum(is.finite(model_df$delay_green_to_red))
n_nonprod <- sum(!is.finite(model_df$delay_green_to_red))
cat(sprintf("Combined (40-frame window): %d cells total\n", nrow(model_df)))
cat(sprintf("  productive (red after frame 40): %d\n", n_prod))
cat(sprintf("  non-productive (never red):      %d\n", n_nonprod))

# ── shared feature matrix prep ─────────────────────────────────────────────────
feat_cols <- setdiff(colnames(feat_df), "Track.ID")
feat_cols <- feat_cols[vapply(feat_cols,
  function(c) sum(is.finite(model_df[[c]])) >= 5, logical(1))]
cat(sprintf("Features used (>= 5 finite values): %d\n", length(feat_cols)))
cat("  Excluded (all/mostly missing):",
    setdiff(setdiff(colnames(feat_df), "Track.ID"), feat_cols), "\n")

X_raw <- as.matrix(model_df[, feat_cols])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[, j], na.rm = TRUE)
  X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
}
X_all <- scale(X_raw)

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
cat("\n── Feature correlation matrix ──\n")

# Include outcome as last variable; pairwise.complete.obs handles NA for non-productive.
# Features are from the imputed X_raw so no NAs; delay_green_to_red has NAs for
# non-productive cells, which are naturally excluded from feature-outcome pairs.
cor_data  <- cbind(X_raw, delay_green_to_red = model_df$delay_green_to_red)
cor_mat   <- cor(cor_data, method = "spearman", use = "pairwise.complete.obs")

# cluster all variables together so features that track the outcome appear near it
dist_mat  <- as.dist((1 - cor_mat) / 2)   # maps [-1,1] → [1,0]
hc        <- hclust(dist_mat, method = "complete")
ord       <- hc$order
cor_reord <- cor_mat[ord, ord]
n_var     <- ncol(cor_reord)

pal      <- colorRampPalette(c("steelblue", "white", "tomato"))(201)
img_size <- max(950, n_var * 28 + 280)

png(file.path(fig_dir, "feature_correlation_matrix.png"),
    width = img_size + 130, height = img_size, type = "cairo")

# main heatmap
par(fig = c(0, 0.87, 0, 1), new = FALSE, mar = c(12, 12, 4, 1))
image(seq_len(n_var), seq_len(n_var),
      t(cor_reord)[, rev(seq_len(n_var))],
      col = pal, zlim = c(-1, 1),
      xaxt = "n", yaxt = "n", xlab = "", ylab = "",
      main = sprintf("Feature Spearman correlation — A2+A3 combined, 40-frame window (n=%d cells)",
                     nrow(model_df)))
axis(1, at = seq_len(n_var), labels = colnames(cor_reord),  las = 2, cex.axis = 0.65)
axis(2, at = seq_len(n_var), labels = rev(rownames(cor_reord)), las = 1, cex.axis = 0.65)
abline(h = seq_len(n_var) + 0.5, col = "white", lwd = 0.4)
abline(v = seq_len(n_var) + 0.5, col = "white", lwd = 0.4)

# colorbar
par(fig = c(0.89, 0.93, 0.12, 0.88), new = TRUE, mar = c(0, 0.5, 0, 2))
cb_y <- seq(-1, 1, length.out = 202)
image(1, cb_y[-1] - diff(cb_y)[1] / 2,
      matrix(cb_y[-1], nrow = 1),
      col = pal, xaxt = "n", xlab = "", ylab = "", yaxt = "n")
axis(4, at = c(-1, -0.5, 0, 0.5, 1), las = 1, cex.axis = 0.8)
mtext("ρ", side = 4, line = 2.5, cex = 0.85)

dev.off()
cat(sprintf("Saved more_time_test/figures/combined/feature_correlation_matrix.png\n"))

# ══════════════════════════════════════════════════════════════════════════════
# 3. ELASTICNET REGRESSION: predicting delay_green_to_red
# ══════════════════════════════════════════════════════════════════════════════
cat("\n── ElasticNet regression (delay_green_to_red, all cells) ──\n")

# Non-productive cells are included as censored observations (imputed to max*1.1).
max_obs  <- max(model_df$delay_green_to_red[is.finite(model_df$delay_green_to_red)], na.rm = TRUE)
model_df$y <- ifelse(is.finite(model_df$delay_green_to_red),
                     model_df$delay_green_to_red, max_obs * 1.1)
y <- model_df$y

set.seed(42)
cv_en    <- cv.glmnet(X_all, y, alpha = 0.5, nfolds = 10)
en_model <- glmnet(X_all, y, alpha = 0.5, lambda = cv_en$lambda.min)

coefs    <- as.matrix(coef(en_model))
coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop = FALSE]
coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing = TRUE), , drop = FALSE]
y_hat    <- as.numeric(predict(en_model, newx = X_all, s = cv_en$lambda.min))
r2       <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)
cat(sprintf("In-sample R² = %.3f  non-zero coefs = %d\n", r2, nrow(coefs_nz)))
print(round(coefs_nz, 4))

# 10-fold outer CV for honest out-of-sample R²
n_outer     <- 10
set.seed(42)
outer_folds <- sample(rep(seq_len(n_outer), length.out = nrow(X_all)))
cv_preds    <- numeric(length(y))
for (k in seq_len(n_outer)) {
  ti  <- which(outer_folds == k)
  tri <- which(outer_folds != k)
  icv <- cv.glmnet(X_all[tri,], y[tri], alpha = 0.5, nfolds = 5)
  fm  <- glmnet(X_all[tri,], y[tri], alpha = 0.5, lambda = icv$lambda.min)
  cv_preds[ti] <- as.numeric(predict(fm, newx = X_all[ti,], s = icv$lambda.min))
}
prod_mask  <- is.finite(model_df$delay_green_to_red)
r2_cv      <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
r_cv       <- cor(y, cv_preds, use = "complete.obs")
r2_cv_prod <- 1 - sum((y[prod_mask] - cv_preds[prod_mask])^2) /
                  sum((y[prod_mask] - mean(y[prod_mask]))^2)
r_cv_prod  <- cor(y[prod_mask], cv_preds[prod_mask], use = "complete.obs")
cat(sprintf("CV (%d-fold) R² = %.3f   r = %.3f\n", n_outer, r2_cv, r_cv))
cat(sprintf("CV prod-only  R² = %.3f   r = %.3f\n", r2_cv_prod, r_cv_prod))

# bootstrap stability (50 resamples)
set.seed(99)
boot_feats <- replicate(50, {
  idx <- sample(nrow(X_all), replace = TRUE)
  cv  <- cv.glmnet(X_all[idx,], y[idx], alpha = 0.5, nfolds = 5)
  m   <- glmnet(X_all[idx,], y[idx], alpha = 0.5, lambda = cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc != 0)]
}, simplify = FALSE)
freq_df <- sort(table(unlist(boot_feats)), decreasing = TRUE)
freq_df <- freq_df[names(freq_df) != "(Intercept)"]

# plots
png(file.path(fig_dir, "en_coefs.png"), width = 850, height = 550, type = "cairo")
par(mar = c(5, 12, 3, 2))
if (nrow(coefs_nz) > 0) {
  ord_c <- nrow(coefs_nz):1
  barplot(coefs_nz[ord_c, 1], names.arg = rownames(coefs_nz)[ord_c],
          horiz = TRUE, las = 1,
          col = ifelse(coefs_nz[ord_c, 1] > 0, "steelblue", "tomato"),
          main = sprintf("A2+A3 (40-frame) ElasticNet — delay regression\n%d coefs   CV R²=%.3f   prod-only CV R²=%.3f",
                         nrow(coefs_nz), r2_cv, r2_cv_prod),
          xlab = "Coefficient (z-scored features)   blue=longer delay")
  abline(v = 0, col = "grey50")
}
dev.off()

n_show <- min(20, length(freq_df))
png(file.path(fig_dir, "en_bootstrap.png"), width = 850, height = 550, type = "cairo")
par(mar = c(5, 13, 3, 2))
barplot(freq_df[n_show:1], names.arg = names(freq_df)[n_show:1],
        horiz = TRUE, las = 1, col = "steelblue",
        main = "A2+A3 (40-frame) — ElasticNet bootstrap stability (n=50)",
        xlab = "Times selected out of 50")
abline(v = 25, col = "red", lty = 2)
dev.off()

png(file.path(fig_dir, "en_predicted_vs_actual.png"), width = 600, height = 600, type = "cairo")
plot(y, cv_preds, pch = 16, cex = 0.5,
     col = ifelse(prod_mask, rgb(0, 0.4, 0.8, 0.4), rgb(0.7, 0.7, 0.7, 0.4)),
     xlab = "Actual delay (min;  grey = censored at max×1.1)",
     ylab = "Predicted delay (min)",
     main = sprintf("A2+A3 (40-frame) — %d-fold CV predicted vs actual\nCV R²=%.3f   prod-only CV R²=%.3f",
                    n_outer, r2_cv, r2_cv_prod))
abline(a = 0, b = 1, col = "red", lty = 2)
legend("topleft", bty = "n",
       legend = c("Observed red", "Censored (non-productive)"),
       col = c(rgb(0, 0.4, 0.8), rgb(0.7, 0.7, 0.7)), pch = 16)
dev.off()

cat("Saved en_coefs.png, en_bootstrap.png, en_predicted_vs_actual.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BINARY ELASTICNET: productive vs non-productive
# ══════════════════════════════════════════════════════════════════════════════
cat("\n── Binary ElasticNet: productive vs non-productive ──\n")

y_bin <- as.integer(is.finite(model_df$delay_green_to_red))
cat(sprintf("Productive: %d  Non-productive: %d  (class balance: %.1f%% productive)\n",
            sum(y_bin), sum(1 - y_bin), 100 * mean(y_bin)))

set.seed(42)
cv_bin  <- cv.glmnet(X_all, y_bin, family = "binomial", alpha = 0.5,
                     nfolds = 10, type.measure = "auc")
bin_mod <- glmnet(X_all, y_bin, family = "binomial", alpha = 0.5,
                  lambda = cv_bin$lambda.min)

bin_coefs    <- as.matrix(coef(bin_mod))
bin_coefs_nz <- bin_coefs[bin_coefs[,1] != 0 & rownames(bin_coefs) != "(Intercept)", , drop = FALSE]
bin_coefs_nz <- bin_coefs_nz[order(abs(bin_coefs_nz[,1]), decreasing = TRUE), , drop = FALSE]
cat(sprintf("Binary EN: inner CV AUC = %.3f   non-zero coefs = %d\n",
            max(cv_bin$cvm), nrow(bin_coefs_nz)))
print(round(bin_coefs_nz, 4))

# 5-fold outer CV for unbiased ROC
k_folds <- 5
set.seed(42)
fold_ids_bin <- sample(rep(seq_len(k_folds), length.out = nrow(X_all)))
cv_prob_bin  <- numeric(nrow(X_all))
for (fold in seq_len(k_folds)) {
  ti  <- which(fold_ids_bin == fold)
  tri <- which(fold_ids_bin != fold)
  icv <- cv.glmnet(X_all[tri,], y_bin[tri], family = "binomial",
                   alpha = 0.5, nfolds = 5, type.measure = "auc")
  fm  <- glmnet(X_all[tri,], y_bin[tri], family = "binomial",
                alpha = 0.5, lambda = icv$lambda.min)
  cv_prob_bin[ti] <- as.numeric(predict(fm, newx = X_all[ti,],
                                        type = "response", s = icv$lambda.min))
}

# manual ROC (no external packages required)
roc_data <- function(probs, labels) {
  thresholds <- sort(unique(probs), decreasing = TRUE)
  tpr <- fpr <- numeric(length(thresholds) + 1)
  for (k in seq_along(thresholds)) {
    pred     <- as.integer(probs >= thresholds[k])
    tpr[k+1] <- sum(pred == 1 & labels == 1) / max(sum(labels == 1), 1)
    fpr[k+1] <- sum(pred == 1 & labels == 0) / max(sum(labels == 0), 1)
  }
  tpr[1] <- fpr[1] <- 0
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  list(tpr = tpr, fpr = fpr, auc = abs(auc))
}
roc <- roc_data(cv_prob_bin, y_bin)
cat(sprintf("Binary outer CV AUC (5-fold, manual ROC): %.3f\n", roc$auc))

# plots
png(file.path(fig_dir, "binary_en.png"), width = 1100, height = 500, type = "cairo")
par(mfrow = c(1, 2), mar = c(5, 12, 3, 2), oma = c(0, 0, 2, 0))

if (nrow(bin_coefs_nz) > 0) {
  ord_b <- nrow(bin_coefs_nz):1
  barplot(bin_coefs_nz[ord_b, 1],
          names.arg = rownames(bin_coefs_nz)[ord_b],
          horiz = TRUE, las = 1,
          col = ifelse(bin_coefs_nz[ord_b, 1] > 0, "tomato", "steelblue"),
          main = sprintf("Binary EN (%d coefs)\ninner CV AUC = %.3f", nrow(bin_coefs_nz), max(cv_bin$cvm)),
          xlab = "Coefficient (z-scored)\npositive = more likely productive")
  abline(v = 0, col = "grey50")
}

par(mar = c(5, 5, 3, 2))
plot(roc$fpr, roc$tpr, type = "l", lwd = 2, col = "steelblue",
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = sprintf("Binary model ROC\n5-fold outer CV AUC = %.3f", roc$auc))
abline(0, 1, col = "grey60", lty = 2)
legend("bottomright", bty = "n",
       legend = sprintf("AUC = %.3f", roc$auc), col = "steelblue", lwd = 2)
mtext("Binary ElasticNet: productive vs non-productive (40-frame window)",
      outer = TRUE, cex = 1, font = 2)
dev.off()
cat("Saved more_time_test/figures/combined/binary_en.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE COMBINED CACHES
# ══════════════════════════════════════════════════════════════════════════════
saveRDS(feat_df,  file.path(cache_comb, "feat_df.rds"))
saveRDS(model_df, file.path(cache_comb, "model_df.rds"))
saveRDS(cor_mat,  file.path(cache_comb, "feature_correlation_matrix.rds"))
saveRDS(list(model     = en_model,  cv       = cv_en,
             X         = X_all,     y        = y,
             coefs_nz  = coefs_nz,  r2       = r2,
             r2_cv     = r2_cv,     r_cv     = r_cv,
             cv_preds  = cv_preds,
             r2_cv_prod = r2_cv_prod, r_cv_prod = r_cv_prod,
             freq      = freq_df),
        file.path(cache_comb, "en_results.rds"))
saveRDS(list(model        = bin_mod,       cv          = cv_bin,
             X            = X_all,         y_bin       = y_bin,
             coefs_nz     = bin_coefs_nz,
             auc_inner_cv = max(cv_bin$cvm),
             auc_outer_cv = roc$auc,
             cv_probs     = cv_prob_bin,
             roc          = roc),
        file.path(cache_comb, "binary_en_results.rds"))
cat("\nSaved more_time_test/cache/combined/:\n")
cat("  feat_df.rds, model_df.rds, feature_correlation_matrix.rds\n")
cat("  en_results.rds, binary_en_results.rds\n")

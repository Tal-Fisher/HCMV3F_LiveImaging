options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs",
            .libPaths()))
library(glmnet)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", "combined")
fig_dir   <- file.path(base_dir, "figures", "combined")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── load data (identical setup to previous scripts) ───────────────────────────
model_df <- readRDS(file.path(cache_dir, "model_df.rds"))
en_res   <- readRDS(file.path(cache_dir, "en_results.rds"))
meta     <- read.csv(file.path(base_dir, "Forecast", "cell_metadata.csv"),
                     stringsAsFactors = FALSE)

red_df <- model_df[is.finite(model_df$delay_green_to_red), ]
df     <- merge(red_df, meta[, c("Track.ID", "group")], by = "Track.ID")
df$group <- factor(df$group, levels = c("early", "medium", "late"))

en_feats <- rownames(en_res$coefs_nz)
en_feats <- intersect(en_feats, names(df))

X_raw <- as.matrix(df[, en_feats])
for (j in seq_len(ncol(X_raw))) {
  na_j <- is.na(X_raw[, j])
  if (any(na_j)) X_raw[na_j, j] <- median(X_raw[!na_j, j])
}
X_sc  <- scale(X_raw)
y_log <- log(df$delay_green_to_red)

# ── same 75/25 stratified split ───────────────────────────────────────────────
set.seed(42)
train_idx <- unlist(lapply(levels(df$group), function(g) {
  idx <- which(df$group == g)
  sample(idx, size = floor(0.75 * length(idx)))
}))
test_idx <- setdiff(seq_len(nrow(df)), train_idx)

X_tr <- X_sc[train_idx, ];  X_te <- X_sc[test_idx, ]
y_tr <- y_log[train_idx];   y_te <- y_log[test_idx]
grp_tr <- df$group[train_idx]
grp_te <- df$group[test_idx]

cat(sprintf("Train: %d  |  Test: %d\n", length(train_idx), length(test_idx)))

# ── Stage 1: multinomial ElasticNet classifier (features → group) ─────────────
cat("\n--- Stage 1: multinomial classifier ---\n")
set.seed(42)
cv_s1 <- cv.glmnet(X_tr, grp_tr, family = "multinomial",
                   alpha = 0.5, nfolds = 10, type.measure = "class")
cat(sprintf("Best lambda: %.4f  |  CV misclassification: %.3f\n",
            cv_s1$lambda.min,
            min(cv_s1$cvm)))

# training classification accuracy
pred_grp_tr <- as.character(predict(cv_s1, newx = X_tr,
                                    s = "lambda.min", type = "class"))
acc_tr <- mean(pred_grp_tr == as.character(grp_tr))
cat(sprintf("Train accuracy: %.1f%%\n", acc_tr * 100))
cat("Train confusion matrix:\n")
print(table(predicted = pred_grp_tr, actual = grp_tr))

# test classification
pred_grp_te <- as.character(predict(cv_s1, newx = X_te,
                                    s = "lambda.min", type = "class"))
acc_te <- mean(pred_grp_te == as.character(grp_te))
cat(sprintf("Test accuracy: %.1f%%\n", acc_te * 100))
cat("Test confusion matrix:\n")
print(table(predicted = pred_grp_te, actual = grp_te))

# posterior probabilities for test cells
prob_te <- predict(cv_s1, newx = X_te, s = "lambda.min", type = "response")
prob_te <- prob_te[, , 1]   # drop extra dimension; rows=cells, cols=groups
colnames(prob_te) <- levels(df$group)

# ── Stage 2: per-group ElasticNet regressors ──────────────────────────────────
cat("\n--- Stage 2: per-group regressors ---\n")
reg_models <- list()
for (g in levels(df$group)) {
  idx_g <- which(grp_tr == g)
  set.seed(42)
  cv_g <- cv.glmnet(X_tr[idx_g, ], y_tr[idx_g],
                    alpha = 0.5, nfolds = min(10, length(idx_g)))
  r2_g <- 1 - min(cv_g$cvm) / var(y_tr[idx_g])
  cat(sprintf("  %s  n=%d  CV R²≈%.3f\n", g, length(idx_g), r2_g))
  reg_models[[g]] <- cv_g
}

# ── Test-set prediction: P(group) × ŷ_group, summed over groups ───────────────
pred_each <- sapply(levels(df$group), function(g) {
  as.numeric(predict(reg_models[[g]], newx = X_te, s = "lambda.min"))
})                                    # n_test × 3 matrix of log-delay predictions

# weighted combination
pred_te_log   <- rowSums(prob_te * pred_each)   # soft: weight by P(group)
pred_te_hard  <- pred_each[cbind(               # hard: use Stage 1 class label
  seq_len(nrow(X_te)),
  match(pred_grp_te, levels(df$group)))]

# ── metrics ───────────────────────────────────────────────────────────────────
cat("\n--- Test-set metrics ---\n")
for (label in c("soft (weighted)", "hard (class label)")) {
  p  <- if (label == "soft (weighted)") pred_te_log else pred_te_hard
  r2 <- 1 - sum((y_te - p)^2) / sum((y_te - mean(y_te))^2)
  r  <- cor(y_te, p)
  mae <- mean(abs(exp(y_te) - exp(p)))
  cat(sprintf("  %-22s  R²=%.3f  r=%.3f  MAE=%.0f min\n", label, r2, r, mae))
}

cat("\nPer-group test metrics (soft prediction, true group label):\n")
for (g in levels(df$group)) {
  m   <- grp_te == g
  r2g <- 1 - sum((y_te[m] - pred_te_log[m])^2) /
             sum((y_te[m] - mean(y_te[m]))^2)
  rg  <- cor(y_te[m], pred_te_log[m])
  cat(sprintf("  %-7s n=%-3d  R²=%.3f  r=%.3f\n", g, sum(m), r2g, rg))
}

# ── plot ───────────────────────────────────────────────────────────────────────
GROUP_COLS <- c(early = "#e67e22", medium = "#2980b9", late = "#27ae60")
actual_h   <- exp(y_te) / 60
pred_soft_h <- exp(pred_te_log) / 60
all_lim    <- range(c(actual_h, pred_soft_h), finite = TRUE) * c(0.9, 1.1)

png(file.path(fig_dir, "twostage_en_cv.png"),
    width = 1200, height = 430, res = 115, type = "cairo")
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 4, 1.5), oma = c(0, 0, 2.5, 0))

for (g in levels(df$group)) {
  mask <- grp_te == g
  x_g  <- actual_h[mask]
  y_g  <- pred_soft_h[mask]
  r2_g <- 1 - sum((log(x_g) - log(y_g))^2) / sum((log(x_g) - mean(log(x_g)))^2)
  r_g  <- cor(log(x_g), log(y_g))
  col  <- GROUP_COLS[g]

  plot(x_g, y_g,
       col  = adjustcolor(col, alpha.f = 0.6),
       pch  = 16, cex = 0.85,
       xlim = all_lim, ylim = all_lim,
       xlab = "Actual delay (h)", ylab = "Predicted delay (h)",
       main = sprintf("%s  (n=%d)\nR²=%.2f   r=%.2f", g, sum(mask), r2_g, r_g))
  abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
  if (sum(mask) > 2)
    abline(lm(y_g ~ x_g), col = col, lwd = 1.8)
}

r2_ov <- 1 - sum((y_te - pred_te_log)^2) / sum((y_te - mean(y_te))^2)
r_ov  <- cor(y_te, pred_te_log)
mtext(sprintf("Two-stage EN (Stage1: multinomial classifier → Stage2: per-group regressor)  |  overall R²=%.2f  r=%.2f",
              r2_ov, r_ov),
      outer = TRUE, cex = 0.82, font = 2, line = 0.8)
dev.off()
cat(sprintf("\nSaved figures/combined/twostage_en_cv.png\n"))

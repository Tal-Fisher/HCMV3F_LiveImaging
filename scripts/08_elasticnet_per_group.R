options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs",
            .libPaths()))

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", "combined")
fig_dir   <- file.path(base_dir, "figures", "combined")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

library(glmnet)

# ── load data (identical setup to 07_flexmix_cv.R) ────────────────────────────
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

# ── same 75/25 stratified split as flexmix CV ─────────────────────────────────
set.seed(42)
train_idx <- unlist(lapply(levels(df$group), function(g) {
  idx <- which(df$group == g)
  sample(idx, size = floor(0.75 * length(idx)))
}))
test_idx <- setdiff(seq_len(nrow(df)), train_idx)

cat(sprintf("Train: %d  |  Test: %d\n", length(train_idx), length(test_idx)))
print(table(df$group[train_idx]))
print(table(df$group[test_idx]))

# ── fit separate ElasticNet per group ─────────────────────────────────────────
GROUP_COLS <- c(early = "#e67e22", medium = "#2980b9", late = "#27ae60")

results <- list()
for (g in c("early", "medium", "late")) {
  tr <- train_idx[df$group[train_idx] == g]
  te <- test_idx[df$group[test_idx]   == g]

  X_tr <- X_sc[tr, ];  y_tr <- y_log[tr]
  X_te <- X_sc[te, ];  y_te <- y_log[te]

  set.seed(42)
  cv_fit <- cv.glmnet(X_tr, y_tr, alpha = 0.5, nfolds = min(10, length(tr)))
  pred_tr <- as.numeric(predict(cv_fit, newx = X_tr, s = "lambda.min"))
  pred_te <- as.numeric(predict(cv_fit, newx = X_te, s = "lambda.min"))

  r2_tr <- 1 - sum((y_tr - pred_tr)^2) / sum((y_tr - mean(y_tr))^2)
  r2_te <- 1 - sum((y_te - pred_te)^2) / sum((y_te - mean(y_te))^2)
  r_tr  <- cor(y_tr, pred_tr)
  r_te  <- cor(y_te, pred_te)
  mae_te <- mean(abs(exp(y_te) - exp(pred_te)))

  nz <- sum(coef(cv_fit, s = "lambda.min")[-1] != 0)
  cat(sprintf("\n%s (train n=%d, test n=%d)  non-zero features: %d\n",
              g, length(tr), length(te), nz))
  cat(sprintf("  Train: R²=%.3f  r=%.3f\n", r2_tr, r_tr))
  cat(sprintf("  Test:  R²=%.3f  r=%.3f  MAE=%.0f min\n", r2_te, r_te, mae_te))

  results[[g]] <- list(
    model   = cv_fit,
    tr_idx  = tr, te_idx = te,
    pred_te = pred_te, y_te = y_te,
    r2_tr = r2_tr, r_tr = r_tr,
    r2_te = r2_te, r_te = r_te,
    mae_te = mae_te
  )
}

# ── plot: predicted vs actual per group (same layout as flexmix CV plot) ───────
actual_h_all <- exp(y_log) / 60
all_lim <- range(actual_h_all, finite = TRUE) * c(0.9, 1.1)

png(file.path(fig_dir, "elasticnet_per_group_cv.png"),
    width = 1200, height = 430, res = 115, type = "cairo")
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 4, 1.5), oma = c(0, 0, 2.5, 0))

for (g in c("early", "medium", "late")) {
  res    <- results[[g]]
  actual <- exp(res$y_te) / 60
  pred   <- exp(res$pred_te) / 60
  col    <- GROUP_COLS[g]

  plot(actual, pred,
       col  = adjustcolor(col, alpha.f = 0.6),
       pch  = 16, cex = 0.85,
       xlim = all_lim, ylim = all_lim,
       xlab = "Actual delay (h)", ylab = "Predicted delay (h)",
       main = sprintf("%s  (n=%d)\nR²=%.2f   r=%.2f",
                      g, length(actual), res$r2_te, res$r_te))
  abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
  if (length(actual) > 2)
    abline(lm(pred ~ actual), col = col, lwd = 1.8)
}

mtext("ElasticNet per group (α=0.5, CV λ) — same 75/25 stratified split",
      outer = TRUE, cex = 0.88, font = 2, line = 0.8)
dev.off()
cat(sprintf("\nSaved figures/combined/elasticnet_per_group_cv.png\n"))

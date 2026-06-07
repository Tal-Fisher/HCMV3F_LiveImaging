options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/Rlibs", .libPaths()))
library(flexmix)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", "combined")
fig_dir   <- file.path(base_dir, "figures", "combined")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── load data ──────────────────────────────────────────────────────────────────
model_df <- readRDS(file.path(cache_dir, "model_df.rds"))
en_res   <- readRDS(file.path(cache_dir, "en_results.rds"))
meta     <- read.csv(file.path(base_dir, "Forecast", "cell_metadata.csv"),
                     stringsAsFactors = FALSE)

# GMM-defined groups: cutoffs at 911 and 2163 min
cat("GMM group sizes:\n"); print(table(meta$group))
cat(sprintf("GMM cutoffs: early < %.0f min (%.1f h), late > %.0f min (%.1f h)\n",
            911, 911/60, 2163, 2163/60))

# ── merge: model_df ∩ GMM groups (red cells only) ────────────────────────────
red_df <- model_df[is.finite(model_df$delay_green_to_red), ]
df     <- merge(red_df, meta[, c("Track.ID", "group")], by = "Track.ID")
df$group <- factor(df$group, levels = c("early", "medium", "late"))
cat(sprintf("\nCells with red onset + GMM group: %d\n", nrow(df)))
print(table(df$group))

# ── ElasticNet-selected features ──────────────────────────────────────────────
en_feats <- rownames(en_res$coefs_nz)
en_feats <- intersect(en_feats, names(df))
cat(sprintf("\nUsing %d EN-selected features: %s\n", length(en_feats),
            paste(en_feats, collapse=", ")))

# ── prepare feature matrix (scaled) and log-outcome ──────────────────────────
X_raw <- as.matrix(df[, en_feats])
# impute NA with column median (12 cells with missing slope features)
for (j in seq_len(ncol(X_raw))) {
  na_j <- is.na(X_raw[, j])
  if (any(na_j)) X_raw[na_j, j] <- median(X_raw[!na_j, j])
}
X_sc  <- scale(X_raw)
y_log <- log(df$delay_green_to_red)   # outcome on log scale

# ── stratified 75 / 25 split ──────────────────────────────────────────────────
set.seed(42)
train_idx <- unlist(lapply(levels(df$group), function(g) {
  idx <- which(df$group == g)
  sample(idx, size = floor(0.75 * length(idx)))
}))
test_idx <- setdiff(seq_len(nrow(df)), train_idx)
cat(sprintf("\nTrain: %d  |  Test: %d\n", length(train_idx), length(test_idx)))
cat("Train group distribution:\n"); print(table(df$group[train_idx]))
cat("Test  group distribution:\n"); print(table(df$group[test_idx]))

# ── fit flexmix on training set, K=3, multiple restarts ──────────────────────
fit_train <- as.data.frame(X_sc[train_idx, ])
fit_train$y <- y_log[train_idx]

# initialise component posteriors from GMM group labels to guide convergence
group_train  <- as.integer(df$group[train_idx])   # 1=early, 2=medium, 3=late
init_post    <- matrix(0.05 / 2, nrow = length(train_idx), ncol = 3)
for (i in seq_along(train_idx)) init_post[i, group_train[i]] <- 0.90
init_post    <- init_post / rowSums(init_post)

set.seed(42)
fm <- flexmix(y ~ ., data = fit_train, k = 3,
              model   = FLXMRglm(family = "gaussian"),
              cluster = init_post,
              control = list(iter.max = 300, tol = 1e-5, verbose = 0))
cat(sprintf("\nFlexmix converged: %d iterations, train log-lik=%.2f\n",
            fm@iter, logLik(fm)))
print(table(clusters(fm)))

# map component numbers to early/medium/late by median
comp_med   <- tapply(fit_train$y, clusters(fm), median)
comp_order <- order(comp_med)          # sorted: lowest=early
comp_label <- setNames(c("early","medium","late"), names(sort(comp_med)))
cat("\nComponent → group mapping (by median log-delay):\n")
for (nm in names(comp_med[comp_order]))
  cat(sprintf("  Component %s → %s  (median=%.2f = %.0f min)\n",
              nm, comp_label[nm], comp_med[nm], exp(comp_med[nm])))

# ── in-sample metrics ─────────────────────────────────────────────────────────
pred_train_list <- predict(fm)
pred_train_mat  <- do.call(cbind, lapply(pred_train_list, as.numeric))
pred_train_hard <- pred_train_mat[cbind(seq_along(train_idx), clusters(fm))]
r2_train <- 1 - sum((fit_train$y - pred_train_hard)^2) /
                sum((fit_train$y - mean(fit_train$y))^2)
r_train  <- cor(fit_train$y, pred_train_hard)
cat(sprintf("\nIn-sample (train): R²=%.3f  r=%.3f  (log scale)\n", r2_train, r_train))

# ── predict on test set ────────────────────────────────────────────────────────
fit_test <- as.data.frame(X_sc[test_idx, ])
fit_test$y <- y_log[test_idx]

# Extract per-component parameters: rows are coef.(Intercept), coef.<feat>, sigma
get_comp_pred <- function(fm, X_mat) {
  K <- fm@k
  pred_mat <- matrix(NA, nrow(X_mat), K)
  for (k in seq_len(K)) {
    p      <- parameters(fm, component = k)          # matrix: (p+2) x 1
    rnames <- rownames(p)
    int_k  <- p["coef.(Intercept)", 1]
    feat_rows <- rnames[startsWith(rnames, "coef.") & rnames != "coef.(Intercept)"]
    feat_names_k <- sub("^coef\\.", "", feat_rows)
    coef_k <- p[feat_rows, 1]
    pred_mat[, k] <- int_k + as.matrix(X_mat[, feat_names_k]) %*% coef_k
  }
  pred_mat
}

# posterior: Gaussian likelihood per component (using y from test set)
get_posterior <- function(fm, pred_mat, y_vec) {
  K    <- fm@k
  llik <- matrix(NA, length(y_vec), K)
  for (k in seq_len(K)) {
    p       <- parameters(fm, component = k)
    sigma_k <- p["sigma", 1]
    if (!is.finite(sigma_k) || sigma_k <= 0) sigma_k <- 1e-3
    llik[, k] <- dnorm(y_vec, mean = pred_mat[, k], sd = sigma_k, log = FALSE)
  }
  prior_k <- prior(fm)
  llik    <- sweep(llik, 2, prior_k, "*")
  rs      <- rowSums(llik)
  rs[rs == 0] <- 1e-300
  llik / rs
}

pred_test_mat  <- get_comp_pred(fm, as.matrix(fit_test[, en_feats]))
post_test      <- get_posterior(fm, pred_test_mat, fit_test$y)
comp_test      <- apply(post_test, 1, which.max)
label_test     <- comp_label[as.character(comp_test)]
pred_test_hard <- pred_test_mat[cbind(seq_len(nrow(fit_test)), comp_test)]

# back-transform to minutes for metrics
actual_min <- df$delay_green_to_red[test_idx]
pred_min   <- exp(pred_test_hard)

mae_test   <- mean(abs(actual_min - pred_min))
medae_test <- median(abs(actual_min - pred_min))
r2_test    <- 1 - sum((log(actual_min) - pred_test_hard)^2) /
                  sum((log(actual_min) - mean(log(actual_min)))^2)
r_test     <- cor(log(actual_min), pred_test_hard)

cat(sprintf("\nTest-set metrics (n=%d):\n", length(test_idx)))
cat(sprintf("  R²=%.3f  r=%.3f  (log scale)\n", r2_test, r_test))
cat(sprintf("  MAE=%.0f min (%.1f h)   MedAE=%.0f min (%.1f h)\n",
            mae_test, mae_test/60, medae_test, medae_test/60))

cat("\nTest-set group breakdown:\n")
for (g in c("early","medium","late")) {
  m  <- label_test == g
  if (!sum(m)) next
  a  <- log(actual_min[m]); p <- pred_test_hard[m]
  r2 <- 1 - sum((a-p)^2)/sum((a-mean(a))^2)
  r_g <- cor(a, p)
  cat(sprintf("  %-7s n=%-3d  R²=%.3f  r=%.3f\n", g, sum(m), r2, r_g))
}

# ── plot ───────────────────────────────────────────────────────────────────────
GROUP_COLS <- c(early="#e67e22", medium="#2980b9", late="#27ae60")

# convert to hours for axis readability
actual_h <- actual_min / 60
pred_h   <- pred_min   / 60
all_lim  <- range(c(actual_h, pred_h), finite = TRUE) * c(0.9, 1.1)

png(file.path(fig_dir, "flexmix_cv_predicted_vs_actual.png"),
    width = 1200, height = 430, res = 115, type = "cairo")
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 4, 1.5), oma = c(0, 0, 2.5, 0))

true_group_test <- df$group[test_idx]

for (g in c("early", "medium", "late")) {
  # show cells that BELONG to this GMM group (true label)
  mask <- true_group_test == g
  x_g  <- actual_h[mask]
  y_g  <- pred_h[mask]
  r2_g <- if (sum(mask) > 2)
    1 - sum((log(x_g) - log(y_g))^2) / sum((log(x_g) - mean(log(x_g)))^2)
  else NA
  r_g  <- if (sum(mask) > 2) cor(log(x_g), log(y_g)) else NA
  col  <- GROUP_COLS[g]

  plot(x_g, y_g,
       col  = adjustcolor(col, alpha.f = 0.6),
       pch  = 16, cex = 0.85,
       xlim = all_lim, ylim = all_lim,
       xlab = "Actual delay (h)", ylab = "Predicted delay (h)",
       main = sprintf("%s  (n=%d)\nR²=%.2f   r=%.2f",
                      g, sum(mask),
                      ifelse(is.na(r2_g), 0, r2_g),
                      ifelse(is.na(r_g),  0, r_g)))
  abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
  if (sum(mask) > 2) {
    fit_g <- lm(y_g ~ x_g)
    abline(fit_g, col = col, lwd = 1.8)
  }
}

mtext(sprintf("FlexMix (K=3, EN features, log outcome) — 75/25 CV  |  test R²=%.2f  r=%.2f  MAE=%.0f min",
              r2_test, r_test, mae_test),
      outer = TRUE, cex = 0.82, font = 2, line = 0.8)
dev.off()
cat(sprintf("\nSaved figures/combined/flexmix_cv_predicted_vs_actual.png\n"))

# ── also save results ──────────────────────────────────────────────────────────
cv_results <- list(
  model          = fm,
  comp_label     = comp_label,
  train_idx      = train_idx,
  test_idx       = test_idx,
  pred_test_min  = pred_min,
  actual_test_min = actual_min,
  label_test     = label_test,
  r2_train       = r2_train, r_train = r_train,
  r2_test        = r2_test,  r_test  = r_test,
  mae_test       = mae_test, medae_test = medae_test
)
saveRDS(cv_results, file.path(cache_dir, "flexmix_cv_results.rds"))
cat("Saved cache/combined/flexmix_cv_results.rds\n")

options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"
fig_dir  <- file.path(base_dir, "figures", "combined")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

N_PERM    <- 1000
N_OUTER   <- 10
ALPHA_EN  <- 0.5
SEED_REAL <- 42
SEED_PERM <- 123

# ── load and combine A2 + A3 ──────────────────────────────────────────────────
a2_m <- readRDS(file.path(base_dir, "cache", "A2", "model_df.rds"))
a3_m <- readRDS(file.path(base_dir, "cache", "A3", "model_df.rds"))
a2_m$dataset_A3 <- 0L
a3_m$dataset_A3 <- 1L
model_df <- rbind(a2_m, a3_m)
cat(sprintf("Combined: %d cells (A2=%d, A3=%d)\n", nrow(model_df), nrow(a2_m), nrow(a3_m)))

# ── keep productive cells only (finite delay_green_to_red) ────────────────────
model_df <- model_df[is.finite(model_df$delay_green_to_red), ]
cat(sprintf("Productive cells only: %d\n", nrow(model_df)))

# ── prepare X / y ─────────────────────────────────────────────────────────────
model_df$y <- model_df$delay_green_to_red

exclude_cols <- c("Track.ID", "delay_green_to_red", "y")
feat_cols    <- setdiff(colnames(model_df), exclude_cols)
feat_cols    <- feat_cols[vapply(feat_cols,
  function(c) sum(is.finite(model_df[[c]])) >= 5, logical(1))]
cat(sprintf("Features: %d\n", length(feat_cols)))

X_raw <- as.matrix(model_df[, feat_cols])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[, j], na.rm = TRUE)
  X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
}
X <- scale(X_raw)
y <- model_df$y
n <- length(y)

# ── Real model: global CV lambda + outer 10-fold CV ───────────────────────────
set.seed(SEED_REAL)
cv_en        <- cv.glmnet(X, y, alpha = ALPHA_EN, nfolds = 10)
lambda_fixed <- cv_en$lambda.min
cat(sprintf("lambda.min (real model): %.5f\n", lambda_fixed))

set.seed(SEED_REAL)
outer_folds <- sample(rep(seq_len(N_OUTER), length.out = n))
cv_preds    <- numeric(n)
for (k in seq_len(N_OUTER)) {
  test_idx  <- which(outer_folds == k)
  train_idx <- which(outer_folds != k)
  inner_cv  <- cv.glmnet(X[train_idx, ], y[train_idx], alpha = ALPHA_EN, nfolds = 5)
  fold_m    <- glmnet(X[train_idx, ], y[train_idx], alpha = ALPHA_EN, lambda = inner_cv$lambda.min)
  cv_preds[test_idx] <- as.numeric(predict(fold_m, newx = X[test_idx, ], s = inner_cv$lambda.min))
}
r2_cv <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
r_cv  <- cor(y, cv_preds, use = "complete.obs")

cat(sprintf("Real CV R² = %.4f   r = %.4f\n", r2_cv, r_cv))

# ── Permutation test ──────────────────────────────────────────────────────────
# Each permutation: fresh shuffle of y, same outer folds, fixed lambda.
# Fixed lambda avoids re-running inner CV 1000× and keeps the procedure clean.
cat(sprintf("\nRunning %d permutations (fixed lambda=%.5f) ...\n", N_PERM, lambda_fixed))
set.seed(SEED_PERM)
perm_r2 <- numeric(N_PERM)

for (p in seq_len(N_PERM)) {
  if (p %% 100 == 0) { cat(sprintf("  %d/%d\n", p, N_PERM)); flush(stdout()) }
  y_perm  <- sample(y)
  preds_p <- numeric(n)
  for (k in seq_len(N_OUTER)) {
    test_idx  <- which(outer_folds == k)
    train_idx <- which(outer_folds != k)
    m_p       <- glmnet(X[train_idx, ], y_perm[train_idx],
                        alpha = ALPHA_EN, lambda = lambda_fixed)
    preds_p[test_idx] <- as.numeric(predict(m_p, newx = X[test_idx, ], s = lambda_fixed))
  }
  perm_r2[p] <- 1 - sum((y_perm - preds_p)^2) / sum((y_perm - mean(y_perm))^2)
}

p_val <- mean(perm_r2 >= r2_cv)
cat(sprintf("p-value: %.4f   (real R²=%.4f, null mean=%.4f)\n",
            p_val, r2_cv, mean(perm_r2)))

# ── Save results ───────────────────────────────────────────────────────────────
saveRDS(list(perm_r2=perm_r2, r2_cv=r2_cv, r_cv=r_cv,
             p_val=p_val, n_perm=N_PERM, n_cells=n,
             lambda_fixed=lambda_fixed),
        file.path(base_dir, "cache", "perm_test_results.rds"))
cat("Saved → cache/perm_test_results.rds\n")

# ── Plot ───────────────────────────────────────────────────────────────────────
fmt_p <- function(p) if (p == 0) sprintf("< %.4f", 1/N_PERM) else sprintf("= %.4f", p)

out_png <- file.path(fig_dir, "permutation_test.png")
png(out_png, width = 700, height = 520, type = "cairo")
par(mar = c(5, 5, 4, 2))

h <- hist(perm_r2, breaks = 40, plot = FALSE)
xlim <- range(c(h$breaks, r2_cv))
cols <- ifelse(h$mids >= r2_cv, "#e74c3c66", "#2980b966")
plot(h, col = cols, border = "white", xlim = xlim,
     main = sprintf("ElasticNet permutation test  —  A2+A3 productive cells\n(n=%d cells, %d permutations)", n, N_PERM),
     xlab = "Cross-validated R²  (permuted labels)", ylab = "Count", las = 1)
abline(v = r2_cv, col = "#e74c3c", lwd = 2.5)
legend("topright", bty = "n", cex = 0.9,
       legend = c(sprintf("Real CV R² = %.3f", r2_cv),
                  sprintf("Null mean  = %.3f", mean(perm_r2)),
                  sprintf("p %s", fmt_p(p_val))),
       text.col = c("#c0392b", "#2471a3", "black"))
dev.off()
cat(sprintf("Saved → %s\n", out_png))

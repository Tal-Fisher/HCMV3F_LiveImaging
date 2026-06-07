options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

args    <- commandArgs(trailingOnly=TRUE)
dataset <- if (length(args) >= 1) args[1] else "A2"   # pass as: Rscript 05_elasticnet.R A3

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", dataset)
fig_dir   <- file.path(base_dir, "figures", dataset)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── load upstream cache ────────────────────────────────────────────────────────
if (!file.exists(file.path(cache_dir, "model_df.rds"))) stop("Run 04_features.R first")
model_df <- readRDS(file.path(cache_dir, "model_df.rds"))
feat_df  <- readRDS(file.path(cache_dir, "feat_df.rds"))
cat(sprintf("=== %s: ElasticNet (%d cells) ===\n", dataset, nrow(model_df)))

# ── prepare X / y ─────────────────────────────────────────────────────────────
max_obs  <- max(model_df$delay_green_to_red[is.finite(model_df$delay_green_to_red)], na.rm=TRUE)
model_df$y <- ifelse(is.finite(model_df$delay_green_to_red),
                     model_df$delay_green_to_red, max_obs * 1.1)

feat_cols <- setdiff(colnames(feat_df), "Track.ID")
feat_cols <- feat_cols[vapply(feat_cols,
  function(c) sum(is.finite(model_df[[c]])) >= 5, logical(1))]

X_raw <- as.matrix(model_df[, feat_cols])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[, j], na.rm=TRUE)
  X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
}
X <- scale(X_raw)
y <- model_df$y

# ── fit ───────────────────────────────────────────────────────────────────────
set.seed(42)
cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)

coefs    <- as.matrix(coef(en_model))
coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]

y_hat <- as.numeric(predict(en_model, newx=X, s=cv_en$lambda.min))
r2    <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)
r_cor <- cor(y, y_hat, use="complete.obs")
cat(sprintf("In-sample  R² = %.3f   r = %.3f\n", r2, r_cor))
cat(sprintf("Non-zero coefficients: %d\n", nrow(coefs_nz)))
print(round(coefs_nz, 4))

# ── outer CV for honest out-of-sample R² ──────────────────────────────────────
n_outer <- 10
set.seed(42)
outer_folds <- sample(rep(seq_len(n_outer), length.out = nrow(X)))
cv_preds    <- numeric(length(y))
for (k in seq_len(n_outer)) {
  test_idx  <- which(outer_folds == k)
  train_idx <- which(outer_folds != k)
  inner_cv  <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0.5, nfolds=5)
  fold_m    <- glmnet(X[train_idx,], y[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
  cv_preds[test_idx] <- as.numeric(predict(fold_m, newx=X[test_idx,], s=inner_cv$lambda.min))
}
r2_cv  <- 1 - sum((y - cv_preds)^2) / sum((y - mean(y))^2)
r_cv   <- cor(y, cv_preds, use="complete.obs")

prod_mask  <- is.finite(model_df$delay_green_to_red)
r2_cv_prod <- 1 - sum((y[prod_mask] - cv_preds[prod_mask])^2) /
                  sum((y[prod_mask] - mean(y[prod_mask]))^2)
r_cv_prod  <- cor(y[prod_mask], cv_preds[prod_mask], use="complete.obs")

cat(sprintf("CV (%d-fold)   R² = %.3f   r = %.3f\n", n_outer, r2_cv, r_cv))
cat(sprintf("CV prod-only   R² = %.3f   r = %.3f\n", r2_cv_prod, r_cv_prod))

# ── bootstrap stability ────────────────────────────────────────────────────────
set.seed(99)
boot_feats <- replicate(50, {
  idx <- sample(nrow(X), replace=TRUE)
  cv  <- cv.glmnet(X[idx,], y[idx], alpha=0.5, nfolds=5)
  m   <- glmnet(X[idx,], y[idx], alpha=0.5, lambda=cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc != 0)]
}, simplify=FALSE)
freq_df <- sort(table(unlist(boot_feats)), decreasing=TRUE)
freq_df <- freq_df[names(freq_df) != "(Intercept)"]

# ── plots ─────────────────────────────────────────────────────────────────────
png(file.path(fig_dir, "en_cv.png"), width=700, height=500, type="cairo")
plot(cv_en, main=sprintf("%s — ElasticNet CV (alpha=0.5, 10-fold)", dataset))
dev.off()

png(file.path(fig_dir, "en_coefs.png"), width=850, height=550, type="cairo")
par(mar=c(5,12,3,2))
if (nrow(coefs_nz) > 0) {
  ord <- nrow(coefs_nz):1
  barplot(coefs_nz[ord,1], names.arg=rownames(coefs_nz)[ord], horiz=TRUE, las=1,
          col=ifelse(coefs_nz[ord,1] > 0, "steelblue", "tomato"),
          main=sprintf("%s — ElasticNet coefs (%d non-zero)   CV R²=%.3f", dataset, nrow(coefs_nz), r2_cv),
          xlab="Coefficient (z-scored features)")
  abline(v=0, col="grey50")
}
dev.off()

n_show <- min(20, length(freq_df))
png(file.path(fig_dir, "en_bootstrap.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
barplot(freq_df[n_show:1], names.arg=names(freq_df)[n_show:1],
        horiz=TRUE, las=1, col="steelblue",
        main=sprintf("%s — bootstrap feature stability (n=50)", dataset),
        xlab="Times selected out of 50")
abline(v=25, col="red", lty=2)
dev.off()

png(file.path(fig_dir, "en_predicted_vs_actual.png"), width=600, height=600, type="cairo")
plot(y, cv_preds, pch=16, cex=0.5,
     col=ifelse(prod_mask, rgb(0,0.4,0.8,0.4), rgb(0.7,0.7,0.7,0.4)),
     xlab="Actual delay (min;  grey=censored)",
     ylab="Predicted delay (min)",
     main=sprintf("%s — %d-fold CV predicted vs actual\nCV R²=%.3f   prod-only CV R²=%.3f",
                  dataset, n_outer, r2_cv, r2_cv_prod))
abline(a=0, b=1, col="red", lty=2)
legend("topleft", bty="n",
       legend=c("Observed red","Censored"),
       col=c(rgb(0,0.4,0.8), rgb(0.7,0.7,0.7)), pch=16)
dev.off()

cat(sprintf("Saved figures/%s/en_*.png\n", dataset))

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(list(model=en_model, cv=cv_en, X=X, y=y, coefs_nz=coefs_nz,
             r2=r2, r_cor=r_cor, freq=freq_df,
             r2_cv=r2_cv, r_cv=r_cv, cv_preds=cv_preds,
             r2_cv_prod=r2_cv_prod, r_cv_prod=r_cv_prod),
        file.path(cache_dir, "en_results.rds"))
cat(sprintf("Saved cache/%s/en_results.rds\n", dataset))

options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache", "combined")
fig_dir    <- file.path(base_dir, "figures", "combined")

# ── load combined cache ────────────────────────────────────────────────────────
model_df <- readRDS(file.path(cache_comb, "model_df.rds"))
feat_df  <- readRDS(file.path(cache_comb, "feat_df.rds"))

cat(sprintf("Full dataset: %d cells\n", nrow(model_df)))

# ── apply late-bloomer filter ──────────────────────────────────────────────────
# Late bloomers: BFP onset >10 frames after GFP onset (10 * 15 min/frame = 150 min)
LATE_BLOOMER_CUTOFF <- 150   # minutes

model_filt <- model_df[model_df$delay_green_to_blue <= LATE_BLOOMER_CUTOFF, ]
cat(sprintf("After removing late bloomers (delay_green_to_blue >%d min): %d cells  (removed %d)\n",
            LATE_BLOOMER_CUTOFF, nrow(model_filt), nrow(model_df) - nrow(model_filt)))
cat(sprintf("  Productive (red) in filtered set: %d / %d\n",
            sum(is.finite(model_filt$delay_green_to_red)), nrow(model_filt)))

# ── helper: run ElasticNet with 10-fold nested CV ─────────────────────────────
run_elasticnet <- function(df, feat_df_full, label, n_outer=10, seed=42) {
  max_obs <- max(df$delay_green_to_red[is.finite(df$delay_green_to_red)], na.rm=TRUE)
  df$y    <- ifelse(is.finite(df$delay_green_to_red), df$delay_green_to_red, max_obs * 1.1)

  feat_cols <- setdiff(colnames(feat_df_full), "Track.ID")
  feat_cols <- feat_cols[vapply(feat_cols,
    function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]

  X_raw <- as.matrix(df[, feat_cols])
  for (j in seq_len(ncol(X_raw))) {
    med_j <- median(X_raw[, j], na.rm=TRUE)
    X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
  }
  X <- scale(X_raw)
  y <- df$y

  # in-sample fit
  set.seed(seed)
  cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
  en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)
  coefs    <- as.matrix(coef(en_model))
  coefs_nz <- coefs[coefs[,1] != 0 & rownames(coefs) != "(Intercept)", , drop=FALSE]
  coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]
  y_hat    <- as.numeric(predict(en_model, newx=X, s=cv_en$lambda.min))
  r2_in    <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

  # outer CV
  set.seed(seed)
  outer_folds <- sample(rep(seq_len(n_outer), length.out=nrow(X)))
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

  prod_mask  <- is.finite(df$delay_green_to_red)
  r2_cv_prod <- 1 - sum((y[prod_mask] - cv_preds[prod_mask])^2) /
                    sum((y[prod_mask] - mean(y[prod_mask]))^2)
  r_cv_prod  <- cor(y[prod_mask], cv_preds[prod_mask], use="complete.obs")

  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  n=%d  (productive=%d)\n", nrow(df), sum(prod_mask)))
  cat(sprintf("  In-sample R²: %.3f\n", r2_in))
  cat(sprintf("  CV R² (all):  %.3f   r=%.3f\n", r2_cv, r_cv))
  cat(sprintf("  CV R² (prod): %.3f   r=%.3f\n", r2_cv_prod, r_cv_prod))
  cat(sprintf("  Non-zero coefs: %d\n", nrow(coefs_nz)))

  list(model=en_model, cv=cv_en, X=X, y=y, coefs_nz=coefs_nz,
       r2_in=r2_in, r2_cv=r2_cv, r_cv=r_cv, cv_preds=cv_preds,
       r2_cv_prod=r2_cv_prod, r_cv_prod=r_cv_prod,
       prod_mask=prod_mask, df=df, feat_cols=feat_cols)
}

# ── run both ───────────────────────────────────────────────────────────────────
cat("\nRunning ElasticNet on FULL dataset...\n")
res_full <- run_elasticnet(model_df,   feat_df, "Full dataset (A2+A3, n=861)")

cat("\nRunning ElasticNet on FILTERED dataset (no late bloomers)...\n")
res_filt <- run_elasticnet(model_filt, feat_df, sprintf("Filtered (delay_blue<=150min, n=%d)", nrow(model_filt)))

# ── side-by-side summary ───────────────────────────────────────────────────────
cat("\n\n══════════════════════════════════════════\n")
cat("  COMPARISON: Full vs. No-Late-Bloomers\n")
cat("══════════════════════════════════════════\n")
fmt <- function(lbl, vals) sprintf("  %-28s  %8.3f  %8.3f  %+8.3f\n", lbl, vals[1], vals[2], vals[2]-vals[1])
cat(sprintf("  %-28s  %8s  %8s  %8s\n", "Metric", "Full", "Filtered", "Delta"))
cat(sprintf("  %-28s  %8s  %8s  %8s\n", "------", "----", "--------", "-----"))
cat(fmt("CV R² (all cells)",    c(res_full$r2_cv,           res_filt$r2_cv)))
cat(fmt("CV r (all cells)",     c(res_full$r_cv,            res_filt$r_cv)))
cat(fmt("CV R² (productive)",   c(res_full$r2_cv_prod,      res_filt$r2_cv_prod)))
cat(fmt("CV r (productive)",    c(res_full$r_cv_prod,       res_filt$r_cv_prod)))
cat(fmt("In-sample R²",         c(res_full$r2_in,           res_filt$r2_in)))
cat(fmt("Non-zero coefs",       c(nrow(res_full$coefs_nz),  nrow(res_filt$coefs_nz))))
cat("══════════════════════════════════════════\n")

# ── plot: predicted vs actual, side by side ───────────────────────────────────
png(file.path(fig_dir, "en_no_late_bloomers_pred_vs_actual.png"),
    width=1200, height=600, type="cairo")
par(mfrow=c(1,2), mar=c(4,4,3,1))
for (res in list(res_full, res_filt)) {
  lbl <- if (nrow(res$df) == nrow(model_df)) "Full (n=861)" else
           sprintf("No late bloomers (n=%d)", nrow(res$df))
  plot(res$y, res$cv_preds, pch=16, cex=0.5,
       col=ifelse(res$prod_mask, rgb(0,0.4,0.8,0.4), rgb(0.7,0.7,0.7,0.4)),
       xlab="Actual delay (min;  grey=censored)",
       ylab="Predicted delay (min)",
       main=sprintf("%s\nCV R²=%.3f  prod-only CV R²=%.3f", lbl, res$r2_cv, res$r2_cv_prod))
  abline(a=0, b=1, col="red", lty=2)
  legend("topleft", bty="n",
         legend=c("Observed red","Censored"),
         col=c(rgb(0,0.4,0.8), rgb(0.7,0.7,0.7)), pch=16)
}
dev.off()
cat("\nSaved figures/combined/en_no_late_bloomers_pred_vs_actual.png\n")

# ── plot: coefficient comparison ──────────────────────────────────────────────
all_coef_names <- union(rownames(res_full$coefs_nz), rownames(res_filt$coefs_nz))
coef_mat <- matrix(0, nrow=length(all_coef_names), ncol=2,
                   dimnames=list(all_coef_names, c("Full","Filtered")))
coef_mat[rownames(res_full$coefs_nz), "Full"]     <- res_full$coefs_nz[,1]
coef_mat[rownames(res_filt$coefs_nz), "Filtered"] <- res_filt$coefs_nz[,1]
ord    <- order(abs(coef_mat[,"Full"]), decreasing=TRUE)
coef_mat <- coef_mat[ord, , drop=FALSE]
n_show <- min(20, nrow(coef_mat))

png(file.path(fig_dir, "en_no_late_bloomers_coef_comparison.png"),
    width=1000, height=600, type="cairo")
par(mar=c(5,13,3,2))
barplot(t(coef_mat[n_show:1, , drop=FALSE]),
        beside=TRUE, horiz=TRUE, las=1,
        col=c("steelblue", "tomato"),
        main=sprintf("ElasticNet coefficients: Full vs No-Late-Bloomers\n(top %d by |full coef|)", n_show),
        xlab="Coefficient (z-scored features)")
legend("bottomright", bty="n", legend=c("Full","Filtered"),
       fill=c("steelblue","tomato"))
abline(v=0, col="grey50")
dev.off()
cat("Saved figures/combined/en_no_late_bloomers_coef_comparison.png\n")

# ── save filtered results ─────────────────────────────────────────────────────
saveRDS(res_filt,
        file.path(cache_comb, "en_results_no_late_bloomers.rds"))
cat("Saved cache/combined/en_results_no_late_bloomers.rds\n")

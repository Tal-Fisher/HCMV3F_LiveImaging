options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)
library(mclust)

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache",   "combined")
fig_comb   <- file.path(base_dir, "figures", "combined")
dir.create(fig_comb, recursive = TRUE, showWarnings = FALSE)

# ── load combined caches ───────────────────────────────────────────────────────
for (f in c("model_df.rds", "feat_df.rds")) {
  if (!file.exists(file.path(cache_comb, f)))
    stop(sprintf("Missing cache/combined/%s — run 07_combine.R first", f))
}
model_df <- readRDS(file.path(cache_comb, "model_df.rds"))
feat_df  <- readRDS(file.path(cache_comb, "feat_df.rds"))

cat(sprintf("Loaded %d cells: %d productive, %d non-productive\n",
            nrow(model_df),
            sum(is.finite(model_df$delay_green_to_red)),
            sum(!is.finite(model_df$delay_green_to_red))))

# ══════════════════════════════════════════════════════════════════════════════
# 1. GAUSSIAN MIXTURE MODEL (G=3) → Bayes-optimal category cutoffs
# ══════════════════════════════════════════════════════════════════════════════
valid_delays <- model_df$delay_green_to_red[is.finite(model_df$delay_green_to_red)]

set.seed(42)
mc3 <- Mclust(valid_delays, G = 3, verbose = FALSE)
mu  <- mc3$parameters$mean
# sigmasq may be a scalar (equal-variance models) or a vector — always expand to length 3
sig <- sqrt(mc3$parameters$variance$sigmasq)
if (length(sig) == 1) sig <- rep(sig, 3)
pro <- mc3$parameters$pro
# sort components by mean so early < medium < late
ord <- order(mu); mu <- mu[ord]; sig <- sig[ord]; pro <- pro[ord]

# Bayes-optimal cutoffs: x where posterior probability switches majority
x_grid   <- seq(0, max(valid_delays), by = 1)
dens_mat <- sapply(1:3, function(i) pro[i] * dnorm(x_grid, mu[i], sig[i]))
cls_pred <- apply(dens_mat, 1, which.max)

find_last_dominant <- function(cls_pred, x_grid, k) {
  idx <- which(cls_pred == k)
  if (length(idx) == 0) return(NA_real_)
  x_grid[max(idx)]
}
cutoff1 <- find_last_dominant(cls_pred, x_grid, 1)
cutoff2 <- find_last_dominant(cls_pred, x_grid, 2)
# Fallback when medium component never dominates: use posterior crossover P2=P3
if (is.na(cutoff2)) {
  cat("GMM medium class never dominant — using posterior crossover P2=P3 as second cutoff\n")
  diff23  <- dens_mat[, 2] - dens_mat[, 3]
  cross   <- which(diff(sign(diff23)) != 0)
  cutoff2 <- if (length(cross) > 0) x_grid[cross[1]] else (mu[2] + mu[3]) / 2
}
cutoffs <- c(cutoff1, cutoff2)

cat(sprintf("GMM G=3 means: %.0f / %.0f / %.0f min\n", mu[1], mu[2], mu[3]))
cat(sprintf("Bayes cutoffs: %.0f min  |  %.0f min\n", cutoffs[1], cutoffs[2]))

# assign categories
category <- cut(valid_delays,
                breaks = c(-Inf, cutoffs[1], cutoffs[2], Inf),
                labels = c("early", "medium", "late"))
cat(sprintf("Categories — early: %d  medium: %d  late: %d\n",
            sum(category=="early"), sum(category=="medium"), sum(category=="late")))

# ── figure 1: KDE with cutoffs + category distribution ────────────────────────
png(file.path(fig_comb, "categorical_overview.png"), width=1100, height=500, type="cairo")
par(mfrow=c(1,2), mar=c(4,4,3,1), oma=c(0,0,2,0))

# panel A: histogram + GMM components + cutoffs
cat_cols <- c(early="steelblue", medium="darkorange", late="tomato")
n_cat    <- table(category)

hist(valid_delays, breaks=50, freq=FALSE,
     col="grey88", border="white",
     main="GFP→mCherry delay — GMM G=3",
     xlab="Delay from GFP onset (min)", ylab="Density", bty="l")

x_seq <- seq(0, max(valid_delays), length.out=1000)
gmm_cols <- c("steelblue","darkorange","tomato")
for (i in 1:3) {
  comp_dens <- pro[i] * dnorm(x_seq, mu[i], sig[i])
  lines(x_seq, comp_dens, col=gmm_cols[i], lwd=2)
}
total_dens <- rowSums(sapply(1:3, function(i) pro[i]*dnorm(x_seq, mu[i], sig[i])))
lines(x_seq, total_dens, col="black", lwd=1.5, lty=2)

abline(v=cutoffs, col="grey30", lwd=2, lty=2)
text(c(cutoffs[1]/2, mean(cutoffs), cutoffs[2]+(max(valid_delays)-cutoffs[2])/2),
     par("usr")[4]*0.95,
     labels=sprintf("%s\nn=%d", names(n_cat), as.integer(n_cat)),
     col=cat_cols[names(n_cat)], cex=0.85, font=2, adj=c(0.5,1))
legend("topright", bty="n", lwd=2, col=c(gmm_cols,"black"),
       legend=c(sprintf("early (μ=%.0f)", mu[1]),
                sprintf("medium (μ=%.0f)", mu[2]),
                sprintf("late (μ=%.0f)", mu[3]),
                "GMM total"), cex=0.75)

# panel B: bar chart including non-productive
n_nonprod <- sum(!is.finite(model_df$delay_green_to_red))
bar_n   <- c(as.integer(n_cat), n_nonprod)
bar_lab <- c("early","medium","late","non-\nproductive")
bar_col <- c(cat_cols, `non-\nproductive` = "grey60")
bp <- barplot(bar_n, names.arg = bar_lab, col = bar_col,
              border = "white", ylim = c(0, max(bar_n)*1.2),
              main = "Cells per category",
              ylab = "Number of cells", las = 1)
text(bp, bar_n + max(bar_n)*0.03, bar_n, cex = 0.85, font = 2, adj = c(0.5,0))

mtext("Categorical analysis — delay categories", outer=TRUE, cex=1, font=2)
dev.off()
cat("Saved figures/combined/categorical_overview.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SHARED FEATURE MATRIX PREP
# ══════════════════════════════════════════════════════════════════════════════
feat_cols <- setdiff(colnames(feat_df), "Track.ID")
feat_cols <- feat_cols[vapply(feat_cols,
  function(c) sum(is.finite(model_df[[c]])) >= 5, logical(1))]

prep_X <- function(df, cols) {
  Xr <- as.matrix(df[, cols])
  for (j in seq_len(ncol(Xr))) {
    med_j <- median(Xr[, j], na.rm = TRUE)
    Xr[is.na(Xr[, j]), j] <- if (is.finite(med_j)) med_j else 0
  }
  scale(Xr)
}

X_all <- prep_X(model_df, feat_cols)

# ══════════════════════════════════════════════════════════════════════════════
# 3. BINARY MODEL: productive vs non-productive
# ══════════════════════════════════════════════════════════════════════════════
cat("\n── Binary model: productive vs non-productive ──\n")

y_bin <- as.integer(is.finite(model_df$delay_green_to_red))

set.seed(42)
cv_bin  <- cv.glmnet(X_all, y_bin, family="binomial", alpha=0.5,
                     nfolds=10, type.measure="auc")
bin_mod <- glmnet(X_all, y_bin, family="binomial", alpha=0.5,
                  lambda=cv_bin$lambda.min)

bin_coefs    <- as.matrix(coef(bin_mod))
bin_coefs_nz <- bin_coefs[bin_coefs[,1]!=0 & rownames(bin_coefs)!="(Intercept)",, drop=FALSE]
bin_coefs_nz <- bin_coefs_nz[order(abs(bin_coefs_nz[,1]), decreasing=TRUE),, drop=FALSE]
cat(sprintf("Binary EN: CV AUC=%.3f  non-zero coefs=%d\n",
            max(cv_bin$cvm), nrow(bin_coefs_nz)))

# 5-fold CV predictions for ROC
k_folds <- 5
set.seed(42)
fold_ids_bin <- sample(rep(1:k_folds, length.out=nrow(X_all)))
cv_prob_bin  <- numeric(nrow(X_all))
for (fold in seq_len(k_folds)) {
  ti  <- which(fold_ids_bin == fold)
  tri <- which(fold_ids_bin != fold)
  icv <- cv.glmnet(X_all[tri,], y_bin[tri], family="binomial",
                   alpha=0.5, nfolds=5, type.measure="auc")
  fm  <- glmnet(X_all[tri,], y_bin[tri], family="binomial",
                alpha=0.5, lambda=icv$lambda.min)
  cv_prob_bin[ti] <- as.numeric(predict(fm, newx=X_all[ti,],
                                        type="response", s=icv$lambda.min))
}

# manual ROC
roc_data <- function(probs, labels) {
  thresholds <- sort(unique(probs), decreasing=TRUE)
  tpr <- fpr <- numeric(length(thresholds)+1)
  for (k in seq_along(thresholds)) {
    pred     <- as.integer(probs >= thresholds[k])
    tpr[k+1] <- sum(pred==1 & labels==1) / max(sum(labels==1),1)
    fpr[k+1] <- sum(pred==1 & labels==0) / max(sum(labels==0),1)
  }
  tpr[1] <- fpr[1] <- 0
  auc <- sum(diff(fpr) * (tpr[-1]+tpr[-length(tpr)])/2)
  list(tpr=tpr, fpr=fpr, auc=abs(auc))
}
roc <- roc_data(cv_prob_bin, y_bin)
cat(sprintf("Binary CV AUC (manual ROC): %.3f\n", roc$auc))

# ── binary figures ─────────────────────────────────────────────────────────────
png(file.path(fig_comb, "categorical_binary.png"), width=1100, height=500, type="cairo")
par(mfrow=c(1,2), mar=c(5,12,3,2), oma=c(0,0,2,0))

# coef plot
if (nrow(bin_coefs_nz) > 0) {
  ord <- nrow(bin_coefs_nz):1
  barplot(bin_coefs_nz[ord,1],
          names.arg=rownames(bin_coefs_nz)[ord],
          horiz=TRUE, las=1,
          col=ifelse(bin_coefs_nz[ord,1]>0,"tomato","steelblue"),
          main=sprintf("Binary EN (%d coefs)", nrow(bin_coefs_nz)),
          xlab="Coefficient (z-scored)\npositive = more productive")
  abline(v=0, col="grey50")
}

# ROC
par(mar=c(5,5,3,2))
plot(roc$fpr, roc$tpr, type="l", lwd=2, col="steelblue",
     xlim=c(0,1), ylim=c(0,1),
     xlab="False Positive Rate", ylab="True Positive Rate",
     main=sprintf("Binary model ROC\n5-fold CV AUC = %.3f", roc$auc))
abline(0,1, col="grey60", lty=2)
legend("bottomright", bty="n",
       legend=sprintf("AUC = %.3f", roc$auc),
       col="steelblue", lwd=2)

mtext("Binary model: productive vs non-productive", outer=TRUE, cex=1, font=2)
dev.off()
cat("Saved figures/combined/categorical_binary.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CATEGORICAL MODEL: regress delay → threshold at GMM cutoffs
# Rationale: categories are a discretisation of the continuous delay.
# Predicting the continuous delay and thresholding preserves ordinal structure,
# unlike multinomial EN which treats labels as unordered.
# ══════════════════════════════════════════════════════════════════════════════
cat("\n── Categorical model (regression + threshold): early / medium / late ──\n")

prod_mask <- is.finite(model_df$delay_green_to_red)
X_prod    <- X_all[prod_mask, , drop=FALSE]
y_cont    <- model_df$delay_green_to_red[prod_mask]
y_cat     <- category   # factor with levels early/medium/late

cat(sprintf("n=%d productive cells  |  early=%d  medium=%d  late=%d\n",
            length(y_cat), sum(y_cat=="early"), sum(y_cat=="medium"), sum(y_cat=="late")))

# fit continuous EN on productive cells
set.seed(42)
cv_cont  <- cv.glmnet(X_prod, y_cont, alpha=0.5, nfolds=10)
cont_mod <- glmnet(X_prod, y_cont, alpha=0.5, lambda=cv_cont$lambda.min)

cont_coefs    <- as.matrix(coef(cont_mod))
cont_coefs_nz <- cont_coefs[cont_coefs[,1]!=0 & rownames(cont_coefs)!="(Intercept)",, drop=FALSE]
cont_coefs_nz <- cont_coefs_nz[order(abs(cont_coefs_nz[,1]), decreasing=TRUE),, drop=FALSE]
y_hat_full    <- as.numeric(predict(cont_mod, newx=X_prod, s=cv_cont$lambda.min))
r2_prod <- 1 - sum((y_cont - y_hat_full)^2) / sum((y_cont - mean(y_cont))^2)
cat(sprintf("Continuous EN on productive cells: R²=%.3f  non-zero coefs=%d\n",
            r2_prod, nrow(cont_coefs_nz)))

# 5-fold CV predictions → rank-based threshold → category
# Regularisation shrinks absolute predictions toward the mean (falls in "medium"),
# but the model CAN rank cells. We assign categories by predicted rank quantile,
# matching the true class proportions (early=21.5%, medium=44.7%, late=33.8%).
set.seed(42)
fold_ids_cat  <- sample(rep(1:k_folds, length.out=nrow(X_prod)))
cv_pred_delay <- numeric(nrow(X_prod))
for (fold in seq_len(k_folds)) {
  ti  <- which(fold_ids_cat == fold)
  tri <- which(fold_ids_cat != fold)
  icv <- cv.glmnet(X_prod[tri,], y_cont[tri], alpha=0.5, nfolds=5)
  fm  <- glmnet(X_prod[tri,], y_cont[tri], alpha=0.5, lambda=icv$lambda.min)
  cv_pred_delay[ti] <- as.numeric(predict(fm, newx=X_prod[ti,], s=icv$lambda.min))
}

spearman_r <- cor(cv_pred_delay, y_cont, method="spearman")
cat(sprintf("Spearman r (predicted vs true delay): %.3f\n", spearman_r))

# rank-based cutoffs matching true class proportions
prop_early  <- sum(y_cat=="early")  / length(y_cat)
prop_medium <- sum(y_cat=="medium") / length(y_cat)
q_cuts <- quantile(cv_pred_delay, c(prop_early, prop_early + prop_medium))
cv_pred_cat <- factor(
  ifelse(cv_pred_delay <= q_cuts[1], "early",
         ifelse(cv_pred_delay <= q_cuts[2], "medium", "late")),
  levels=c("early","medium","late"))
true_cat <- y_cat

cv_acc <- mean(cv_pred_cat == true_cat)
cat(sprintf("Categorical CV accuracy (rank-based): %.1f%%  (chance=%.1f%%)\n",
            100*cv_acc, 100*max(table(y_cat)/length(y_cat))))

# per-class recall
for (cl in c("early","medium","late")) {
  recall <- mean(cv_pred_cat[true_cat==cl] == cl)
  cat(sprintf("  %s recall: %.1f%%\n", cl, 100*recall))
}

conf_mat <- table(Predicted=cv_pred_cat, True=true_cat)
print(conf_mat)

# ── categorical figures ────────────────────────────────────────────────────────
cat_cls_cols <- c(early="steelblue", medium="darkorange", late="tomato")
lvls         <- c("early","medium","late")

png(file.path(fig_comb, "categorical_multinomial.png"), width=1800, height=600, type="cairo")
par(mfrow=c(1,3), oma=c(0,0,2,0))

# panel A: continuous EN coefficients
par(mar=c(5,13,3,2))
if (nrow(cont_coefs_nz) > 0) {
  n_show <- min(20, nrow(cont_coefs_nz))
  cc     <- cont_coefs_nz[seq_len(n_show), , drop=FALSE]
  ord    <- n_show:1
  barplot(cc[ord, 1], names.arg=rownames(cc)[ord], horiz=TRUE, las=1,
          col=ifelse(cc[ord,1] > 0, "tomato", "steelblue"),
          main=sprintf("Delay EN — productive cells\n(%d coefs, R²=%.3f)\nred=longer delay",
                       nrow(cont_coefs_nz), r2_prod),
          xlab="Coefficient (z-scored)")
  abline(v=0, col="grey50")
}

# panel B: predicted vs true delay, coloured by true category
par(mar=c(5,5,3,2))
pt_cols <- cat_cls_cols[as.character(true_cat)]
plot(cv_pred_delay, y_cont,
     col=adjustcolor(pt_cols, 0.6), pch=16, cex=0.7,
     xlab="CV predicted delay (min)", ylab="True delay (min)",
     main=sprintf("Predicted vs true delay\nSpearman r=%.3f", spearman_r))
abline(0, 1, col="grey50", lty=2)
abline(lm(y_cont ~ cv_pred_delay), col="black", lwd=1.5)
legend("topleft", bty="n", pch=16, col=cat_cls_cols[lvls],
       legend=sprintf("%s (n=%d)", lvls, as.integer(table(y_cat)[lvls])), cex=0.75)

# panel C: confusion matrix
par(mar=c(5,6,3,2))
cm      <- matrix(as.integer(conf_mat), nrow=3,
                  dimnames=list(Predicted=lvls, True=lvls))
cm_prop <- sweep(cm, 2, colSums(cm), "/")

image(seq_along(lvls), seq_along(lvls), t(cm_prop)[, rev(seq_along(lvls))],
      col=colorRampPalette(c("white","steelblue"))(50),
      zlim=c(0,1), xaxt="n", yaxt="n",
      xlab="True class", ylab="Predicted class",
      main=sprintf("Confusion matrix (5-fold CV, rank-based)\nacc=%.1f%%  chance=%.1f%%",
                   100*cv_acc, 100*max(table(y_cat)/length(y_cat))))
axis(1, at=seq_along(lvls), labels=lvls)
axis(2, at=seq_along(lvls), labels=rev(lvls), las=1)
for (i in seq_along(lvls)) for (j in seq_along(lvls)) {
  val   <- cm_prop[rev(lvls)[i], lvls[j]]
  n_val <- cm[rev(lvls)[i], lvls[j]]
  text(j, i, sprintf("%.0f%%\n(n=%d)", 100*val, n_val),
       cex=0.85, col=if(val > 0.5) "white" else "black")
}

mtext("Categorical model: regression + rank-based threshold (early/medium/late)",
      outer=TRUE, cex=1, font=2)
dev.off()
cat("Saved figures/combined/categorical_multinomial.png\n")

# ── save category assignments ──────────────────────────────────────────────────
cat_df <- data.frame(
  Track.ID = model_df$Track.ID,
  productive = is.finite(model_df$delay_green_to_red),
  delay_green_to_red = model_df$delay_green_to_red,
  category = NA_character_,
  stringsAsFactors = FALSE
)
cat_df$category[prod_mask] <- as.character(category)
cat_df$category[!prod_mask] <- "non-productive"

saveRDS(cat_df,    file.path(cache_comb, "category_df.rds"))
saveRDS(cutoffs,   file.path(cache_comb, "category_cutoffs.rds"))
cat(sprintf("Saved cache/combined/category_df.rds  (cutoffs: %.0f, %.0f min)\n",
            cutoffs[1], cutoffs[2]))

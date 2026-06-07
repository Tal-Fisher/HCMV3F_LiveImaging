options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)
library(survival)

dataset  <- "A2"   # change to "A3" or "combined"

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", dataset)
fig_dir   <- file.path(base_dir, "figures", dataset)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── load upstream caches ───────────────────────────────────────────────────────
for (f in c("model_df.rds","spots_clean.rds","en_results.rds")) {
  if (!file.exists(file.path(cache_dir, f)))
    stop(sprintf("Missing cache/%s/%s", dataset, f))
}
model_df <- readRDS(file.path(cache_dir, "model_df.rds"))
spots    <- readRDS(file.path(cache_dir, "spots_clean.rds"))
en_res   <- readRDS(file.path(cache_dir, "en_results.rds"))
X        <- en_res$X
cat(sprintf("=== %s: Cox ElasticNet (%d cells) ===\n", dataset, nrow(model_df)))

# ── survival outcome ───────────────────────────────────────────────────────────
track_dur <- vapply(split(spots, spots$Track.ID),
                    function(df) (max(df$T..sec.) - min(df$T..sec.)) / 60, numeric(1))
cox_df <- model_df
cox_df$event     <- as.integer(is.finite(cox_df$delay_green_to_red))
cox_df$surv_time <- ifelse(cox_df$event == 1,
                           cox_df$delay_green_to_red,
                           track_dur[as.character(cox_df$Track.ID)])
cox_df <- cox_df[is.finite(cox_df$surv_time) & cox_df$surv_time > 0, ]

cat(sprintf("Cox: %d cells | %d events | %d censored\n",
            nrow(cox_df), sum(cox_df$event), sum(cox_df$event == 0)))

X_cox  <- X[match(rownames(cox_df), rownames(model_df)), , drop=FALSE]
surv_y <- Surv(cox_df$surv_time, cox_df$event)

# ── fit ───────────────────────────────────────────────────────────────────────
set.seed(42)
cv_cox <- cv.glmnet(X_cox, surv_y, family="cox", alpha=0.5, nfolds=10)
cox_en <- glmnet(X_cox, surv_y, family="cox", alpha=0.5, lambda=cv_cox$lambda.min)

cox_coefs    <- as.matrix(coef(cox_en))
cox_coefs_nz <- cox_coefs[cox_coefs[,1] != 0, , drop=FALSE]
cox_coefs_nz <- cox_coefs_nz[order(abs(cox_coefs_nz[,1]), decreasing=TRUE), , drop=FALSE]

risk_score <- as.numeric(predict(cox_en, newx=X_cox, s=cv_cox$lambda.min, type="link"))
conc       <- concordance(surv_y ~ risk_score)
c_stat     <- max(conc$concordance, 1 - conc$concordance)
cat(sprintf("In-sample C: %.3f  |  Non-zero coefs: %d\n", c_stat, nrow(cox_coefs_nz)))
print(round(cox_coefs_nz, 4))

# ── bootstrap ─────────────────────────────────────────────────────────────────
set.seed(99)
boot_cox <- replicate(50, {
  idx <- sample(nrow(X_cox), replace=TRUE)
  cv  <- cv.glmnet(X_cox[idx,], surv_y[idx], family="cox", alpha=0.5, nfolds=5)
  m   <- glmnet(X_cox[idx,], surv_y[idx], family="cox", alpha=0.5, lambda=cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc != 0)]
}, simplify=FALSE)
cox_freq <- sort(table(unlist(boot_cox)), decreasing=TRUE)

# ── 5-fold cross-validated KM ─────────────────────────────────────────────────
k_folds  <- 5; set.seed(42)
fold_ids <- sample(rep(1:k_folds, length.out=nrow(X_cox)))
cv_risk  <- numeric(nrow(X_cox))
for (fold in seq_len(k_folds)) {
  test_idx <- which(fold_ids==fold); train_idx <- which(fold_ids!=fold)
  inner_cv <- cv.glmnet(X_cox[train_idx,], surv_y[train_idx],
                         family="cox", alpha=0.5, nfolds=5)
  fm <- glmnet(X_cox[train_idx,], surv_y[train_idx],
               family="cox", alpha=0.5, lambda=inner_cv$lambda.min)
  cv_risk[test_idx] <- as.numeric(predict(fm, newx=X_cox[test_idx,], type="link"))
}
conc_cv   <- concordance(surv_y ~ cv_risk)
c_stat_cv <- max(conc_cv$concordance, 1 - conc_cv$concordance)
cat(sprintf("CV C: %.3f\n", c_stat_cv))

risk_group_cv <- ifelse(cv_risk > median(cv_risk), "High risk", "Low risk")
km_fit_cv     <- survfit(surv_y ~ risk_group_cv)
logrank       <- survdiff(surv_y ~ risk_group_cv)
p_val         <- 1 - pchisq(logrank$chisq, df=1)
cat(sprintf("Log-rank p = %.4f\n", p_val))
km_tbl <- summary(km_fit_cv)$table
med_hi <- km_tbl[grep("High", rownames(km_tbl)), "median"]
med_lo <- km_tbl[grep("Low",  rownames(km_tbl)), "median"]

# ── plots ─────────────────────────────────────────────────────────────────────
png(file.path(fig_dir, "cox_cv.png"), width=700, height=500, type="cairo")
plot(cv_cox, main=sprintf("%s — Cox ElasticNet CV\nC=%.3f", dataset, c_stat))
dev.off()

png(file.path(fig_dir, "cox_coefs.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
if (nrow(cox_coefs_nz) > 0) {
  ord <- nrow(cox_coefs_nz):1
  barplot(cox_coefs_nz[ord,1], names.arg=rownames(cox_coefs_nz)[ord],
          horiz=TRUE, las=1, col=ifelse(cox_coefs_nz[ord,1]>0,"tomato","steelblue"),
          main=sprintf("%s — Cox coefs (%d non-zero)  |  blue=faster to red", dataset, nrow(cox_coefs_nz)),
          xlab="log Hazard Ratio (per SD)")
  abline(v=0, col="grey50")
}
dev.off()

n_show <- min(20, length(cox_freq))
png(file.path(fig_dir, "cox_bootstrap.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
barplot(cox_freq[n_show:1], names.arg=names(cox_freq)[n_show:1],
        horiz=TRUE, las=1, col="steelblue",
        main=sprintf("%s — Cox bootstrap stability (n=50)", dataset),
        xlab="Times selected out of 50")
abline(v=25, col="red", lty=2)
dev.off()

png(file.path(fig_dir, "cox_km.png"), width=700, height=600, type="cairo")
plot(km_fit_cv, col=c("tomato","steelblue"), lwd=2,
     xlab="Time from green onset (min)", ylab="P(not yet red)",
     main=sprintf("%s — KM %d-fold CV  |  C=%.3f  |  p=%.4f",
                  dataset, k_folds, c_stat_cv, p_val))
legend("topright", bty="n",
       legend=c(sprintf("High risk (n=%d, median=%.0f min)", sum(risk_group_cv=="High risk"), med_hi),
                sprintf("Low risk  (n=%d, median=%.0f min)", sum(risk_group_cv=="Low risk"),  med_lo)),
       col=c("tomato","steelblue"), lwd=2)
dev.off()

cat(sprintf("Saved figures/%s/cox_*.png\n", dataset))

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(list(model=cox_en, cv=cv_cox, X_cox=X_cox, surv_y=surv_y,
             coefs_nz=cox_coefs_nz, c_stat=c_stat, c_stat_cv=c_stat_cv,
             p_val=p_val, cv_risk=cv_risk, risk_group_cv=risk_group_cv),
        file.path(cache_dir, "cox_results.rds"))
cat(sprintf("Saved cache/%s/cox_results.rds\n", dataset))

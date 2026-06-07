options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(glmnet)
library(survival)

datasets  <- c("A2", "A3")
combined  <- "combined"

base_dir      <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb    <- file.path(base_dir, "cache",   combined)
fig_comb      <- file.path(base_dir, "figures", combined)
dir.create(cache_comb, recursive=TRUE, showWarnings=FALSE)
dir.create(fig_comb,   recursive=TRUE, showWarnings=FALSE)

# ── load and tag each dataset ──────────────────────────────────────────────────
load_rds <- function(ds, file) {
  path <- file.path(base_dir, "cache", ds, file)
  if (!file.exists(path)) stop(sprintf("Missing cache/%s/%s — run that dataset's pipeline first", ds, file))
  readRDS(path)
}

spots_list   <- lapply(datasets, load_rds, "spots_clean.rds")
nuc_list     <- lapply(datasets, load_rds, "nuc_assigned.rds")
onset_list   <- lapply(datasets, load_rds, "onset_df.rds")
feat_list    <- lapply(datasets, load_rds, "feat_df.rds")
model_list   <- lapply(datasets, load_rds, "model_df.rds")

# prefix Track.ID with dataset label to avoid collisions
for (i in seq_along(datasets)) {
  ds <- datasets[i]
  spots_list[[i]]$Track.ID  <- paste0(ds, "_", spots_list[[i]]$Track.ID)
  spots_list[[i]]$dataset   <- ds
  nuc_list[[i]]$Track.ID    <- paste0(ds, "_", nuc_list[[i]]$Track.ID)
  onset_list[[i]]$Track.ID  <- paste0(ds, "_", onset_list[[i]]$Track.ID)
  onset_list[[i]]$dataset   <- ds
  feat_list[[i]]$Track.ID   <- paste0(ds, "_", feat_list[[i]]$Track.ID)
  model_list[[i]]$Track.ID  <- paste0(ds, "_", model_list[[i]]$Track.ID)
  model_list[[i]]$dataset   <- ds
}

# keep only columns present in all datasets before rbind
common_spots <- Reduce(intersect, lapply(spots_list, names))
common_nuc   <- Reduce(intersect, lapply(nuc_list,   names))
spots    <- do.call(rbind, lapply(spots_list, function(d) d[, common_spots, drop=FALSE]))
nuc      <- do.call(rbind, lapply(nuc_list,   function(d) d[, common_nuc,   drop=FALSE]))
onset_df <- do.call(rbind, onset_list)
feat_df  <- do.call(rbind, feat_list)
model_df <- do.call(rbind, model_list)

cat(sprintf("Combined: %d cells  |  %d with nucleus  |  green %d  blue %d  red %d\n",
            length(unique(spots$Track.ID)),
            length(unique(nuc$Track.ID)),
            sum(!is.na(onset_df$green_onset_min)),
            sum(!is.na(onset_df$blue_onset_min)),
            sum(!is.na(onset_df$red_onset_min))))

# ══════════════════════════════════════════════════════════════════════════════
# 1. TIME-OF-APPEARANCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

colors <- c(A2="steelblue", A3="tomato")

# helper: overlapping density per dataset
plot_density_ds <- function(x_list, main, xlab, cols, xlim=NULL) {
  all_x <- unlist(x_list)
  if (is.null(xlim)) xlim <- quantile(all_x, c(0, 0.99), na.rm=TRUE)
  dens <- lapply(x_list, function(x) density(x[!is.na(x) & is.finite(x)],
                                              from=xlim[1], to=xlim[2], n=512))
  ylim <- c(0, max(sapply(dens, function(d) max(d$y)), na.rm=TRUE) * 1.1)
  plot(NULL, xlim=xlim, ylim=ylim, xlab=xlab, ylab="Density", main=main)
  for (i in seq_along(dens))
    polygon(c(dens[[i]]$x, rev(dens[[i]]$x)),
            c(dens[[i]]$y, rep(0, length(dens[[i]]$y))),
            col=adjustcolor(cols[i], 0.35), border=cols[i])
  legend("topright", bty="n", legend=names(x_list),
         fill=adjustcolor(cols, 0.5), border=cols)
}

onset_split <- split(onset_df, onset_df$dataset)

png(file.path(fig_comb, "onset_distributions.png"), width=1100, height=420, type="cairo")
par(mfrow=c(1,3), mar=c(4,4,3,1), oma=c(0,0,2,0))

plot_density_ds(
  lapply(onset_split, function(d) d$green_onset_min[!is.na(d$green_onset_min)]),
  main="GFP onset", xlab="Time from track start (min)", cols=colors)

plot_density_ds(
  lapply(onset_split, function(d) d$blue_onset_min[!is.na(d$blue_onset_min)]),
  main="BFP onset", xlab="Time from track start (min)", cols=colors)

plot_density_ds(
  lapply(onset_split, function(d) d$red_onset_min[!is.na(d$red_onset_min)]),
  main="mCherry onset", xlab="Time from track start (min)", cols=colors)

mtext("A2 vs A3 — time of fluorophore appearance", outer=TRUE, cex=1, font=2)
dev.off()
cat("Saved figures/combined/onset_distributions.png\n")

# green-to-red delay per dataset + combined
png(file.path(fig_comb, "green_to_red_delay.png"), width=800, height=450, type="cairo")
par(mfrow=c(1,3), mar=c(4,4,3,1), oma=c(0,0,2,0))
for (ds in datasets) {
  d <- onset_split[[ds]]
  v <- d$delay_green_to_red[is.finite(d$delay_green_to_red)]
  hist(v, breaks=35, col=adjustcolor(colors[ds],0.7), border="white",
       main=sprintf("%s\nn=%d, median=%.0f min", ds, length(v), median(v)),
       xlab="GFP → mCherry delay (min)", ylab="Cells")
  abline(v=median(v), col="black", lwd=2, lty=2)
}
v_all <- onset_df$delay_green_to_red[is.finite(onset_df$delay_green_to_red)]
hist(v_all, breaks=40, col="grey70", border="white",
     main=sprintf("Combined\nn=%d, median=%.0f min", length(v_all), median(v_all)),
     xlab="GFP → mCherry delay (min)", ylab="Cells")
abline(v=median(v_all), col="black", lwd=2, lty=2)
mtext("Green to red delay", outer=TRUE, cex=1, font=2)
dev.off()
cat("Saved figures/combined/green_to_red_delay.png\n")

# normalized channel dynamics — combined
spots_m <- merge(
  spots[, c("Track.ID","dataset","Frame","T..sec.","ch2_corrected","Mean.ch3")],
  nuc[,   c("Track.ID","Frame","Mean.ch1")],
  by=c("Track.ID","Frame"), all.x=TRUE)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

sc01 <- function(x) {
  r <- range(x, na.rm=TRUE)
  if (diff(r) > 1e-9) (x - r[1]) / diff(r) else rep(NA_real_, length(x))
}

norm_list <- lapply(split(spots_m, spots_m$Track.ID), function(df) {
  df <- df[order(df$Frame), ]
  data.frame(dataset    = df$dataset[1],
             norm_frame = seq_len(nrow(df)) - 1L,
             gfp_norm   = sc01(df$ch2_corrected),
             bfp_norm   = sc01(df$Mean.ch1),
             red_norm   = sc01(df$Mean.ch3))
})
norm_df <- do.call(rbind, norm_list)

pop <- do.call(rbind, lapply(split(norm_df, norm_df$norm_frame), function(d) {
  data.frame(norm_frame=d$norm_frame[1],
             mean_gfp=mean(d$gfp_norm,na.rm=TRUE), sem_gfp=sd(d$gfp_norm,na.rm=TRUE)/sqrt(sum(!is.na(d$gfp_norm))),
             mean_bfp=mean(d$bfp_norm,na.rm=TRUE), sem_bfp=sd(d$bfp_norm,na.rm=TRUE)/sqrt(sum(!is.na(d$bfp_norm))),
             mean_red=mean(d$red_norm,na.rm=TRUE),  sem_red=sd(d$red_norm,na.rm=TRUE)/sqrt(sum(!is.na(d$red_norm))),
             n=sum(!is.na(d$gfp_norm)))
}))
pop <- pop[order(pop$norm_frame), ]

draw_ribbon <- function(x, mid, err, col_rgb) {
  ok <- is.finite(mid) & is.finite(err) & !is.na(x)
  if (sum(ok) < 2) return(invisible(NULL))
  polygon(c(x[ok], rev(x[ok])),
          c(mid[ok] + err[ok], rev(mid[ok] - err[ok])),
          col=col_rgb, border=NA)
}

png(file.path(fig_comb, "norm_channels.png"), width=900, height=550, type="cairo")
ylim <- range(c(pop$mean_gfp-pop$sem_gfp, pop$mean_gfp+pop$sem_gfp,
                pop$mean_bfp-pop$sem_bfp, pop$mean_bfp+pop$sem_bfp,
                pop$mean_red-pop$sem_red, pop$mean_red+pop$sem_red), na.rm=TRUE)
plot(pop$norm_frame, pop$mean_gfp, type="l", col="green3", lwd=2, ylim=ylim,
     xlab="Normalized frame", ylab="Normalized intensity (0-1, per cell)",
     main="A2+A3 combined — population mean normalized channels\n(each cell 0-1 per channel; mean ± SEM)")
draw_ribbon(pop$norm_frame, pop$mean_gfp, pop$sem_gfp, rgb(0,.7,0,.20))
lines(pop$norm_frame, pop$mean_gfp, col="green3", lwd=2)
lines(pop$norm_frame, pop$mean_bfp, col="dodgerblue", lwd=2)
draw_ribbon(pop$norm_frame, pop$mean_bfp, pop$sem_bfp, rgb(0,.4,1,.20))
lines(pop$norm_frame, pop$mean_bfp, col="dodgerblue", lwd=2)
lines(pop$norm_frame, pop$mean_red, col="tomato", lwd=2)
draw_ribbon(pop$norm_frame, pop$mean_red, pop$sem_red, rgb(1,.2,.2,.20))
lines(pop$norm_frame, pop$mean_red, col="tomato", lwd=2)
legend("topleft", bty="n", legend=c("Corrected GFP","BFP (nuclear)","mCherry"),
       col=c("green3","dodgerblue","tomato"), lwd=2)
dev.off()
cat("Saved figures/combined/norm_channels.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ELASTICNET + COX ON COMBINED DATA
# ══════════════════════════════════════════════════════════════════════════════
dataset <- combined   # for axis labels

max_obs  <- max(model_df$delay_green_to_red[is.finite(model_df$delay_green_to_red)], na.rm=TRUE)
model_df$y <- ifelse(is.finite(model_df$delay_green_to_red),
                     model_df$delay_green_to_red, max_obs * 1.1)

feat_cols <- setdiff(colnames(feat_df), c("Track.ID"))
feat_cols <- feat_cols[vapply(feat_cols,
  function(c) sum(is.finite(model_df[[c]])) >= 5, logical(1))]

X_raw <- as.matrix(model_df[, feat_cols])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[,j], na.rm=TRUE)
  X_raw[is.na(X_raw[,j]),j] <- if (is.finite(med_j)) med_j else 0
}
X <- scale(X_raw)
y <- model_df$y

set.seed(42)
cv_en    <- cv.glmnet(X, y, alpha=0.5, nfolds=10)
en_model <- glmnet(X, y, alpha=0.5, lambda=cv_en$lambda.min)

coefs    <- as.matrix(coef(en_model))
coefs_nz <- coefs[coefs[,1]!=0 & rownames(coefs)!="(Intercept)",, drop=FALSE]
coefs_nz <- coefs_nz[order(abs(coefs_nz[,1]), decreasing=TRUE),, drop=FALSE]
y_hat    <- as.numeric(predict(en_model, newx=X, s=cv_en$lambda.min))
r2       <- 1 - sum((y-y_hat)^2)/sum((y-mean(y))^2)
r_cor    <- cor(y, y_hat, use="complete.obs")
cat(sprintf("Combined ElasticNet: In-sample R²=%.3f  r=%.3f  non-zero=%d\n", r2, r_cor, nrow(coefs_nz)))

# ── outer CV for honest out-of-sample R² ──────────────────────────────────────
n_outer <- 10
set.seed(42)
outer_folds_c <- sample(rep(seq_len(n_outer), length.out = nrow(X)))
cv_preds_c    <- numeric(length(y))
for (k in seq_len(n_outer)) {
  test_idx  <- which(outer_folds_c == k)
  train_idx <- which(outer_folds_c != k)
  inner_cv  <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0.5, nfolds=5)
  fold_m    <- glmnet(X[train_idx,], y[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
  cv_preds_c[test_idx] <- as.numeric(predict(fold_m, newx=X[test_idx,], s=inner_cv$lambda.min))
}
r2_cv_c  <- 1 - sum((y - cv_preds_c)^2) / sum((y - mean(y))^2)
r_cv_c   <- cor(y, cv_preds_c, use="complete.obs")

prod_mask_c  <- is.finite(model_df$delay_green_to_red)
r2_cv_prod_c <- 1 - sum((y[prod_mask_c] - cv_preds_c[prod_mask_c])^2) /
                    sum((y[prod_mask_c] - mean(y[prod_mask_c]))^2)
r_cv_prod_c  <- cor(y[prod_mask_c], cv_preds_c[prod_mask_c], use="complete.obs")
cat(sprintf("Combined CV (%d-fold) R²=%.3f   prod-only CV R²=%.3f\n", n_outer, r2_cv_c, r2_cv_prod_c))

set.seed(99)
boot_feats <- replicate(50, {
  idx <- sample(nrow(X), replace=TRUE)
  cv  <- cv.glmnet(X[idx,], y[idx], alpha=0.5, nfolds=5)
  m   <- glmnet(X[idx,], y[idx], alpha=0.5, lambda=cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc!=0)]
}, simplify=FALSE)
freq_df <- sort(table(unlist(boot_feats)), decreasing=TRUE)
freq_df <- freq_df[names(freq_df)!="(Intercept)"]

# Cox
track_dur <- vapply(split(spots, spots$Track.ID),
                    function(df) (max(df$T..sec.)-min(df$T..sec.))/60, numeric(1))
cox_df           <- model_df
cox_df$event     <- as.integer(is.finite(cox_df$delay_green_to_red))
cox_df$surv_time <- ifelse(cox_df$event==1, cox_df$delay_green_to_red,
                           track_dur[as.character(cox_df$Track.ID)])
cox_df <- cox_df[is.finite(cox_df$surv_time) & cox_df$surv_time>0, ]
X_cox  <- X[match(rownames(cox_df), rownames(model_df)),, drop=FALSE]
surv_y <- Surv(cox_df$surv_time, cox_df$event)

cat(sprintf("Cox: %d cells | %d events | %d censored\n",
            nrow(cox_df), sum(cox_df$event), sum(cox_df$event==0)))

set.seed(42)
cv_cox <- cv.glmnet(X_cox, surv_y, family="cox", alpha=0.5, nfolds=10)
cox_en <- glmnet(X_cox, surv_y, family="cox", alpha=0.5, lambda=cv_cox$lambda.min)

cox_coefs    <- as.matrix(coef(cox_en))
cox_coefs_nz <- cox_coefs[cox_coefs[,1]!=0,, drop=FALSE]
cox_coefs_nz <- cox_coefs_nz[order(abs(cox_coefs_nz[,1]), decreasing=TRUE),, drop=FALSE]
risk_score   <- as.numeric(predict(cox_en, newx=X_cox, s=cv_cox$lambda.min, type="link"))
conc         <- concordance(surv_y ~ risk_score)
c_stat       <- max(conc$concordance, 1-conc$concordance)

set.seed(99)
boot_cox <- replicate(50, {
  idx <- sample(nrow(X_cox), replace=TRUE)
  cv  <- cv.glmnet(X_cox[idx,], surv_y[idx], family="cox", alpha=0.5, nfolds=5)
  m   <- glmnet(X_cox[idx,], surv_y[idx], family="cox", alpha=0.5, lambda=cv$lambda.min)
  cc  <- as.matrix(coef(m)); rownames(cc)[which(cc!=0)]
}, simplify=FALSE)
cox_freq <- sort(table(unlist(boot_cox)), decreasing=TRUE)

k_folds <- 5; set.seed(42)
fold_ids <- sample(rep(1:k_folds, length.out=nrow(X_cox)))
cv_risk  <- numeric(nrow(X_cox))
for (fold in seq_len(k_folds)) {
  test_idx <- which(fold_ids==fold); train_idx <- which(fold_ids!=fold)
  inner_cv <- cv.glmnet(X_cox[train_idx,], surv_y[train_idx], family="cox", alpha=0.5, nfolds=5)
  fm <- glmnet(X_cox[train_idx,], surv_y[train_idx], family="cox", alpha=0.5, lambda=inner_cv$lambda.min)
  cv_risk[test_idx] <- as.numeric(predict(fm, newx=X_cox[test_idx,], type="link"))
}
conc_cv   <- concordance(surv_y ~ cv_risk)
c_stat_cv <- max(conc_cv$concordance, 1-conc_cv$concordance)
risk_group_cv <- ifelse(cv_risk>median(cv_risk), "High risk", "Low risk")
km_fit_cv     <- survfit(surv_y ~ risk_group_cv)
logrank       <- survdiff(surv_y ~ risk_group_cv)
p_val         <- 1 - pchisq(logrank$chisq, df=1)
km_tbl        <- summary(km_fit_cv)$table
med_hi  <- km_tbl[grep("High",rownames(km_tbl)),"median"]
med_lo  <- km_tbl[grep("Low", rownames(km_tbl)),"median"]

cat(sprintf("Combined Cox — CV C=%.3f  log-rank p=%.4f\n", c_stat_cv, p_val))

# ── combined plots ─────────────────────────────────────────────────────────────
png(file.path(fig_comb, "en_coefs.png"), width=850, height=550, type="cairo")
par(mar=c(5,12,3,2))
if (nrow(coefs_nz)>0) {
  ord <- nrow(coefs_nz):1
  barplot(coefs_nz[ord,1], names.arg=rownames(coefs_nz)[ord], horiz=TRUE, las=1,
          col=ifelse(coefs_nz[ord,1]>0,"steelblue","tomato"),
          main=sprintf("A2+A3 ElasticNet (%d coefs)  CV R²=%.3f", nrow(coefs_nz), r2_cv_c),
          xlab="Coefficient (z-scored)")
  abline(v=0, col="grey50")
}
dev.off()

n_show <- min(20, length(freq_df))
png(file.path(fig_comb, "en_bootstrap.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
barplot(freq_df[n_show:1], names.arg=names(freq_df)[n_show:1],
        horiz=TRUE, las=1, col="steelblue",
        main="A2+A3 — ElasticNet bootstrap stability (n=50)",
        xlab="Times selected out of 50")
abline(v=25, col="red", lty=2)
dev.off()

png(file.path(fig_comb, "en_predicted_vs_actual.png"), width=600, height=600, type="cairo")
plot(y, cv_preds_c, pch=16, cex=0.5,
     col=ifelse(prod_mask_c, rgb(0,0.4,0.8,0.4), rgb(0.7,0.7,0.7,0.4)),
     xlab="Actual delay (min;  grey=censored)",
     ylab="Predicted delay (min)",
     main=sprintf("A2+A3 — %d-fold CV predicted vs actual\nCV R²=%.3f   prod-only CV R²=%.3f",
                  n_outer, r2_cv_c, r2_cv_prod_c))
abline(a=0, b=1, col="red", lty=2)
legend("topleft", bty="n",
       legend=c("Observed red","Censored"),
       col=c(rgb(0,0.4,0.8), rgb(0.7,0.7,0.7)), pch=16)
dev.off()

png(file.path(fig_comb, "cox_coefs.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
if (nrow(cox_coefs_nz)>0) {
  ord <- nrow(cox_coefs_nz):1
  barplot(cox_coefs_nz[ord,1], names.arg=rownames(cox_coefs_nz)[ord],
          horiz=TRUE, las=1, col=ifelse(cox_coefs_nz[ord,1]>0,"tomato","steelblue"),
          main=sprintf("A2+A3 Cox coefs (%d non-zero)  C=%.3f", nrow(cox_coefs_nz), c_stat),
          xlab="log Hazard Ratio (per SD)")
  abline(v=0, col="grey50")
}
dev.off()

n_show <- min(20, length(cox_freq))
png(file.path(fig_comb, "cox_bootstrap.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
barplot(cox_freq[n_show:1], names.arg=names(cox_freq)[n_show:1],
        horiz=TRUE, las=1, col="steelblue",
        main="A2+A3 — Cox bootstrap stability (n=50)",
        xlab="Times selected out of 50")
abline(v=25, col="red", lty=2)
dev.off()

png(file.path(fig_comb, "cox_km.png"), width=700, height=600, type="cairo")
plot(km_fit_cv, col=c("tomato","steelblue"), lwd=2,
     xlab="Time from green onset (min)", ylab="P(not yet red)",
     main=sprintf("A2+A3 KM %d-fold CV  |  C=%.3f  |  p=%.4f", k_folds, c_stat_cv, p_val))
legend("topright", bty="n",
       legend=c(sprintf("High risk (n=%d, median=%.0f min)", sum(risk_group_cv=="High risk"), med_hi),
                sprintf("Low risk  (n=%d, median=%.0f min)", sum(risk_group_cv=="Low risk"),  med_lo)),
       col=c("tomato","steelblue"), lwd=2)
dev.off()

cat("Saved figures/combined/*.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COX WITH GREEN-NORMALIZED TIME (t=0 at green onset per cell)
# ══════════════════════════════════════════════════════════════════════════════
# Biologically correct: time origin is when each cell turns green (IE stage).
# Only cells where green onset is detected are included.
# surv_time = time from green onset to red onset (event) or track end (censored).

cat("\n── Cox with green-normalized time ──\n")

# join green and red onset times to model_df
onset_slim <- onset_df[, c("Track.ID", "green_onset_min", "red_onset_min")]

cox_norm <- merge(model_df, onset_slim, by="Track.ID", all.x=TRUE)
# keep only cells with a detected green onset
cox_norm <- cox_norm[!is.na(cox_norm$green_onset_min), ]

# survival time measured FROM green onset
track_dur_min <- track_dur[as.character(cox_norm$Track.ID)]
cox_norm$event     <- as.integer(!is.na(cox_norm$red_onset_min))
cox_norm$surv_time <- ifelse(
  cox_norm$event == 1,
  cox_norm$red_onset_min   - cox_norm$green_onset_min,   # green→red delay
  track_dur_min            - cox_norm$green_onset_min    # green→track end
)
cox_norm <- cox_norm[is.finite(cox_norm$surv_time) & cox_norm$surv_time > 0, ]

cat(sprintf("Green-norm Cox: %d cells | %d events | %d censored\n",
            nrow(cox_norm), sum(cox_norm$event), sum(cox_norm$event == 0)))

X_norm  <- X[match(cox_norm$Track.ID, model_df$Track.ID), , drop=FALSE]
surv_yn <- Surv(cox_norm$surv_time, cox_norm$event)

set.seed(42)
cv_cox_n  <- cv.glmnet(X_norm, surv_yn, family="cox", alpha=0.5, nfolds=10)
cox_en_n  <- glmnet(X_norm, surv_yn, family="cox", alpha=0.5, lambda=cv_cox_n$lambda.min)

cox_coefs_n    <- as.matrix(coef(cox_en_n))
cox_coefs_nz_n <- cox_coefs_n[cox_coefs_n[,1] != 0, , drop=FALSE]
cox_coefs_nz_n <- cox_coefs_nz_n[order(abs(cox_coefs_nz_n[,1]), decreasing=TRUE), , drop=FALSE]

risk_n <- as.numeric(predict(cox_en_n, newx=X_norm, s=cv_cox_n$lambda.min, type="link"))
conc_n <- concordance(surv_yn ~ risk_n)
c_n    <- max(conc_n$concordance, 1 - conc_n$concordance)

# 5-fold CV
set.seed(42)
fold_ids_n <- sample(rep(1:k_folds, length.out=nrow(X_norm)))
cv_risk_n  <- numeric(nrow(X_norm))
for (fold in seq_len(k_folds)) {
  ti <- which(fold_ids_n == fold); tri <- which(fold_ids_n != fold)
  icv <- cv.glmnet(X_norm[tri,], surv_yn[tri], family="cox", alpha=0.5, nfolds=5)
  fm  <- glmnet(X_norm[tri,], surv_yn[tri], family="cox", alpha=0.5, lambda=icv$lambda.min)
  cv_risk_n[ti] <- as.numeric(predict(fm, newx=X_norm[ti,], type="link"))
}
conc_cv_n   <- concordance(surv_yn ~ cv_risk_n)
c_stat_cv_n <- max(conc_cv_n$concordance, 1 - conc_cv_n$concordance)
rg_n        <- ifelse(cv_risk_n > median(cv_risk_n), "High risk", "Low risk")
km_n        <- survfit(surv_yn ~ rg_n)
lr_n        <- survdiff(surv_yn ~ rg_n)
p_n         <- 1 - pchisq(lr_n$chisq, df=1)
kt_n        <- summary(km_n)$table
mhi_n       <- kt_n[grep("High", rownames(kt_n)), "median"]
mlo_n       <- kt_n[grep("Low",  rownames(kt_n)), "median"]

cat(sprintf("Green-norm Cox CV C=%.3f  log-rank p=%.4f\n", c_stat_cv_n, p_n))
cat(sprintf("Non-zero coefs: %d\n", nrow(cox_coefs_nz_n)))
print(round(cox_coefs_nz_n, 4))

# ── plots (new figures, old ones untouched) ────────────────────────────────────
png(file.path(fig_comb, "cox_km_green_norm.png"), width=700, height=600, type="cairo")
plot(km_n, col=c("tomato","steelblue"), lwd=2,
     xlab="Time from GFP onset (min)", ylab="P(not yet red)",
     main=sprintf("A2+A3 KM — time from green onset  |  %d-fold CV\nC=%.3f  |  p=%.4f",
                  k_folds, c_stat_cv_n, p_n))
legend("topright", bty="n",
       legend=c(sprintf("High risk (n=%d, median=%.0f min)", sum(rg_n=="High risk"), mhi_n),
                sprintf("Low risk  (n=%d, median=%.0f min)", sum(rg_n=="Low risk"),  mlo_n)),
       col=c("tomato","steelblue"), lwd=2)
dev.off()

png(file.path(fig_comb, "cox_coefs_green_norm.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
if (nrow(cox_coefs_nz_n) > 0) {
  ord <- nrow(cox_coefs_nz_n):1
  barplot(cox_coefs_nz_n[ord,1], names.arg=rownames(cox_coefs_nz_n)[ord],
          horiz=TRUE, las=1,
          col=ifelse(cox_coefs_nz_n[ord,1] > 0, "tomato", "steelblue"),
          main=sprintf("A2+A3 Cox (green-norm) — %d coefs  C=%.3f\nblue=faster to red",
                       nrow(cox_coefs_nz_n), c_n),
          xlab="log Hazard Ratio (per SD)")
  abline(v=0, col="grey50")
}
dev.off()

cat("Saved figures/combined/cox_km_green_norm.png + cox_coefs_green_norm.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4. EARLY-GREEN SUBGROUP: cells that turn green within first 1000 min
# ElasticNet + Cox re-run on this subset; new figures only (old ones untouched)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n── Early-green subgroup (green onset ≤ 1000 min) ──\n")

early_ids  <- onset_df$Track.ID[!is.na(onset_df$green_onset_min) & onset_df$green_onset_min <= 1000]
eg_model   <- model_df[model_df$Track.ID %in% early_ids, ]

cat(sprintf("Early-green: %d / %d cells  |  %d red events\n",
            nrow(eg_model), nrow(model_df),
            sum(is.finite(eg_model$delay_green_to_red))))

# build X/y for this subset
max_obs_eg  <- max(eg_model$delay_green_to_red[is.finite(eg_model$delay_green_to_red)], na.rm=TRUE)
eg_model$y  <- ifelse(is.finite(eg_model$delay_green_to_red),
                      eg_model$delay_green_to_red, max_obs_eg * 1.1)
feat_cols_eg <- setdiff(colnames(feat_df), "Track.ID")
feat_cols_eg <- feat_cols_eg[vapply(feat_cols_eg,
  function(c) sum(is.finite(eg_model[[c]])) >= 5, logical(1))]

X_eg_raw <- as.matrix(eg_model[, feat_cols_eg])
for (j in seq_len(ncol(X_eg_raw))) {
  med_j <- median(X_eg_raw[,j], na.rm=TRUE)
  X_eg_raw[is.na(X_eg_raw[,j]),j] <- if (is.finite(med_j)) med_j else 0
}
X_eg <- scale(X_eg_raw)
y_eg <- eg_model$y

# ElasticNet
set.seed(42)
cv_en_eg    <- cv.glmnet(X_eg, y_eg, alpha=0.5, nfolds=10)
en_eg       <- glmnet(X_eg, y_eg, alpha=0.5, lambda=cv_en_eg$lambda.min)
coefs_eg    <- as.matrix(coef(en_eg))
coefs_nz_eg <- coefs_eg[coefs_eg[,1]!=0 & rownames(coefs_eg)!="(Intercept)",, drop=FALSE]
coefs_nz_eg <- coefs_nz_eg[order(abs(coefs_nz_eg[,1]), decreasing=TRUE),, drop=FALSE]
y_hat_eg    <- as.numeric(predict(en_eg, newx=X_eg, s=cv_en_eg$lambda.min))
r2_eg       <- 1 - sum((y_eg-y_hat_eg)^2) / sum((y_eg-mean(y_eg))^2)
r_eg        <- cor(y_eg, y_hat_eg, use="complete.obs")
cat(sprintf("Early-green ElasticNet: In-sample R²=%.3f  r=%.3f  non-zero=%d\n", r2_eg, r_eg, nrow(coefs_nz_eg)))

# outer CV for early-green
set.seed(42)
outer_folds_eg2 <- sample(rep(seq_len(n_outer), length.out = nrow(X_eg)))
cv_preds_eg2    <- numeric(length(y_eg))
for (k in seq_len(n_outer)) {
  test_idx  <- which(outer_folds_eg2 == k)
  train_idx <- which(outer_folds_eg2 != k)
  inner_cv  <- cv.glmnet(X_eg[train_idx,], y_eg[train_idx], alpha=0.5, nfolds=5)
  fold_m    <- glmnet(X_eg[train_idx,], y_eg[train_idx], alpha=0.5, lambda=inner_cv$lambda.min)
  cv_preds_eg2[test_idx] <- as.numeric(predict(fold_m, newx=X_eg[test_idx,], s=inner_cv$lambda.min))
}
r2_cv_eg2  <- 1 - sum((y_eg - cv_preds_eg2)^2) / sum((y_eg - mean(y_eg))^2)
r_cv_eg2   <- cor(y_eg, cv_preds_eg2, use="complete.obs")
prod_mask_eg2  <- is.finite(eg_model$delay_green_to_red)
r2_cv_prod_eg2 <- 1 - sum((y_eg[prod_mask_eg2] - cv_preds_eg2[prod_mask_eg2])^2) /
                      sum((y_eg[prod_mask_eg2] - mean(y_eg[prod_mask_eg2]))^2)
cat(sprintf("Early-green CV (%d-fold) R²=%.3f   prod-only CV R²=%.3f\n", n_outer, r2_cv_eg2, r2_cv_prod_eg2))

# Cox
track_dur_eg   <- track_dur[as.character(eg_model$Track.ID)]
cox_eg         <- eg_model
cox_eg$event   <- as.integer(is.finite(cox_eg$delay_green_to_red))
cox_eg$surv_time <- ifelse(cox_eg$event==1, cox_eg$delay_green_to_red,
                           track_dur_eg[as.character(cox_eg$Track.ID)])
cox_eg <- cox_eg[is.finite(cox_eg$surv_time) & cox_eg$surv_time > 0, ]
X_cox_eg  <- X_eg[match(rownames(cox_eg), rownames(eg_model)),, drop=FALSE]
surv_eg   <- Surv(cox_eg$surv_time, cox_eg$event)

set.seed(42)
cv_cox_eg  <- cv.glmnet(X_cox_eg, surv_eg, family="cox", alpha=0.5, nfolds=10)
cox_en_eg  <- glmnet(X_cox_eg, surv_eg, family="cox", alpha=0.5, lambda=cv_cox_eg$lambda.min)
coefs_cox_eg    <- as.matrix(coef(cox_en_eg))
coefs_cox_nz_eg <- coefs_cox_eg[coefs_cox_eg[,1]!=0,, drop=FALSE]
coefs_cox_nz_eg <- coefs_cox_nz_eg[order(abs(coefs_cox_nz_eg[,1]), decreasing=TRUE),, drop=FALSE]
risk_eg   <- as.numeric(predict(cox_en_eg, newx=X_cox_eg, s=cv_cox_eg$lambda.min, type="link"))
conc_eg   <- concordance(surv_eg ~ risk_eg)
c_eg      <- max(conc_eg$concordance, 1-conc_eg$concordance)

set.seed(42)
fold_ids_eg <- sample(rep(1:k_folds, length.out=nrow(X_cox_eg)))
cv_risk_eg  <- numeric(nrow(X_cox_eg))
for (fold in seq_len(k_folds)) {
  ti <- which(fold_ids_eg==fold); tri <- which(fold_ids_eg!=fold)
  icv <- cv.glmnet(X_cox_eg[tri,], surv_eg[tri], family="cox", alpha=0.5, nfolds=5)
  fm  <- glmnet(X_cox_eg[tri,], surv_eg[tri], family="cox", alpha=0.5, lambda=icv$lambda.min)
  cv_risk_eg[ti] <- as.numeric(predict(fm, newx=X_cox_eg[ti,], type="link"))
}
conc_cv_eg   <- concordance(surv_eg ~ cv_risk_eg)
c_cv_eg      <- max(conc_cv_eg$concordance, 1-conc_cv_eg$concordance)
rg_eg        <- ifelse(cv_risk_eg > median(cv_risk_eg), "High risk", "Low risk")
km_eg        <- survfit(surv_eg ~ rg_eg)
lr_eg        <- survdiff(surv_eg ~ rg_eg)
p_eg         <- 1 - pchisq(lr_eg$chisq, df=1)
kt_eg        <- summary(km_eg)$table
mhi_eg       <- kt_eg[grep("High", rownames(kt_eg)), "median"]
mlo_eg       <- kt_eg[grep("Low",  rownames(kt_eg)), "median"]
cat(sprintf("Early-green Cox CV C=%.3f  log-rank p=%.4f  non-zero coefs=%d\n",
            c_cv_eg, p_eg, nrow(coefs_cox_nz_eg)))

# plots
png(file.path(fig_comb, "en_coefs_early_green.png"), width=850, height=550, type="cairo")
par(mar=c(5,12,3,2))
if (nrow(coefs_nz_eg) > 0) {
  ord <- nrow(coefs_nz_eg):1
  barplot(coefs_nz_eg[ord,1], names.arg=rownames(coefs_nz_eg)[ord], horiz=TRUE, las=1,
          col=ifelse(coefs_nz_eg[ord,1]>0,"steelblue","tomato"),
          main=sprintf("A2+A3 ElasticNet — early green (≤1000 min)\n%d coefs  CV R²=%.3f  n=%d",
                       nrow(coefs_nz_eg), r2_cv_eg2, nrow(eg_model)),
          xlab="Coefficient (z-scored)")
  abline(v=0, col="grey50")
}
dev.off()

png(file.path(fig_comb, "cox_km_early_green.png"), width=700, height=600, type="cairo")
plot(km_eg, col=c("tomato","steelblue"), lwd=2,
     xlab="Time from track start (min)", ylab="P(not yet red)",
     main=sprintf("A2+A3 KM — early green (≤1000 min)  |  %d-fold CV\nC=%.3f  |  p=%.4f",
                  k_folds, c_cv_eg, p_eg))
legend("topright", bty="n",
       legend=c(sprintf("High risk (n=%d, median=%.0f min)", sum(rg_eg=="High risk"), mhi_eg),
                sprintf("Low risk  (n=%d, median=%.0f min)", sum(rg_eg=="Low risk"),  mlo_eg)),
       col=c("tomato","steelblue"), lwd=2)
dev.off()

png(file.path(fig_comb, "cox_coefs_early_green.png"), width=850, height=550, type="cairo")
par(mar=c(5,13,3,2))
if (nrow(coefs_cox_nz_eg) > 0) {
  ord <- nrow(coefs_cox_nz_eg):1
  barplot(coefs_cox_nz_eg[ord,1], names.arg=rownames(coefs_cox_nz_eg)[ord],
          horiz=TRUE, las=1, col=ifelse(coefs_cox_nz_eg[ord,1]>0,"tomato","steelblue"),
          main=sprintf("A2+A3 Cox — early green (≤1000 min)\n%d coefs  C=%.3f  blue=faster to red",
                       nrow(coefs_cox_nz_eg), c_eg),
          xlab="log Hazard Ratio (per SD)")
  abline(v=0, col="grey50")
}
dev.off()
cat("Saved figures/combined/en_coefs_early_green.png + cox_km_early_green.png + cox_coefs_early_green.png\n")

# ── save combined caches ───────────────────────────────────────────────────────
saveRDS(spots,    file.path(cache_comb, "spots_clean.rds"))
saveRDS(nuc,      file.path(cache_comb, "nuc_assigned.rds"))
saveRDS(onset_df, file.path(cache_comb, "onset_df.rds"))
saveRDS(feat_df,  file.path(cache_comb, "feat_df.rds"))
saveRDS(model_df, file.path(cache_comb, "model_df.rds"))
saveRDS(list(model=en_model, cv=cv_en, X=X, y=y, coefs_nz=coefs_nz,
             r2=r2, r_cor=r_cor, freq=freq_df,
             r2_cv=r2_cv_c, r_cv=r_cv_c, cv_preds=cv_preds_c,
             r2_cv_prod=r2_cv_prod_c, r_cv_prod=r_cv_prod_c),
        file.path(cache_comb, "en_results.rds"))
saveRDS(list(model=cox_en, cv=cv_cox, X_cox=X_cox, surv_y=surv_y,
             coefs_nz=cox_coefs_nz, c_stat=c_stat, c_stat_cv=c_stat_cv,
             p_val=p_val, cv_risk=cv_risk, risk_group_cv=risk_group_cv),
        file.path(cache_comb, "cox_results.rds"))
cat("Saved all combined caches.\n")

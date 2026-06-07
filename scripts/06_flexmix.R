options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/Rlibs", .libPaths()))
library(flexmix)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", "combined")
fig_dir   <- file.path(base_dir, "figures", "combined")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── load data ──────────────────────────────────────────────────────────────────
model_df <- readRDS(file.path(cache_dir, "model_df.rds"))
red_df   <- model_df[is.finite(model_df$delay_green_to_red), ]
cat(sprintf("Red cells: %d (A2=%d, A3=%d)\n",
            nrow(red_df), sum(red_df$dataset=="A2"), sum(red_df$dataset=="A3")))

# ── define early / medium / late by tertiles ──────────────────────────────────
q33 <- quantile(red_df$delay_green_to_red, 1/3)
q67 <- quantile(red_df$delay_green_to_red, 2/3)
red_df$group <- cut(red_df$delay_green_to_red,
                    breaks = c(-Inf, q33, q67, Inf),
                    labels = c("early", "medium", "late"))
cat(sprintf("Group cuts: early < %.0f min, medium %.0f–%.0f min, late > %.0f min\n",
            q33, q33, q67, q67))
print(table(red_df$group))

# ── feature matrix (same set as ElasticNet, scaled) ───────────────────────────
feat_cols <- c("gfp_corr_start","gfp_corr_mean","gfp_corr_sd","gfp_corr_slope",
               "nuc_bfp_start","nuc_bfp_mean","nuc_bfp_sd","nuc_bfp_slope",
               "nuc_area_mean","nuc_area_slope","nuc_circ_mean","nuc_circ_sd",
               "nuc_ratio_mean","nuc_ratio_slope",
               "area_start","area_mean","area_sd","area_slope",
               "solidity_mean","solidity_sd","shape_idx_mean",
               "gfp_snr_mean","gfp_snr_sd","bf_snr_mean","bf_ctrst_mean","bf_ctrst_sd",
               "gfp_ratio_start","gfp_ratio_mean","gfp_ratio_sd",
               "gfp_ratio_slope","gfp_ratio_max")
feat_cols <- intersect(feat_cols, names(red_df))

X_raw <- as.matrix(red_df[, feat_cols])
keep  <- apply(X_raw, 2, function(v) sum(is.finite(v)) == nrow(X_raw) && sd(v, na.rm=TRUE) > 0)
X_raw <- X_raw[, keep]
X_sc  <- scale(X_raw)

y_min  <- red_df$delay_green_to_red / 60   # hours
fit_df <- as.data.frame(X_sc)
fit_df$y <- y_min

# ── fit flexmix: K=3 components, each a linear regression ─────────────────────
set.seed(42)
fm <- flexmix(y ~ ., data = fit_df, k = 3,
              model  = FLXMRglm(family = "gaussian"),
              control = list(iter.max = 200, tol = 1e-5, verbose = 0))
cat(sprintf("\nFlexmix converged: %d iterations, log-lik=%.2f\n",
            fm@iter, logLik(fm)))
print(table(clusters(fm)))

# ── assign component labels to early/medium/late by component median ──────────
comp_medians <- tapply(y_min, clusters(fm), median)
comp_order   <- order(comp_medians)            # comp_order[1]=earliest component
group_labels <- c("early", "medium", "late")
comp_label   <- setNames(group_labels, names(sort(comp_medians)))
cell_label   <- comp_label[as.character(clusters(fm))]

cat("\nComponent medians (h):\n")
for (nm in names(comp_medians[comp_order]))
  cat(sprintf("  Component %s → %s  median=%.1f h  n=%d\n",
              nm, comp_label[nm], comp_medians[nm], sum(clusters(fm)==as.integer(nm))))

# ── in-sample predictions per component ──────────────────────────────────────
pred_y <- predict(fm)   # list of K matrices; pick posterior-weighted or hard assignment
# use posterior-averaged prediction
post   <- posterior(fm)                        # n × K posterior probabilities
pred_each <- do.call(cbind, lapply(pred_y, as.numeric))  # n × K predictions
pred_avg  <- rowSums(pred_each * post)         # soft-assigned prediction (h)
pred_hard <- pred_each[cbind(seq_len(nrow(fit_df)), clusters(fm))]  # hard-assigned (h)

r2_soft <- 1 - sum((y_min - pred_avg)^2)  / sum((y_min - mean(y_min))^2)
r2_hard <- 1 - sum((y_min - pred_hard)^2) / sum((y_min - mean(y_min))^2)
r_soft  <- cor(y_min, pred_avg,  use="complete.obs")
r_hard  <- cor(y_min, pred_hard, use="complete.obs")
cat(sprintf("\nIn-sample  (soft assignment): R²=%.3f  r=%.3f\n", r2_soft, r_soft))
cat(sprintf("In-sample  (hard assignment): R²=%.3f  r=%.3f\n", r2_hard, r_hard))

# ── plot ───────────────────────────────────────────────────────────────────────
GROUP_COLS <- c(early="#e67e22", medium="#2980b9", late="#27ae60")

png(file.path(fig_dir, "flexmix_predicted_vs_actual.png"),
    width=1100, height=420, res=110, type="cairo")
par(mfrow=c(1,3), mar=c(4,4,3.5,1.5), oma=c(0,0,2,0))

all_lim <- range(c(y_min, pred_hard), finite=TRUE) * c(0.95, 1.05)

for (grp in c("early", "medium", "late")) {
  mask <- cell_label == grp
  x_g  <- y_min[mask]
  y_g  <- pred_hard[mask]
  r2_g <- 1 - sum((x_g - y_g)^2) / sum((x_g - mean(x_g))^2)
  r_g  <- cor(x_g, y_g, use="complete.obs")
  col  <- GROUP_COLS[grp]

  plot(x_g, y_g,
       col  = adjustcolor(col, alpha.f=0.55),
       pch  = 16, cex = 0.8,
       xlim = all_lim, ylim = all_lim,
       xlab = "Actual delay (h)", ylab = "Predicted delay (h)",
       main = sprintf("%s  (n=%d)\nR²=%.2f  r=%.2f", grp, sum(mask), r2_g, r_g))
  abline(0, 1, lty=2, col="grey50", lwd=1.2)
  abline(lm(y_g ~ x_g), col=col, lwd=1.8)
}

mtext(sprintf("FlexMix (K=3) — A2+A3  |  soft-assignment overall: R²=%.2f  r=%.2f",
              r2_soft, r_soft),
      outer=TRUE, cex=0.85, font=2, line=0.3)
dev.off()
cat(sprintf("\nSaved figures/combined/flexmix_predicted_vs_actual.png\n"))

# ── also plot delay distributions per component ────────────────────────────────
png(file.path(fig_dir, "flexmix_component_distributions.png"),
    width=900, height=380, res=110, type="cairo")
par(mfrow=c(1,3), mar=c(4,4,3.5,1.5), oma=c(0,0,2,0))

for (grp in c("early", "medium", "late")) {
  mask <- cell_label == grp
  d    <- y_min[mask]
  col  <- GROUP_COLS[grp]
  true_grp_mask <- red_df$group == grp
  hist(d, breaks=25, col=adjustcolor(col, 0.7), border="white",
       main=sprintf("%s  (n=%d)\nmedian=%.1f h", grp, sum(mask), median(d)),
       xlab="Delay GFP → mCherry (h)", ylab="Cells")
  # overlay true tertile group in outline
  hist(y_min[true_grp_mask], breaks=25, add=TRUE,
       col=adjustcolor("black", 0.0), border=adjustcolor("black", 0.5), lty=2)
  abline(v=median(d), col=col, lwd=2, lty=2)
}

mtext("FlexMix component distributions (solid fill) vs tertile groups (dashed outline)",
      outer=TRUE, cex=0.8, line=0.3)
dev.off()
cat("Saved figures/combined/flexmix_component_distributions.png\n")

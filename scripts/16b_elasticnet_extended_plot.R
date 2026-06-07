options(bitmapType = "cairo")
.libPaths(c("/tmp/Rlibs_4.3", "/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"
res_dir  <- file.path(base_dir, "results", "elasticnet_extended")
fig_dir  <- file.path(base_dir, "figures", "combined")

res <- readRDS(file.path(res_dir, "en_extended_results.rds"))
ro  <- res$orig
re  <- res$ext

# ── 1. results tables ──────────────────────────────────────────────────────────
metrics <- data.frame(
  metric    = c("CV R² (all cells)", "CV r (all cells)",
                "CV R² (productive only)", "CV r (productive only)",
                "In-sample R²",
                "Non-zero coefs", "Features used"),
  original  = c(round(ro$r2_cv, 3), round(ro$r_cv, 3),
                round(ro$r2_cv_prod, 3), round(ro$r_cv_prod, 3),
                round(ro$r2_in, 3),
                nrow(ro$coefs_nz), length(ro$fc)),
  extended  = c(round(re$r2_cv, 3), round(re$r_cv, 3),
                round(re$r2_cv_prod, 3), round(re$r_cv_prod, 3),
                round(re$r2_in, 3),
                nrow(re$coefs_nz), length(re$fc))
)
metrics$delta <- metrics$extended - metrics$original
write.csv(metrics, file.path(res_dir, "metrics_comparison.csv"), row.names=FALSE)
cat("Saved metrics_comparison.csv\n")

# coefficients for both models (all features, zero-padded)
all_feats <- union(res$feat_cols_orig, res$feat_cols_ext)
coef_orig <- setNames(rep(0, length(all_feats)), all_feats)
coef_ext  <- setNames(rep(0, length(all_feats)), all_feats)
coef_orig[rownames(ro$coefs_nz)] <- ro$coefs_nz[, 1]
coef_ext[ rownames(re$coefs_nz)] <- re$coefs_nz[, 1]

coef_df <- data.frame(
  feature  = all_feats,
  coef_original = round(coef_orig[all_feats], 5),
  coef_extended = round(coef_ext[all_feats],  5),
  is_new    = !(all_feats %in% res$feat_cols_orig)
)
coef_df <- coef_df[order(abs(coef_df$coef_extended), decreasing=TRUE), ]
write.csv(coef_df, file.path(res_dir, "coefficients_comparison.csv"), row.names=FALSE)
cat("Saved coefficients_comparison.csv\n")

# ── 2. summary figure (3 panels) ──────────────────────────────────────────────
png(file.path(fig_dir, "elasticnet_extended_summary.png"),
    width=1400, height=900, type="cairo")

layout(matrix(c(1,1,2,3,4,4,4,4), nrow=2, byrow=TRUE),
       heights=c(1, 1.4))
par(oma=c(1,1,3,1))

# ── panel 1: metrics comparison bar chart ─────────────────────────────────────
par(mar=c(5, 9, 2, 2))
met_show <- metrics[1:5, ]   # skip count rows
met_mat  <- rbind(met_show$original, met_show$extended)
colnames(met_mat) <- met_show$metric

bp <- barplot(met_mat,
              beside=TRUE,
              horiz=TRUE,
              col=c("steelblue", "tomato"),
              las=1,
              xlim=c(min(met_mat) * 1.4, max(met_mat) * 1.15),
              xlab="Value",
              main="CV metrics: original vs extended features")
abline(v=0, col="grey40", lwd=0.8)
# delta labels
for (i in seq_len(ncol(met_mat))) {
  d   <- met_show$delta[i]
  col <- if (d > 0) "#27ae60" else "#e74c3c"
  x   <- max(met_mat[, i]) + 0.005
  y   <- mean(bp[, i])
  text(x, y, sprintf("%+.3f", d), col=col, cex=0.85, adj=0, font=2)
}
legend("bottomright", bty="n",
       legend=c("Original (29 feat)", "Extended (45 feat)"),
       fill=c("steelblue","tomato"), cex=0.9)

# ── panel 2: scatter original ──────────────────────────────────────────────────
par(mar=c(4, 4, 3, 1))
plot(ro$y / 60, ro$cv_preds / 60,
     pch=16, cex=0.45,
     col=ifelse(ro$prod_mask, rgb(0.16,0.50,0.73,0.35), rgb(0.7,0.7,0.7,0.35)),
     xlab="Actual (h)", ylab="Predicted (h)",
     main=sprintf("Original (29)\nCV R²=%.3f  r=%.3f", ro$r2_cv, ro$r_cv))
abline(0, 1, col="red", lty=2, lwd=1.2)

# ── panel 3: scatter extended ─────────────────────────────────────────────────
par(mar=c(4, 4, 3, 1))
plot(re$y / 60, re$cv_preds / 60,
     pch=16, cex=0.45,
     col=ifelse(re$prod_mask, rgb(0.73,0.18,0.18,0.35), rgb(0.7,0.7,0.7,0.35)),
     xlab="Actual (h)", ylab="Predicted (h)",
     main=sprintf("Extended (45)\nCV R²=%.3f  r=%.3f", re$r2_cv, re$r_cv))
abline(0, 1, col="red", lty=2, lwd=1.2)

# ── panel 4: coefficient comparison (extended model, coloured by model status) ─
par(mar=c(5, 13, 3, 7))

# all features non-zero in at least one model
nz_mask <- abs(coef_orig) > 1e-8 | abs(coef_ext) > 1e-8
nz_feats <- names(coef_orig)[nz_mask]

# sort by extended coef magnitude
ord <- order(abs(coef_ext[nz_feats]))
nz_feats_ord <- nz_feats[ord]
n_show <- length(nz_feats_ord)
ypos <- seq_len(n_show)

# draw extended bars, coloured blue=orig, red=new
bar_colors <- ifelse(nz_feats_ord %in% res$feat_cols_orig, "steelblue", "tomato")
barplot(coef_ext[nz_feats_ord], horiz=TRUE, las=1,
        col=bar_colors, border=NA,
        names.arg=nz_feats_ord, cex.names=0.82,
        xlab="Coefficient (extended model, z-scored features)",
        main="Extended model coefficients\n(red = new feature, blue = original feature)")
# overlay original as points with error bars
# (use coef_orig as points on same x-axis, y = bar positions)
bp2 <- barplot(coef_ext[nz_feats_ord], horiz=TRUE, las=1,
               col=bar_colors, border=NA,
               names.arg=nz_feats_ord, cex.names=0.82,
               xlab="Coefficient (extended model, z-scored features)",
               main="")
points(coef_orig[nz_feats_ord], bp2, pch=18, cex=1.1, col="black")
abline(v=0, col="grey30", lwd=0.8)
legend("bottomright", bty="n",
       legend=c("Extended coef", "Original coef (●)", "New feature", "Original feature"),
       fill=c(NA, NA, "tomato", "steelblue"),
       border=NA,
       pch=c(NA, 18, NA, NA),
       pt.cex=c(NA,1.1,NA,NA),
       col=c(NA,"black",NA,NA),
       merge=FALSE, cex=0.85)

mtext("Elastic Net Linear Regression — GFP→mCherry delay\nNo late bloomers, n=743 cells, 10-fold nested CV",
      outer=TRUE, cex=1.1, font=2, line=0.5)

dev.off()
cat("Saved figures/combined/elasticnet_extended_summary.png\n")

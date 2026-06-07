options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/Rlibs", .libPaths()))
library(flexmix)

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"
fig_dir  <- file.path(base_dir, "figures", "combined")

cv         <- readRDS(file.path(base_dir, "cache", "combined", "flexmix_cv_results.rds"))
fm         <- cv$model
comp_label <- cv$comp_label   # c("1"="early", "2"="medium", "3"="late")

GROUP_COLS <- c(early="#e67e22", medium="#2980b9", late="#27ae60")

# clean feature label lookup
feat_labels <- c(
  nuc_bfp_mean    = "Nuclear BFP mean",
  nuc_bfp_start   = "Nuclear BFP start",
  nuc_bfp_sd      = "Nuclear BFP SD",
  nuc_bfp_slope   = "Nuclear BFP slope",
  nuc_area_mean   = "Nucleus area mean",
  nuc_area_slope  = "Nucleus area slope",
  nuc_circ_mean   = "Nucleus circularity",
  nuc_ratio_mean  = "Nuc/cell area ratio",
  gfp_corr_start  = "GFP start",
  gfp_corr_slope  = "GFP slope",
  gfp_corr_sd     = "GFP SD",
  area_mean       = "Cell area mean",
  solidity_sd     = "Cell solidity SD",
  gfp_snr_sd      = "GFP SNR SD"
)

# extract top-5 features per component by absolute coefficient
top_n <- 5
comp_data <- lapply(seq_len(fm@k), function(k) {
  p      <- parameters(fm, component = k)
  rnames <- rownames(p)
  feat_r <- rnames[startsWith(rnames, "coef.") & rnames != "coef.(Intercept)"]
  coefs  <- setNames(p[feat_r, 1], sub("^coef\\.", "", feat_r))
  coefs_sorted <- sort(abs(coefs), decreasing = TRUE)
  top    <- names(coefs_sorted)[seq_len(min(top_n, length(coefs_sorted)))]
  data.frame(
    feature = top,
    label   = feat_labels[top],
    coef    = coefs[top],
    group   = comp_label[as.character(k)],
    stringsAsFactors = FALSE
  )
})

# ── plot: one panel per group ──────────────────────────────────────────────────
png(file.path(fig_dir, "flexmix_top5_coefs.png"),
    width = 1050, height = 420, res = 115, type = "cairo")
par(mfrow = c(1, 3), mar = c(4, 9, 3.5, 1.5), oma = c(0, 0, 2, 0))

# shared x limits across all panels for comparability
all_coefs <- unlist(lapply(comp_data, `[[`, "coef"))
xlim <- max(abs(all_coefs), na.rm = TRUE) * c(-1.25, 1.25)

for (d in comp_data) {
  grp  <- d$group[1]
  col  <- GROUP_COLS[grp]
  labs <- rev(d$label)
  vals <- rev(d$coef)
  bar_cols <- ifelse(vals >= 0, adjustcolor(col, 0.85), adjustcolor("grey40", 0.7))

  bp <- barplot(vals,
                horiz    = TRUE,
                names.arg = labs,
                las      = 1,
                col      = bar_cols,
                border   = NA,
                xlim     = xlim,
                xlab     = "Coefficient (standardised features, log outcome)",
                main     = sprintf("%s component\n(top %d features by |coef|)", grp, top_n),
                cex.names = 0.82,
                cex.axis  = 0.8)
  abline(v = 0, col = "grey30", lwd = 1)
  # value labels
  text(vals + sign(vals) * diff(xlim) * 0.03,
       bp,
       labels = sprintf("%.3f", vals),
       cex = 0.72, adj = ifelse(vals >= 0, 0, 1))
}

mtext("FlexMix component coefficients — top 5 features per group",
      outer = TRUE, cex = 0.88, font = 2, line = 0.3)
dev.off()
cat("Saved figures/combined/flexmix_top5_coefs.png\n")

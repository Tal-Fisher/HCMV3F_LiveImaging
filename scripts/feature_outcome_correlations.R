options(bitmapType = "cairo")

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"

feat  <- readRDS(file.path(base_dir, "cache", "combined", "feat_df.rds"))
onset <- readRDS(file.path(base_dir, "cache", "combined", "onset_df.rds"))

# compute delays and merge
onset$delay_blue_to_red  <- onset$red_onset_min  - onset$blue_onset_min
onset$delay_green_to_red <- onset$red_onset_min  - onset$green_onset_min

df <- merge(feat, onset[, c("Track.ID", "delay_green_to_red", "delay_blue_to_red")],
            by = "Track.ID", all.x = TRUE)

# filter: remove cells where red comes before blue
df <- df[is.finite(df$delay_blue_to_red) & df$delay_blue_to_red >= 0, ]
cat(sprintf("Cells after filtering red-before-blue: %d\n", nrow(df)))

# ── scatter plot helper ────────────────────────────────────────────────────────
scatter_cor <- function(x, y, xlab, ylab, col = "steelblue") {
  keep <- is.finite(x) & is.finite(y)
  xk <- x[keep]; yk <- y[keep]
  if (length(xk) < 5) { plot.new(); return(invisible(NULL)) }

  r_p <- cor(xk, yk, method = "pearson")
  r_s <- cor(xk, yk, method = "spearman")

  plot(xk, yk, pch = 16, cex = 0.5, col = adjustcolor(col, 0.4),
       xlab = xlab, ylab = ylab,
       main = sprintf("r = %.3f  |  ρ = %.3f  (n = %d)", r_p, r_s, length(xk)))
  abline(lm(yk ~ xk), col = "black", lwd = 1.5)
}

outcomes <- list(
  list(var = "delay_green_to_red", label = "Green→Red delay (min)"),
  list(var = "delay_blue_to_red",  label = "Blue→Red delay (min)")
)

# ── PAGE 1: start features ─────────────────────────────────────────────────────
png(file.path(base_dir, "figures", "combined", "feature_outcome_start.png"),
    width = 1200, height = 1100, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 2), oma = c(0, 0, 3, 0))

scatter_cor(df$nuc_bfp_start,  df$delay_green_to_red,
            "BFP start", "Green→Red delay (min)", col = "dodgerblue")
scatter_cor(df$nuc_bfp_start,  df$delay_blue_to_red,
            "BFP start", "Blue→Red delay (min)",  col = "dodgerblue")
scatter_cor(df$gfp_corr_start, df$delay_green_to_red,
            "GFP start", "Green→Red delay (min)", col = "green3")
scatter_cor(df$gfp_corr_start, df$delay_blue_to_red,
            "GFP start", "Blue→Red delay (min)",  col = "green3")

mtext("Feature–outcome correlations  (start values)  —  red-before-blue excluded",
      outer = TRUE, cex = 1.1, font = 2)
dev.off()
cat("Saved feature_outcome_start.png\n")

# ── PAGE 2: mean features ──────────────────────────────────────────────────────
png(file.path(base_dir, "figures", "combined", "feature_outcome_mean.png"),
    width = 1200, height = 1100, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 2), oma = c(0, 0, 3, 0))

scatter_cor(df$nuc_bfp_mean,  df$delay_green_to_red,
            "BFP mean", "Green→Red delay (min)", col = "dodgerblue")
scatter_cor(df$nuc_bfp_mean,  df$delay_blue_to_red,
            "BFP mean", "Blue→Red delay (min)",  col = "dodgerblue")
scatter_cor(df$gfp_corr_mean, df$delay_green_to_red,
            "GFP mean", "Green→Red delay (min)", col = "green3")
scatter_cor(df$gfp_corr_mean, df$delay_blue_to_red,
            "GFP mean", "Blue→Red delay (min)",  col = "green3")

mtext("Feature–outcome correlations  (mean values)  —  red-before-blue excluded",
      outer = TRUE, cex = 1.1, font = 2)
dev.off()
cat("Saved feature_outcome_mean.png\n")

# ── PAGE 3: GFP/BFP ratio features ────────────────────────────────────────────
png(file.path(base_dir, "figures", "combined", "feature_outcome_ratio.png"),
    width = 1200, height = 1100, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 2), oma = c(0, 0, 3, 0))

scatter_cor(df$gfp_ratio_start, df$delay_green_to_red,
            "GFP/BFP ratio (start)", "Green→Red delay (min)", col = "darkorchid")
scatter_cor(df$gfp_ratio_start, df$delay_blue_to_red,
            "GFP/BFP ratio (start)", "Blue→Red delay (min)",  col = "darkorchid")
scatter_cor(df$gfp_ratio_mean,  df$delay_green_to_red,
            "GFP/BFP ratio (mean)",  "Green→Red delay (min)", col = "darkorchid")
scatter_cor(df$gfp_ratio_mean,  df$delay_blue_to_red,
            "GFP/BFP ratio (mean)",  "Blue→Red delay (min)",  col = "darkorchid")

mtext("Feature–outcome correlations  (GFP/BFP ratio)  —  red-before-blue excluded",
      outer = TRUE, cex = 1.1, font = 2)
dev.off()
cat("Saved feature_outcome_ratio.png\n")

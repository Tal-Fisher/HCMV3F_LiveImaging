options(bitmapType = "cairo")

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache",   "combined")
fig_comb   <- file.path(base_dir, "figures", "combined")

spots <- readRDS(file.path(cache_comb, "spots_clean.rds"))
nuc   <- readRDS(file.path(cache_comb, "nuc_assigned.rds"))

spots_m <- merge(
  spots[, c("Track.ID", "Frame", "T..sec.", "ch2_corrected")],
  nuc[,   c("Track.ID", "Frame", "Mean.ch1")],
  by = c("Track.ID", "Frame"), all.x = TRUE
)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

# align to track start
spots_m$norm_frame <- ave(spots_m$Frame, spots_m$Track.ID,
                          FUN = function(f) f - min(f))

# at each norm_frame: Pearson r between GFP and BFP across cells
min_n <- 15   # require at least 15 cells for a stable estimate

cor_df <- do.call(rbind, lapply(split(spots_m, spots_m$norm_frame), function(d) {
  ok <- is.finite(d$ch2_corrected) & is.finite(d$Mean.ch1)
  n  <- sum(ok)
  if (n < min_n) return(NULL)
  r  <- cor(d$ch2_corrected[ok], d$Mean.ch1[ok], method = "pearson")
  rs <- cor(d$ch2_corrected[ok], d$Mean.ch1[ok], method = "spearman")
  data.frame(norm_frame = d$norm_frame[1], pearson = r, spearman = rs, n = n)
}))
cor_df <- cor_df[order(cor_df$norm_frame), ]

cat(sprintf("Frames included: %d  (n >= %d cells)\n", nrow(cor_df), min_n))
cat(sprintf("Pearson r range: %.3f to %.3f\n", min(cor_df$pearson), max(cor_df$pearson)))

png(file.path(fig_comb, "rolling_correlation_gfp_bfp.png"),
    width = 1000, height = 700, type = "cairo")

par(mfrow = c(2, 1), mar = c(2, 5, 3, 2), oma = c(3, 0, 3, 0))

# panel 1: correlation
plot(cor_df$norm_frame, cor_df$pearson,
     type = "l", col = "steelblue", lwd = 2,
     ylim = c(-1, 1),
     xlab = "", ylab = "Correlation (r)",
     main = "GFP vs BFP cross-sectional correlation at each frame")
lines(cor_df$norm_frame, cor_df$spearman, col = "tomato", lwd = 2, lty = 2)
abline(h = 0, col = "grey60", lty = 3)
legend("topright", bty = "n",
       legend = c("Pearson r", "Spearman ρ"),
       col = c("steelblue", "tomato"), lwd = 2, lty = c(1, 2))

# panel 2: n cells contributing
par(mar = c(4, 5, 1, 2))
plot(cor_df$norm_frame, cor_df$n,
     type = "l", col = "grey40", lwd = 1.5,
     xlab = "Frames from track start",
     ylab = "n cells",
     main = "")
abline(h = min_n, col = "tomato", lty = 2)

mtext("Cross-sectional GFP–BFP correlation — aligned to track start (frame 0 = first detection)",
      outer = TRUE, cex = 1.0, font = 2)

dev.off()
cat("Saved rolling_correlation_gfp_bfp.png\n")

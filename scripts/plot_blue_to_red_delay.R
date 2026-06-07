options(bitmapType = "cairo")

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"
onset    <- readRDS(file.path(base_dir, "cache", "combined", "onset_df.rds"))

onset$delay_blue_to_red <- onset$red_onset_min - onset$blue_onset_min
d <- onset$delay_blue_to_red[is.finite(onset$delay_blue_to_red)]

png(file.path(base_dir, "figures", "combined", "blue_to_red_delay_dist.png"),
    width = 900, height = 600, type = "cairo")

par(mar = c(5, 4, 4, 2))

hist(d, breaks = 60, col = "mediumpurple", border = "white",
     xlab = "Blue → Red delay (min)",
     ylab = "Number of cells",
     main = sprintf("Distribution of blue-to-red delay\n(n = %d cells with both onsets detected)", length(d)))

abline(v = median(d), col = "black",  lwd = 2, lty = 2)
abline(v = 0,         col = "tomato", lwd = 1.5, lty = 3)

legend("topright", bty = "n",
       legend = c(sprintf("Median = %.0f min", median(d)),
                  sprintf("Negative (red before blue): n = %d", sum(d < 0))),
       lty = c(2, 3), col = c("black", "tomato"), lwd = 2)

dev.off()
cat("Saved figures/combined/blue_to_red_delay_dist.png\n")
cat(sprintf("n = %d  |  median = %.0f min  |  range = %.0f – %.0f min\n",
            length(d), median(d), min(d), max(d)))

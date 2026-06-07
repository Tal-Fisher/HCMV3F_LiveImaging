options(bitmapType = "cairo")

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache",   "combined")
fig_comb   <- file.path(base_dir, "figures", "combined")

spots <- readRDS(file.path(cache_comb, "spots_clean.rds"))
nuc   <- readRDS(file.path(cache_comb, "nuc_assigned.rds"))

spots_m <- merge(
  spots[, c("Track.ID", "Frame", "T..sec.", "Mean.ch2", "Mean.ch3")],
  nuc[,   c("Track.ID", "Frame", "Mean.ch1")],
  by = c("Track.ID", "Frame"), all.x = TRUE
)

# population mean at each timepoint (all cells present at that time)
pop_t <- do.call(rbind, lapply(split(spots_m, spots_m$T..sec.), function(d) {
  data.frame(
    t_min   = d$T..sec.[1] / 60,
    mean_gfp = mean(d$Mean.ch2,  na.rm = TRUE),
    mean_bfp = mean(d$Mean.ch1,  na.rm = TRUE),
    mean_red = mean(d$Mean.ch3,  na.rm = TRUE),
    n_cells  = sum(!is.na(d$Mean.ch2))
  )
}))
pop_t <- pop_t[order(pop_t$t_min), ]

# rolling mean helper (window = 5 timepoints)
roll <- function(x, k = 5) {
  n <- length(x)
  vapply(seq_len(n), function(i) {
    v <- x[max(1, i - floor(k/2)) : min(n, i + floor(k/2))]
    mean(v, na.rm = TRUE)
  }, numeric(1))
}

pop_t$r_gfp <- roll(pop_t$mean_gfp)
pop_t$r_bfp <- roll(pop_t$mean_bfp)
pop_t$r_red <- roll(pop_t$mean_red)

# baseline-subtract each channel (subtract first value so all start at 0)
pop_t$r_gfp_bs <- pop_t$r_gfp - pop_t$r_gfp[1]
pop_t$r_bfp_bs <- pop_t$r_bfp - pop_t$r_bfp[1]
pop_t$r_red_bs <- pop_t$r_red - pop_t$r_red[1]

png(file.path(fig_comb, "population_raw_channels.png"),
    width = 1400, height = 1000, type = "cairo")
par(mfrow = c(2, 1), mar = c(4, 5, 3, 2), oma = c(0, 0, 3, 0))

# ── panel 1: raw mean intensities ─────────────────────────────────────────────
ylim1 <- range(c(pop_t$r_gfp, pop_t$r_bfp, pop_t$r_red), na.rm = TRUE)
plot(pop_t$t_min, pop_t$r_gfp,
     type = "l", col = "green3", lwd = 2, ylim = ylim1,
     xlab = "Time (min)", ylab = "Mean intensity (raw)",
     main = "Raw mean intensities — population average")
lines(pop_t$t_min, pop_t$r_bfp, col = "dodgerblue", lwd = 2)
lines(pop_t$t_min, pop_t$r_red, col = "tomato",     lwd = 2)
legend("topleft", bty = "n",
       legend = c(sprintf("GFP (ch2)  n=%d cells max", max(pop_t$n_cells)),
                  "BFP nuclear (ch1)",
                  "mCherry (ch3)"),
       col = c("green3", "dodgerblue", "tomato"), lwd = 2)

# ── panel 2: baseline-subtracted ──────────────────────────────────────────────
ylim2 <- range(c(pop_t$r_gfp_bs, pop_t$r_bfp_bs, pop_t$r_red_bs), na.rm = TRUE)
plot(pop_t$t_min, pop_t$r_gfp_bs,
     type = "l", col = "green3", lwd = 2, ylim = ylim2,
     xlab = "Time (min)", ylab = "Change from baseline",
     main = "Baseline-subtracted (each channel starts at 0)")
lines(pop_t$t_min, pop_t$r_bfp_bs, col = "dodgerblue", lwd = 2)
lines(pop_t$t_min, pop_t$r_red_bs, col = "tomato",     lwd = 2)
abline(h = 0, col = "grey70", lty = 2)
legend("topleft", bty = "n",
       legend = c("GFP (ch2)", "BFP nuclear (ch1)", "mCherry (ch3)"),
       col = c("green3", "dodgerblue", "tomato"), lwd = 2)

mtext("Population-level channel dynamics — A2+A3 combined  (no per-cell normalisation)",
      outer = TRUE, cex = 1.1, font = 2)
dev.off()
cat("Saved figures/combined/population_raw_channels.png\n")

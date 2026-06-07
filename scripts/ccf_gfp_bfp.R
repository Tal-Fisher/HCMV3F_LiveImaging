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

# ── CCF per cell ───────────────────────────────────────────────────────────────
min_frames <- 15   # minimum track length for reliable CCF
min_sd     <- 1e-3 # minimum SD to avoid near-flat signals
lag_max    <- 20

ccf_list <- lapply(split(spots_m, spots_m$Track.ID), function(df) {
  df  <- df[order(df$Frame), ]
  gfp <- df$ch2_corrected
  bfp <- df$Mean.ch1

  keep <- is.finite(gfp) & is.finite(bfp)
  if (sum(keep) < min_frames) return(NULL)

  gfp <- gfp[keep]
  bfp <- bfp[keep]

  # skip cells where either channel has no variation
  if (sd(gfp) < min_sd || sd(bfp) < min_sd) return(NULL)

  # detrend via first differences to remove common monotonic trend
  gfp <- diff(gfp)
  bfp <- diff(bfp)
  if (length(gfp) < 2) return(NULL)

  cc <- ccf(gfp, bfp, lag.max = lag_max, plot = FALSE)
  data.frame(
    lag = as.integer(cc$lag),
    ccf = as.numeric(cc$acf)
  )
})
ccf_list <- Filter(Negate(is.null), ccf_list)
cat(sprintf("Cells contributing to CCF: %d / %d\n",
            length(ccf_list), length(unique(spots_m$Track.ID))))

ccf_df <- do.call(rbind, ccf_list)

# ── average CCF across cells per lag ──────────────────────────────────────────
sem_fn <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

avg_ccf <- do.call(rbind, lapply(split(ccf_df, ccf_df$lag), function(d) {
  data.frame(
    lag      = d$lag[1],
    mean_ccf = mean(d$ccf, na.rm = TRUE),
    sem_ccf  = sem_fn(d$ccf),
    n        = sum(!is.na(d$ccf))
  )
}))
avg_ccf <- avg_ccf[order(avg_ccf$lag), ]

cat(sprintf("Lag 0 mean CCF: %.3f\n", avg_ccf$mean_ccf[avg_ccf$lag == 0]))

# ── plot ───────────────────────────────────────────────────────────────────────
png(file.path(fig_comb, "ccf_gfp_bfp.png"),
    width = 900, height = 600, type = "cairo")
par(mar = c(5, 5, 4, 2))

ylim <- range(c(avg_ccf$mean_ccf + avg_ccf$sem_ccf,
                avg_ccf$mean_ccf - avg_ccf$sem_ccf), na.rm = TRUE)
ylim <- c(min(ylim, -0.1), max(ylim, 0.1))

plot(avg_ccf$lag, avg_ccf$mean_ccf,
     type = "n", ylim = ylim,
     xlab = "Lag (frames)",
     ylab = "Mean cross-correlation",
     main = sprintf(
       "Average CCF: GFP vs BFP  —  first-differenced (detrended)\n(n = %d cells; raw corrected GFP vs nuclear BFP)",
       length(ccf_list)))

# ribbon
polygon(c(avg_ccf$lag, rev(avg_ccf$lag)),
        c(avg_ccf$mean_ccf + avg_ccf$sem_ccf,
          rev(avg_ccf$mean_ccf - avg_ccf$sem_ccf)),
        col = rgb(0.2, 0.5, 0.8, 0.25), border = NA)

abline(h = 0,  col = "grey60", lty = 2)
abline(v = 0,  col = "grey60", lty = 2)
lines(avg_ccf$lag, avg_ccf$mean_ccf, col = "steelblue", lwd = 2)
points(0, avg_ccf$mean_ccf[avg_ccf$lag == 0],
       pch = 19, col = "steelblue", cex = 1.5)

text(0, avg_ccf$mean_ccf[avg_ccf$lag == 0],
     sprintf("  r = %.3f", avg_ccf$mean_ccf[avg_ccf$lag == 0]),
     adj = c(0, -0.8), cex = 0.9)

legend("topright", bty = "n",
       legend = c("Mean CCF ± SEM",
                  sprintf("n = %d cells", length(ccf_list))),
       col = c("steelblue", NA), lwd = c(2, NA))

dev.off()
cat("Saved ccf_gfp_bfp.png\n")

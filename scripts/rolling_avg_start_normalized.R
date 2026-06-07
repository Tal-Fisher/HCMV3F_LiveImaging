options(bitmapType = "cairo")

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache",   "combined")
fig_comb   <- file.path(base_dir, "figures", "combined")

spots <- readRDS(file.path(cache_comb, "spots_clean.rds"))
nuc   <- readRDS(file.path(cache_comb, "nuc_assigned.rds"))

spots_m <- merge(
  spots[, c("Track.ID", "Frame", "T..sec.", "ch2_corrected", "Mean.ch3")],
  nuc[,   c("Track.ID", "Frame", "Mean.ch1")],
  by = c("Track.ID", "Frame"), all.x = TRUE
)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

# assign norm_frame = 0, 1, 2, ... from track start for each cell
spots_m$norm_frame <- ave(spots_m$Frame, spots_m$Track.ID,
                          FUN = function(f) f - min(f))

# population mean ± SEM per norm_frame (raw intensities, no per-cell normalisation)
sem_fn <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

pop <- do.call(rbind, lapply(split(spots_m, spots_m$norm_frame), function(d) {
  data.frame(
    norm_frame = d$norm_frame[1],
    mean_gfp   = mean(d$ch2_corrected, na.rm = TRUE),
    sem_gfp    = sem_fn(d$ch2_corrected),
    mean_bfp   = mean(d$Mean.ch1,      na.rm = TRUE),
    sem_bfp    = sem_fn(d$Mean.ch1),
    mean_red   = mean(d$Mean.ch3,      na.rm = TRUE),
    sem_red    = sem_fn(d$Mean.ch3),
    n          = sum(!is.na(d$ch2_corrected))
  )
}))
pop <- pop[order(pop$norm_frame), ]
pop <- pop[pop$n >= 10, ]   # drop frames with very few cells

cat(sprintf("Cells: %d  |  max norm_frame: %d\n",
            length(unique(spots_m$Track.ID)), max(pop$norm_frame)))

draw_ribbon <- function(x, mid, err, col_rgb) {
  ok <- is.finite(mid) & is.finite(err)
  if (sum(ok) < 2) return(invisible(NULL))
  polygon(c(x[ok], rev(x[ok])),
          c(mid[ok] + err[ok], rev(mid[ok] - err[ok])),
          col = col_rgb, border = NA)
}

png(file.path(fig_comb, "rolling_avg_start_normalized.png"),
    width = 1000, height = 600, type = "cairo")

ylim_all <- range(c(
  pop$mean_gfp - pop$sem_gfp, pop$mean_gfp + pop$sem_gfp,
  pop$mean_bfp - pop$sem_bfp, pop$mean_bfp + pop$sem_bfp,
  pop$mean_red - pop$sem_red, pop$mean_red + pop$sem_red
), na.rm = TRUE)

par(mar = c(4, 5, 4, 2))
plot(pop$norm_frame, pop$mean_gfp,
     type = "n", ylim = ylim_all,
     xlab = "Frames from track start (frame 0 = first detection)",
     ylab = "Mean intensity (raw, averaged across cells)",
     main = sprintf(
       "Population mean channel dynamics — time-aligned to track start\nmean ± SEM;  n up to %d cells", max(pop$n)))

draw_ribbon(pop$norm_frame, pop$mean_gfp, pop$sem_gfp, rgb(0, .7,  0, .25))
draw_ribbon(pop$norm_frame, pop$mean_bfp, pop$sem_bfp, rgb(0, .4,  1, .25))
draw_ribbon(pop$norm_frame, pop$mean_red, pop$sem_red, rgb(1, .2, .2, .25))

lines(pop$norm_frame, pop$mean_gfp, col = "green3",     lwd = 2)
lines(pop$norm_frame, pop$mean_bfp, col = "dodgerblue", lwd = 2)
lines(pop$norm_frame, pop$mean_red, col = "tomato",     lwd = 2)

legend("topleft", bty = "n",
       legend = c("GFP corrected (ch2)", "BFP nuclear (ch1)", "mCherry (ch3)"),
       col = c("green3", "dodgerblue", "tomato"), lwd = 2)

dev.off()
cat("Saved rolling_avg_start_normalized.png\n")

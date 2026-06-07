options(bitmapType = "cairo")

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache",   "combined")
fig_comb   <- file.path(base_dir, "figures", "combined")

spots <- readRDS(file.path(cache_comb, "spots_clean.rds"))
nuc   <- readRDS(file.path(cache_comb, "nuc_assigned.rds"))

# merge to get all channels per frame per cell
spots_m <- merge(
  spots[, c("Track.ID", "Frame", "T..sec.", "ch2_corrected", "Mean.ch2", "Mean.ch3")],
  nuc[,   c("Track.ID", "Frame", "Mean.ch1")],
  by = c("Track.ID", "Frame"), all.x = TRUE
)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

sc01 <- function(x) {
  r <- range(x, na.rm = TRUE)
  if (diff(r) > 1e-9) (x - r[1]) / diff(r) else rep(NA_real_, length(x))
}

# per-cell normalized trajectories
norm_list <- lapply(split(spots_m, spots_m$Track.ID), function(df) {
  df <- df[order(df$Frame), ]
  data.frame(
    norm_frame    = seq_len(nrow(df)) - 1L,
    gfp_corr_norm = sc01(df$ch2_corrected),   # alpha = 0.05
    gfp_raw_norm  = sc01(df$Mean.ch2),          # alpha = 0  (raw)
    bfp_norm      = sc01(df$Mean.ch1),
    red_norm      = sc01(df$Mean.ch3)
  )
})
norm_df <- do.call(rbind, norm_list)

# population mean ± SEM per normalised frame
sem_fn <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

pop <- do.call(rbind, lapply(split(norm_df, norm_df$norm_frame), function(d) {
  data.frame(
    norm_frame    = d$norm_frame[1],
    mean_gfp_corr = mean(d$gfp_corr_norm, na.rm = TRUE),
    sem_gfp_corr  = sem_fn(d$gfp_corr_norm),
    mean_gfp_raw  = mean(d$gfp_raw_norm,  na.rm = TRUE),
    sem_gfp_raw   = sem_fn(d$gfp_raw_norm),
    mean_bfp      = mean(d$bfp_norm, na.rm = TRUE),
    sem_bfp       = sem_fn(d$bfp_norm),
    mean_red      = mean(d$red_norm, na.rm = TRUE),
    sem_red       = sem_fn(d$red_norm),
    n             = sum(!is.na(d$gfp_corr_norm))
  )
}))
pop <- pop[order(pop$norm_frame), ]

# ribbon helper — only draws where mean and SEM are finite
draw_ribbon <- function(x, mid, err, col_rgb) {
  ok <- is.finite(mid) & is.finite(err)
  if (sum(ok) < 2) return(invisible(NULL))
  polygon(c(x[ok], rev(x[ok])),
          c(mid[ok] + err[ok], rev(mid[ok] - err[ok])),
          col = col_rgb, border = NA)
}

# shared y range across both conditions
ylim_all <- range(c(
  pop$mean_gfp_corr - pop$sem_gfp_corr, pop$mean_gfp_corr + pop$sem_gfp_corr,
  pop$mean_gfp_raw  - pop$sem_gfp_raw,  pop$mean_gfp_raw  + pop$sem_gfp_raw,
  pop$mean_bfp - pop$sem_bfp, pop$mean_bfp + pop$sem_bfp,
  pop$mean_red - pop$sem_red, pop$mean_red + pop$sem_red
), na.rm = TRUE)

plot_channels <- function(mean_gfp, sem_gfp, title) {
  plot(pop$norm_frame, mean_gfp,
       type = "n", ylim = ylim_all,
       xlab = "Normalized frame (per cell)", ylab = "Normalized intensity (0-1, per cell)",
       main = title)
  # ribbons drawn first, then lines on top
  draw_ribbon(pop$norm_frame, mean_gfp,      sem_gfp,      rgb(0,  .7,  0, .25))
  draw_ribbon(pop$norm_frame, pop$mean_bfp,  pop$sem_bfp,  rgb(0,  .4,  1, .25))
  draw_ribbon(pop$norm_frame, pop$mean_red,  pop$sem_red,  rgb(1,  .2, .2, .25))
  lines(pop$norm_frame, mean_gfp,     col = "green3",     lwd = 2)
  lines(pop$norm_frame, pop$mean_bfp, col = "dodgerblue", lwd = 2)
  lines(pop$norm_frame, pop$mean_red, col = "tomato",     lwd = 2)
  legend("topleft", bty = "n",
         legend = c("GFP", "BFP (nuclear)", "mCherry"),
         col = c("green3", "dodgerblue", "tomato"), lwd = 2)
}

png(file.path(fig_comb, "norm_channels_comparison.png"),
    width = 1400, height = 600, type = "cairo")
par(mfrow = c(1, 2), mar = c(4, 4, 3, 2), oma = c(0, 0, 3, 0))

plot_channels(pop$mean_gfp_corr, pop$sem_gfp_corr,
              sprintf("With bleedthrough correction  (α = 0.05)\nn = %d cells", max(pop$n)))
plot_channels(pop$mean_gfp_raw,  pop$sem_gfp_raw,
              sprintf("Raw  (no correction,  α = 0)\nn = %d cells", max(pop$n)))

mtext("Population mean normalised channels — A2+A3 combined  (each cell 0–1 per channel, mean ± SEM)",
      outer = TRUE, cex = 1.0, font = 2)
dev.off()
cat("Saved figures/combined/norm_channels_comparison.png\n")

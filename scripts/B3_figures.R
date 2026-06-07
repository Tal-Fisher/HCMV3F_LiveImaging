options(bitmapType = "cairo")

fps      <- 15     # minutes per frame
min_cells <- 10    # drop normalized-time bins with fewer cells

dataset   <- "B3"
base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache",   dataset)
fig_dir   <- file.path(base_dir, "figures", dataset)
dir.create(fig_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

for (f in c("spots_clean.rds", "nuc_assigned.rds", "onset_df.rds"))
  if (!file.exists(file.path(cache_dir, f)))
    stop(sprintf("Missing cache/B3/%s — run scripts 01-03 first", f))

spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
onset_df     <- readRDS(file.path(cache_dir, "onset_df.rds"))

cat(sprintf("B3 figures: %d cells, %d with nucleus, %d with red onset\n",
            length(unique(spots$Track.ID)),
            length(unique(nuc_assigned$Track.ID)),
            sum(!is.na(onset_df$red_onset_min))))

# ── Figure 1: Onset overlay — overlaid KDE density curves ────────────────────
green_v <- onset_df$green_onset_min[!is.na(onset_df$green_onset_min)]
blue_v  <- onset_df$blue_onset_min[!is.na(onset_df$blue_onset_min)]
red_v   <- onset_df$red_onset_min[!is.na(onset_df$red_onset_min)]

png(file.path(fig_dir, "onset_overlay.png"), width = 800, height = 500, type = "cairo")
par(mar = c(4, 5, 4, 2))

d_green <- density(green_v, from = 0, n = 512)
d_blue  <- density(blue_v,  from = 0, n = 512)
d_red   <- density(red_v,   from = 0, n = 512)

xlim_d <- range(c(d_green$x, d_blue$x, d_red$x))
ylim_d <- c(0, max(c(d_green$y, d_blue$y, d_red$y)) * 1.15)

plot(d_green, col = "green3", lwd = 2.5, xlim = xlim_d, ylim = ylim_d,
     main = sprintf("B3 — onset time distributions\n(GFP n=%d, BFP n=%d, mCherry n=%d)",
                    length(green_v), length(blue_v), length(red_v)),
     xlab = "Time from track start (min)", ylab = "Density")
polygon(c(d_green$x, rev(d_green$x)), c(d_green$y, rep(0, length(d_green$y))),
        col = rgb(0, 0.7, 0, 0.15), border = NA)
lines(d_blue, col = "dodgerblue", lwd = 2.5)
polygon(c(d_blue$x, rev(d_blue$x)), c(d_blue$y, rep(0, length(d_blue$y))),
        col = rgb(0, 0.4, 1, 0.15), border = NA)
lines(d_red, col = "tomato", lwd = 2.5)
polygon(c(d_red$x, rev(d_red$x)), c(d_red$y, rep(0, length(d_red$y))),
        col = rgb(1, 0.2, 0.2, 0.15), border = NA)
legend("topright", bty = "n",
       legend = c(sprintf("GFP  (n=%d)",     length(green_v)),
                  sprintf("BFP  (n=%d)",     length(blue_v)),
                  sprintf("mCherry (n=%d)",  length(red_v))),
       col = c("green3", "dodgerblue", "tomato"), lwd = 2.5)
dev.off()
cat("Saved onset_overlay.png\n")

# ── Figure 2: Green-to-red delay distribution ─────────────────────────────────
valid_delay <- onset_df$delay_green_to_red[is.finite(onset_df$delay_green_to_red)]
png(file.path(fig_dir, "green_to_red_delay.png"), width = 600, height = 450, type = "cairo")
hist(valid_delay, breaks = 40, col = "salmon", border = "white",
     main = sprintf("B3 — green to red delay\n(n=%d cells, median=%.0f min)",
                    length(valid_delay), median(valid_delay)),
     xlab = "Delay from GFP onset to mCherry onset (min)", ylab = "Cells")
abline(v = median(valid_delay), col = "red", lwd = 2, lty = 2)
dev.off()
cat("Saved green_to_red_delay.png\n")

# ── build merged per-frame table: spots + nucleus + speed ─────────────────────

# step-to-step Euclidean displacement per cell
spots_xy <- spots[order(spots$Track.ID, spots$Frame), c("Track.ID", "Frame", "X", "Y")]
speed_list <- lapply(split(spots_xy, spots_xy$Track.ID), function(d) {
  d   <- d[order(d$Frame), ]
  spd <- c(NA_real_, sqrt(diff(d$X)^2 + diff(d$Y)^2))
  data.frame(Track.ID = d$Track.ID, Frame = d$Frame, speed_pf = spd)
})
speed_df <- do.call(rbind, speed_list)
rownames(speed_df) <- NULL

# merge spots + nucleus (Area and Circ.) + speed; Area gets suffixes _cell / _nuc
spots_m <- merge(
  spots[,        c("Track.ID", "Frame", "T..sec.", "ch2_corrected", "Mean.ch3", "X", "Y", "Area")],
  nuc_assigned[, c("Track.ID", "Frame", "Mean.ch1", "Area", "Circ.")],
  by = c("Track.ID", "Frame"), all.x = TRUE,
  suffixes = c("_cell", "_nuc")
)
spots_m <- merge(spots_m, speed_df[, c("Track.ID", "Frame", "speed_pf")],
                 by = c("Track.ID", "Frame"), all.x = TRUE)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

# normalized frame: 0 = first detected frame of each cell
spots_m$norm_frame <- ave(spots_m$Frame, spots_m$Track.ID,
                          FUN = function(f) f - min(f))

# ── population mean ± SEM per normalized frame ────────────────────────────────
sem_fn <- function(x) {
  n <- sum(!is.na(x))
  if (n < 2) NA_real_ else sd(x, na.rm = TRUE) / sqrt(n)
}

pop <- do.call(rbind, lapply(split(spots_m, spots_m$norm_frame), function(d) {
  data.frame(
    norm_frame     = d$norm_frame[1],
    norm_time_min  = d$norm_frame[1] * fps,
    mean_gfp       = mean(d$ch2_corrected, na.rm = TRUE),
    sem_gfp        = sem_fn(d$ch2_corrected),
    mean_bfp       = mean(d$Mean.ch1,      na.rm = TRUE),
    sem_bfp        = sem_fn(d$Mean.ch1),
    mean_red       = mean(d$Mean.ch3,      na.rm = TRUE),
    sem_red        = sem_fn(d$Mean.ch3),
    mean_area_cell = mean(d$Area_cell,     na.rm = TRUE),
    sem_area_cell  = sem_fn(d$Area_cell),
    mean_area_nuc  = mean(d$Area_nuc,      na.rm = TRUE),
    sem_area_nuc   = sem_fn(d$Area_nuc),
    mean_circ_nuc  = mean(d$Circ.,         na.rm = TRUE),
    sem_circ_nuc   = sem_fn(d$Circ.),
    mean_speed     = mean(d$speed_pf,      na.rm = TRUE),
    sem_speed      = sem_fn(d$speed_pf),
    n              = sum(!is.na(d$ch2_corrected))
  )
}))
pop <- pop[order(pop$norm_frame), ]
pop <- pop[pop$n >= min_cells, ]
cat(sprintf("Population stats: %d normalized frames (min %d cells/frame)\n",
            nrow(pop), min_cells))

# helper: ribbon around a line
draw_ribbon <- function(x, mid, err, fill_col) {
  ok <- is.finite(mid) & is.finite(err)
  if (sum(ok) < 2) return(invisible(NULL))
  polygon(c(x[ok], rev(x[ok])),
          c(mid[ok] + err[ok], rev(mid[ok] - err[ok])),
          col = fill_col, border = NA)
}

# ── Figure 3: Channel rolling averages (normalized time) ──────────────────────
ylim_ch <- range(c(
  pop$mean_gfp - pop$sem_gfp, pop$mean_gfp + pop$sem_gfp,
  pop$mean_bfp - pop$sem_bfp, pop$mean_bfp + pop$sem_bfp,
  pop$mean_red - pop$sem_red, pop$mean_red + pop$sem_red
), na.rm = TRUE)

png(file.path(fig_dir, "channel_rolling_avg.png"), width = 900, height = 550, type = "cairo")
par(mar = c(4, 5, 4, 2))
plot(pop$norm_time_min, pop$mean_gfp, type = "n", ylim = ylim_ch,
     xlab = "Time from first detection (min)",
     ylab = "Mean intensity (averaged across cells)",
     main = sprintf("B3 — channel dynamics, aligned to each cell's first frame\nmean ± SEM; n up to %d cells",
                    max(pop$n)))
draw_ribbon(pop$norm_time_min, pop$mean_gfp, pop$sem_gfp, rgb(0,  .7,  0, .25))
draw_ribbon(pop$norm_time_min, pop$mean_bfp, pop$sem_bfp, rgb(0,  .4,  1, .25))
draw_ribbon(pop$norm_time_min, pop$mean_red, pop$sem_red, rgb(1,  .2, .2, .25))
lines(pop$norm_time_min, pop$mean_gfp, col = "green3",     lwd = 2)
lines(pop$norm_time_min, pop$mean_bfp, col = "dodgerblue", lwd = 2)
lines(pop$norm_time_min, pop$mean_red, col = "tomato",     lwd = 2)
legend("topleft", bty = "n",
       legend = c("GFP corrected (ch2)", "BFP nuclear (ch1)", "mCherry (ch3)"),
       col    = c("green3", "dodgerblue", "tomato"), lwd = 2)
dev.off()
cat("Saved channel_rolling_avg.png\n")

# ── Figure 4: Morphology rolling averages (2×2, normalized time) ──────────────
make_panel <- function(t, mid, err, ylab, main, line_col) {
  ylim <- range(c(mid - err, mid + err), na.rm = TRUE)
  plot(t, mid, type = "n", ylim = ylim,
       xlab = "Time from first detection (min)", ylab = ylab, main = main)
  draw_ribbon(t, mid, err, adjustcolor(line_col, alpha.f = 0.2))
  lines(t, mid, col = line_col, lwd = 2)
}

png(file.path(fig_dir, "morphology_rolling_avg.png"),
    width = 1000, height = 900, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 5, 3, 1), oma = c(0, 0, 3, 0))

make_panel(pop$norm_time_min, pop$mean_area_cell, pop$sem_area_cell,
           "Area (AU²)", "Cell area",       "gray30")
make_panel(pop$norm_time_min, pop$mean_area_nuc,  pop$sem_area_nuc,
           "Area (AU²)", "Nucleus area",     "darkblue")
make_panel(pop$norm_time_min, pop$mean_circ_nuc,  pop$sem_circ_nuc,
           "Circularity (0–1)", "Nucleus circularity", "steelblue")
make_panel(pop$norm_time_min, pop$mean_speed,     pop$sem_speed,
           "Displacement (px/frame)", "Cell speed",  "firebrick")

mtext("B3 — morphology dynamics, aligned to each cell's first frame\nmean ± SEM",
      outer = TRUE, cex = 1, font = 2, line = 0.5)
dev.off()
cat("Saved morphology_rolling_avg.png\n")

# ── Figure 5: Blue-to-red delay distribution ──────────────────────────────────
both_mask  <- !is.na(onset_df$blue_onset_min) & !is.na(onset_df$red_onset_min)
delay_br   <- onset_df$red_onset_min[both_mask] - onset_df$blue_onset_min[both_mask]
pos_mask   <- delay_br >= 0   # exclude negative delays (BFP after mCherry — noise)
delay_br_pos <- delay_br[pos_mask]
cat(sprintf("Blue-to-red delay: %d cells with both events (%d negative excluded)\n",
            sum(both_mask), sum(!pos_mask)))

png(file.path(fig_dir, "blue_to_red_delay.png"), width = 700, height = 500, type = "cairo")
par(mar = c(5, 4, 4, 2))
hist(delay_br_pos / 60, breaks = 30, col = adjustcolor("mediumpurple", 0.7), border = "white",
     main = sprintf("B3 — BFP → mCherry delay\n(n=%d cells, median=%.1f h)",
                    length(delay_br_pos), median(delay_br_pos) / 60),
     xlab = "Delay from BFP onset to mCherry onset (h)", ylab = "Cells")
abline(v = median(delay_br_pos) / 60, col = "purple4", lwd = 2, lty = 2)
legend("topright", bty = "n",
       legend = sprintf("Negative delays excluded: %d", sum(!pos_mask)),
       col = "grey50", lwd = 1)
dev.off()
cat("Saved blue_to_red_delay.png\n")

# ── save population stats table to cache ──────────────────────────────────────
saveRDS(pop,     file.path(cache_dir, "pop_stats.rds"))
saveRDS(spots_m, file.path(cache_dir, "spots_processed.rds"))
cat(sprintf("Saved cache/B3/pop_stats.rds + spots_processed.rds\n"))
cat("=== B3 figures done ===\n")

options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs", .libPaths()))

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"

# sequential yellow → orange → dark red palette
delay_pal <- colorRampPalette(c("#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"))

# draw a horizontal heatmap bar in the bottom-right of the plot area
draw_heatbar <- function(zlim, pal, label, n = 200) {
  usr     <- par("usr")
  xpd_old <- par(xpd = NA)
  on.exit(par(xpd = xpd_old))

  x_range <- usr[2] - usr[1]
  y_range <- usr[4] - usr[3]

  bar_w  <- x_range * 0.32          # bar spans 32% of x range
  bar_h  <- y_range * 0.035         # bar height
  gap    <- y_range * 0.03          # gap from bottom + some text room
  text_h <- y_range * 0.055         # room below bar for tick labels

  bar_x0 <- usr[2] - bar_w
  bar_x1 <- usr[2]
  bar_y0 <- usr[3] + gap + text_h
  bar_y1 <- bar_y0 + bar_h

  xs   <- seq(bar_x0, bar_x1, length.out = n + 1)
  cols <- pal(n)
  for (i in seq_len(n))
    rect(xs[i], bar_y0, xs[i + 1], bar_y1, col = cols[i], border = NA)
  rect(bar_x0, bar_y0, bar_x1, bar_y1, col = NA, border = "grey30", lwd = 0.6)

  # grey "no mCherry" swatch to the left of the bar
  swatch_w <- bar_w * 0.18
  rect(bar_x0 - swatch_w - x_range * 0.01, bar_y0,
       bar_x0 - x_range * 0.01,            bar_y1,
       col = adjustcolor("grey60", 0.5), border = "grey30", lwd = 0.6)
  text(bar_x0 - swatch_w / 2 - x_range * 0.01, bar_y0 - y_range * 0.025,
       "No\nmCherry", cex = 0.6, adj = 0.5)

  # tick labels at min / mid / max
  ticks   <- c(zlim[1], mean(zlim), zlim[2])
  tick_x  <- bar_x0 + (ticks - zlim[1]) / diff(zlim) * bar_w
  text(tick_x, bar_y0 - y_range * 0.022,
       labels = sprintf("%.0f h", ticks), cex = 0.65, adj = 0.5)

  # bar label above
  text(mean(c(bar_x0, bar_x1)), bar_y1 + y_range * 0.018,
       label, cex = 0.72, adj = 0.5)
}

make_scatter <- function(dataset) {
  cache_dir <- file.path(base_dir, "cache", dataset)
  fig_dir   <- file.path(base_dir, "figures", dataset)
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
  nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
  onset_df     <- readRDS(file.path(cache_dir, "onset_df.rds"))

  # absolute track-start time per cell (= first GFP frame)
  track_start_min <- tapply(spots[["T..sec."]], spots[["Track.ID"]], min) / 60
  track_start_df  <- data.frame(
    Track.ID    = names(track_start_min),
    gfp_first_h = as.numeric(track_start_min) / 60,
    stringsAsFactors = FALSE
  )

  # nucleus-assigned cells with BFP onset; bring in red_onset_min for coloring
  nuc_ids <- unique(nuc_assigned$Track.ID)
  plot_df <- merge(
    onset_df[onset_df$Track.ID %in% nuc_ids & !is.na(onset_df$blue_onset_min),
             c("Track.ID", "blue_onset_min", "red_onset_min")],
    track_start_df, by = "Track.ID")
  plot_df$bfp_first_h    <- plot_df$gfp_first_h + plot_df$blue_onset_min / 60
  plot_df$delay_bfp_red_h <- (plot_df$red_onset_min - plot_df$blue_onset_min) / 60

  r   <- cor(plot_df$gfp_first_h, plot_df$bfp_first_h)
  rho <- cor(plot_df$gfp_first_h, plot_df$bfp_first_h, method = "spearman")
  cat(sprintf("%s: n=%d cells  r=%.3f  rho=%.3f\n", dataset, nrow(plot_df), r, rho))

  # color by delay; cells without mCherry → grey
  has_delay  <- is.finite(plot_df$delay_bfp_red_h) & plot_df$delay_bfp_red_h >= 0
  delay_vals <- plot_df$delay_bfp_red_h[has_delay]
  zlim       <- c(0, quantile(delay_vals, 0.97, na.rm = TRUE))  # cap at 97th pct
  norm_delay <- pmin(pmax((plot_df$delay_bfp_red_h - zlim[1]) / diff(zlim), 0), 1)
  pal_cols   <- delay_pal(256)
  pt_col     <- ifelse(has_delay,
                       adjustcolor(pal_cols[pmax(1L, as.integer(norm_delay * 255) + 1L)], 0.75),
                       adjustcolor("grey60", 0.5))

  outfile <- file.path(fig_dir, "gfp_bfp_first_frame_scatter_delay.png")
  png(outfile, width = 650, height = 580, res = 110, type = "cairo")
  par(mar = c(5, 5, 4, 2))

  plot(plot_df$gfp_first_h, plot_df$bfp_first_h,
       col = pt_col, pch = 16, cex = 0.9,
       xlab = "First GFP frame — absolute time (h)",
       ylab = "First BFP frame — absolute time (h)",
       main = sprintf("%s — GFP vs BFP first appearance\n(n=%d cells, r=%.2f, ρ=%.2f)",
                      dataset, nrow(plot_df), r, rho))
  abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
  abline(lm(bfp_first_h ~ gfp_first_h, data = plot_df), col = "grey20", lwd = 1.8)

  draw_heatbar(zlim, delay_pal, "BFP → mCherry delay (h)")
  dev.off()
  cat(sprintf("Saved figures/%s/gfp_bfp_first_frame_scatter_delay.png\n", dataset))
}

for (ds in c("A2", "A3", "B3")) make_scatter(ds)

## export_red_slope.R
## Reads spots_clean.rds + onset_df.rds, computes the linear slope of
## mCherry (Mean.ch3) over the 10 frames starting at red onset for each
## productive cell, and exports cache/python_export/red_slope.csv.

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"

spots  <- readRDS(file.path(base_dir, "cache/combined/spots_clean.rds"))
onset  <- readRDS(file.path(base_dir, "cache/combined/onset_df.rds"))

cat("spots dims:", nrow(spots), "x", ncol(spots), "\n")
cat("onset dims:", nrow(onset), "x", ncol(onset), "\n")
cat("spot cols:", paste(names(spots), collapse=", "), "\n")
cat("onset cols:", paste(names(onset), collapse=", "), "\n")

# Identify the frame column and red onset column
# Common naming: Frame (0-based or 1-based), red_onset_frame or red_frame
track_col  <- "Track.ID"
red_ch_col <- "Mean.ch3"
time_col   <- "T..sec."   # time in seconds

# onset_df uses red_onset_min (minutes)
red_onset_col <- "red_onset_min"

# Keep only productive cells (finite red onset)
prod <- onset[is.finite(onset[[red_onset_col]]), c(track_col, red_onset_col)]
cat("Productive cells:", nrow(prod), "\n")

# Estimate frame interval
t_sorted <- sort(unique(spots[[time_col]]))
frame_interval_sec <- median(diff(t_sorted))
cat("Estimated frame interval:", round(frame_interval_sec/60, 2), "min\n")

N_FRAMES  <- 10
win_sec   <- N_FRAMES * frame_interval_sec

# Vectorised: merge onset onto spots, filter window, then fit per group
prod$red_onset_sec <- prod[[red_onset_col]] * 60
spots_prod <- merge(spots[, c(track_col, time_col, red_ch_col)],
                    prod[, c(track_col, "red_onset_sec")],
                    by=track_col)

win_spots <- spots_prod[spots_prod[[time_col]] >= spots_prod$red_onset_sec &
                        spots_prod[[time_col]] <  spots_prod$red_onset_sec + win_sec, ]

cat("Window spots:", nrow(win_spots), "\n")

# Fit slope per cell using by()
slope_list <- by(win_spots, win_spots[[track_col]], function(g) {
  g <- g[order(g[[time_col]]), ]
  if (nrow(g) < 2) return(data.frame(Track.ID=g[[track_col]][1], red_slope=NA_real_, n_frames_used=nrow(g)))
  fit <- lm(g[[red_ch_col]] ~ I(g[[time_col]] / 60))
  data.frame(Track.ID=g[[track_col]][1], red_slope=coef(fit)[2], n_frames_used=nrow(g))
})
red_slope_df <- do.call(rbind, slope_list)
cat("Computed slopes for", sum(!is.na(red_slope_df$red_slope)), "cells\n")
cat("Slope summary:\n"); print(summary(red_slope_df$red_slope))

out_path <- file.path(base_dir, "cache/python_export/red_slope.csv")
write.csv(red_slope_df, out_path, row.names=FALSE)
cat("Saved", out_path, "\n")

options(bitmapType = "cairo")

dataset  <- "B3"   # change to "A3"

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", dataset)
fig_dir   <- file.path(base_dir, "figures", dataset)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(file.path(cache_dir, "spots_clean.rds"))) stop("Run 01_preprocess.R first")
if (!file.exists(file.path(cache_dir, "nuc_assigned.rds"))) stop("Run 02_nuclei.R first")
spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
cat(sprintf("=== %s: onset detection (%d cells) ===\n",
            dataset, length(unique(spots$Track.ID))))

# ── helper: detect onset (5 consecutive smoothed frames > thresh, delta > 0.15) ─
# Loops through ALL valid windows so late-rising cells are not missed.
detect_onset_time <- function(x, t, thresh, n_consec = 5, delta_min = 0.15) {
  n      <- length(x)
  smooth <- vapply(seq_len(n), function(i)
    mean(x[max(1, i-1):min(n, i+1)], na.rm = TRUE), numeric(1))
  pos <- !is.na(smooth) & smooth > thresh
  if (n < n_consec) return(NA_real_)
  for (i in n_consec:n) {
    if (all(pos[(i - n_consec + 1):i])) {
      delta <- smooth[i] - smooth[i - n_consec + 1]
      if (is.finite(delta) && delta > delta_min)
        return(t[i - n_consec + 1])
    }
  }
  NA_real_
}

# ── detect all three onsets per track ─────────────────────────────────────────
# Green (ch2_corrected): threshold 0.2, 3 consecutive frames, no delta constraint
# Blue  (nuc Mean.ch1):  threshold 0.5, 3 consecutive frames, no delta constraint
# Red   (Mean.ch3):      threshold 2.25, 5 consecutive frames, delta > 0.15
#   B3 override → 3.14: computed as mean + 2.4×SD of the first 10 frames per
#   cell (pre-infection baseline), matching the same k calibrated from A2's
#   validated threshold.  See cache/B3/threshold_rationale.txt.
green_thresh <- 0.2   # just above noise floor; NOT the exclusion cutoff (1.5)
blue_thresh  <- 0.5   # just above segmentation noise floor (~0.78 at 5th pct)
red_thresh   <- 2.25
if (dataset == "B3") red_thresh <- 3.00

# merge nuclear BFP into spots by Track.ID + Frame for blue detection
spots_m <- merge(
  spots[, c("Track.ID","Frame","T..sec.","ch2_corrected","Mean.ch3")],
  nuc_assigned[, c("Track.ID","Frame","Mean.ch1")],
  by = c("Track.ID","Frame"), all.x = TRUE
)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

onset_list <- lapply(split(spots_m, spots_m$Track.ID), function(df) {
  df <- df[order(df$T..sec.), ]
  t  <- df$T..sec.
  t0 <- t[1]

  green_t <- detect_onset_time(df$ch2_corrected, t, green_thresh, n_consec=3, delta_min=-Inf)
  blue_t  <- detect_onset_time(df$Mean.ch1,      t, blue_thresh, n_consec=3, delta_min=-Inf)
  red_t   <- detect_onset_time(df$Mean.ch3,      t, red_thresh)

  # MODEL delay: track-start → red (all cells with a red event contribute;
  # track start ≈ pre-green since we filtered out early-GFP cells)
  delay_green_to_red <- ifelse(is.finite(red_t), (red_t - t0) / 60, Inf)

  data.frame(
    Track.ID              = df$Track.ID[1],
    # individual onset times from track start (for visualization)
    green_onset_min       = ifelse(is.finite(green_t), (green_t - t0) / 60, NA_real_),
    blue_onset_min        = ifelse(is.finite(blue_t),  (blue_t  - t0) / 60, NA_real_),
    red_onset_min         = ifelse(is.finite(red_t),   (red_t   - t0) / 60, NA_real_),
    # model outcome: track-start to red onset
    delay_green_to_red    = delay_green_to_red,
    # explicit green-to-red delay (only cells with both events; for visualization)
    delay_green_to_red_explicit = ifelse(is.finite(green_t) & is.finite(red_t),
                                         (red_t - green_t) / 60, NA_real_),
    delay_green_to_blue   = ifelse(is.finite(green_t) & is.finite(blue_t),
                                   (blue_t - green_t) / 60, NA_real_)
  )
})

onset_df <- do.call(rbind, onset_list)

cat(sprintf("Green events: %d / %d\n",
            sum(!is.na(onset_df$green_onset_min)), nrow(onset_df)))
cat(sprintf("Blue  events: %d / %d\n",
            sum(!is.na(onset_df$blue_onset_min)),  nrow(onset_df)))
cat(sprintf("Red   events: %d / %d  (median delay from green: %.0f min)\n",
            sum(!is.na(onset_df$red_onset_min)), nrow(onset_df),
            median(onset_df$delay_green_to_red[is.finite(onset_df$delay_green_to_red)])))

# ── plot: time of appearance distributions (BFP + mCherry only) ───────────────
nuc_cell_ids  <- unique(nuc_assigned$Track.ID)
has_red        <- !is.na(onset_df$red_onset_min)
blue_v_nuc    <- onset_df$blue_onset_min[
  !is.na(onset_df$blue_onset_min) & has_red & onset_df$Track.ID %in% nuc_cell_ids]

png(file.path(fig_dir, "onset_distributions.png"), width=700, height=400, type="cairo")
par(mfrow=c(1,2), mar=c(4,4,3,1))

hist(blue_v_nuc,
     breaks=40, col="dodgerblue", border="white",
     main=sprintf("BFP onset\n(n=%d / %d cells with mCherry)",
                  length(blue_v_nuc), sum(has_red & onset_df$Track.ID %in% nuc_cell_ids)),
     xlab="Time from track start (min)", ylab="Cells")

hist(onset_df$red_onset_min[!is.na(onset_df$red_onset_min)],
     breaks=40, col="tomato", border="white",
     main=sprintf("mCherry onset\n(n=%d / %d cells)",
                  sum(!is.na(onset_df$red_onset_min)), nrow(onset_df)),
     xlab="Time from track start (min)", ylab="Cells")

mtext(sprintf("%s — time of fluorophore appearance", dataset),
      outer=TRUE, cex=0.9, font=2, line=-1.5)
dev.off()
cat(sprintf("Saved figures/%s/onset_distributions.png\n", dataset))

# ── plot: green-to-red delay distribution ─────────────────────────────────────
valid_delay <- onset_df$delay_green_to_red[is.finite(onset_df$delay_green_to_red)]
png(file.path(fig_dir, "green_to_red_delay.png"), width=600, height=450, type="cairo")
hist(valid_delay, breaks=40, col="salmon", border="white",
     main=sprintf("%s — green to red delay\n(n=%d cells, median=%.0f min)",
                  dataset, length(valid_delay), median(valid_delay)),
     xlab="Delay from GFP onset to mCherry onset (min)", ylab="Cells")
abline(v=median(valid_delay), col="red", lwd=2, lty=2)
dev.off()
cat(sprintf("Saved figures/%s/green_to_red_delay.png\n", dataset))

# ── flag and remove cells with red but missing green or blue ──────────────────
# Biologically impossible: red (late) requires green (IE) and blue (early) first.
# These cells likely entered the movie mid-infection.
flag_mask     <- !is.na(onset_df$red_onset_min) &
                 (is.na(onset_df$green_onset_min) | is.na(onset_df$blue_onset_min))
onset_flagged <- onset_df[flag_mask, ]
onset_df      <- onset_df[!flag_mask, ]
cat(sprintf("Flagged %d cells (red without green or blue) — saved to onset_flagged.rds\n",
            nrow(onset_flagged)))
saveRDS(onset_flagged, file.path(cache_dir, "onset_flagged.rds"))

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(onset_df, file.path(cache_dir, "onset_df.rds"))
cat(sprintf("Saved cache/%s/onset_df.rds\n", dataset))

options(bitmapType = "cairo")

data_dir <- "/home/labs/ginossar/talfis/LiveImaging/CompleteImage"
out_dir  <- "/home/labs/ginossar/talfis/LiveImaging"

# ── Load & basic filter ────────────────────────────────────────────────────────
spots <- read.csv(file.path(data_dir, "A2_Merged_spots.csv"))
spots <- spots[!is.na(spots$Track.ID), ]
spots <- spots[order(spots$Track.ID, spots$T..sec.), ]

first_gfp <- tapply(spots$Mean.ch2, spots$Track.ID, function(x) x[1])
keep_ids  <- as.integer(names(first_gfp)[first_gfp < 2.5])
spots     <- spots[spots$Track.ID %in% keep_ids, ]

# ── Use first N frames of each track as "pre-green baseline" ──────────────────
# After filtering (first_gfp < 2.5), the first frames are at or before GFP onset.
# These rows represent: true GFP ≈ 0, so ch2 ≈ bleedthrough from ch1 only.
n_baseline <- 5

spots_list <- split(spots, spots$Track.ID)
baseline   <- do.call(rbind, lapply(spots_list, function(df) {
  df <- df[order(df$T..sec.), ]
  head(df, n_baseline)
}))

baseline <- baseline[is.finite(baseline$Mean.ch1) & baseline$Mean.ch1 > 0 &
                     is.finite(baseline$Mean.ch2), ]

cat("Cells contributing to baseline:", length(unique(baseline$Track.ID)), "\n")
cat("Total baseline rows:           ", nrow(baseline), "\n")
cat(sprintf("ch1 range: %.3f – %.3f\n", min(baseline$Mean.ch1), max(baseline$Mean.ch1)))
cat(sprintf("ch2 range: %.3f – %.3f\n", min(baseline$Mean.ch2), max(baseline$Mean.ch2)))

# ── Fitted alpha (diagnostic only — NOT used for correction) ──────────────────
# The data-fitted alpha reflects biological co-expression of GFP + BFP, not
# true optical bleedthrough. It is shown here for reference only.
fit0 <- lm(Mean.ch2 ~ Mean.ch1 - 1, data = baseline)
fit1 <- lm(Mean.ch2 ~ Mean.ch1,     data = baseline)

alpha_fitted <- unname(coef(fit0)[1])
alpha_int    <- unname(coef(fit1)[2])
intercept    <- unname(coef(fit1)[1])
r2_0         <- summary(fit0)$r.squared
r2_1         <- summary(fit1)$r.squared

cat(sprintf("\n[Diagnostic] Fitted alpha (through origin): %.6f  R² = %.4f\n", alpha_fitted, r2_0))
cat(sprintf("[Diagnostic] Fitted alpha (with intercept): %.6f  R² = %.4f  intercept = %.4f\n",
            alpha_int, r2_1, intercept))
cat("NOTE: fitted alpha is biologically contaminated. Using theoretical alpha = 0.05\n")

# Theoretical α: Pacific Blue emission at GFP detection window (500-550 nm)
# is ~3-8% of its peak (455 nm) under simultaneous 405/488 nm acquisition.
alpha <- 0.05
cat(sprintf("Alpha used for correction: %.5f  (theoretical, simultaneous acquisition)\n", alpha))

# Per-cell alpha
per_cell_alpha <- vapply(split(baseline, baseline$Track.ID), function(df) {
  if (nrow(df) < 3 || sd(df$Mean.ch1, na.rm = TRUE) < 1e-6) return(NA_real_)
  unname(coef(lm(Mean.ch2 ~ Mean.ch1 - 1, data = df))[1])
}, numeric(1))
per_cell_alpha <- per_cell_alpha[is.finite(per_cell_alpha)]

cat(sprintf("Per-cell α: n=%d  median=%.5f  SD=%.5f  IQR=[%.5f, %.5f]\n",
            length(per_cell_alpha),
            median(per_cell_alpha),
            sd(per_cell_alpha),
            quantile(per_cell_alpha, 0.25),
            quantile(per_cell_alpha, 0.75)))

# ── Plot 1: Scatter with regression lines ─────────────────────────────────────
png(file.path(out_dir, "plot1_bleedthrough_scatter.png"),
    width = 800, height = 650, type = "cairo")

plot(baseline$Mean.ch1, baseline$Mean.ch2,
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.12),
     xlab = "Cell BFP intensity (ch1)  —  first 5 frames per track",
     ylab = "Cell GFP intensity (ch2)  —  first 5 frames per track",
     main = "BFP → GFP bleedthrough estimation\n(first frames, before true GFP expression)")

x_seq <- seq(0, max(baseline$Mean.ch1, na.rm = TRUE), length.out = 200)
lines(x_seq, alpha        * x_seq,             col = "forestgreen", lwd = 2)
lines(x_seq, alpha_fitted * x_seq,             col = "red",  lwd = 2, lty = 2)
lines(x_seq, alpha_int    * x_seq + intercept, col = "blue", lwd = 2, lty = 3)

legend("topleft", bty = "n",
       legend = c(sprintf("Theoretical α=%.5f  (used)", alpha),
                  sprintf("Fitted (origin)  α=%.5f  R²=%.3f  [diagnostic]", alpha_fitted, r2_0),
                  sprintf("Fitted (intercept) α=%.5f  R²=%.3f  [diagnostic]", alpha_int, r2_1)),
       col = c("forestgreen","red","blue"), lwd = 2, lty = c(1, 2, 3))
dev.off()
cat("\nSaved plot1_bleedthrough_scatter.png\n")

# ── Plot 2: Per-cell alpha distribution ───────────────────────────────────────
png(file.path(out_dir, "plot2_per_cell_alpha.png"),
    width = 800, height = 500, type = "cairo")

hist(per_cell_alpha, breaks = 40, col = "steelblue", border = "white",
     main = sprintf("Per-cell α  (n=%d cells)\nMedian=%.5f  SD=%.5f",
                    length(per_cell_alpha), median(per_cell_alpha), sd(per_cell_alpha)),
     xlab = "α  (bleedthrough coefficient per cell)", ylab = "Cells")
abline(v = alpha,                  col = "forestgreen", lwd = 2)
abline(v = alpha_fitted,           col = "red",         lwd = 2, lty = 2)
abline(v = median(per_cell_alpha), col = "orange",      lwd = 2, lty = 3)
legend("topright", bty = "n",
       legend = c(sprintf("Theoretical α = %.5f  (used)", alpha),
                  sprintf("Fitted global α = %.5f  [diagnostic]", alpha_fitted),
                  sprintf("Median per-cell α = %.5f  [diagnostic]", median(per_cell_alpha))),
       col = c("forestgreen","red","orange"), lwd = 2, lty = c(1, 2, 3))
dev.off()
cat("Saved plot2_per_cell_alpha.png\n")

# ── Apply correction (no floor — let signal go slightly negative) ─────────────
spots$ch2_corrected <- spots$Mean.ch2 - alpha * spots$Mean.ch1

# ── Plot 3: Raw vs corrected GFP – 16 example tracks ─────────────────────────
set.seed(42)
example_ids <- sort(sample(unique(spots$Track.ID), 16))

png(file.path(out_dir, "plot3_corrected_tracks.png"),
    width = 1200, height = 900, type = "cairo")
par(mfrow = c(4, 4), mar = c(2, 2.5, 1.8, 0.5), oma = c(0, 0, 2.5, 0))

for (tid in example_ids) {
  df <- spots[spots$Track.ID == tid, ]
  df <- df[order(df$T..sec.), ]
  t_min <- df$T..sec. / 60
  ylim  <- range(c(df$Mean.ch2, df$ch2_corrected), na.rm = TRUE)

  plot(t_min, df$Mean.ch2,
       type = "l", col = "darkgreen", lwd = 1.5, ylim = ylim,
       xlab = "", ylab = "", main = paste("Track", tid), cex.main = 0.85)
  lines(t_min, df$ch2_corrected, col = "chartreuse3", lwd = 1.5, lty = 2)
}
mtext("GFP: raw (dark green)  vs  BFP-corrected (dashed)",
      outer = TRUE, cex = 0.9, font = 2)
dev.off()
cat("Saved plot3_corrected_tracks.png\n")

# ── Plot 4: BFP and GFP over time (scaled) – same 16 cells ───────────────────
png(file.path(out_dir, "plot4_bfp_vs_gfp_trajectories.png"),
    width = 1200, height = 900, type = "cairo")
par(mfrow = c(4, 4), mar = c(2, 2.5, 1.8, 0.5), oma = c(0, 0, 2.5, 0))

sc <- function(x) {
  r <- range(x, na.rm = TRUE)
  if (diff(r) < 1e-9) return(rep(0.5, length(x)))
  (x - r[1]) / diff(r)
}

for (tid in example_ids) {
  df    <- spots[spots$Track.ID == tid, ]
  df    <- df[order(df$T..sec.), ]
  t_min <- df$T..sec. / 60

  plot(t_min, sc(df$Mean.ch2), type = "l", col = "green3",     lwd = 1.5,
       ylim = c(0,1), xlab = "", ylab = "scaled 0–1",
       main = paste("Track", tid), cex.main = 0.85)
  lines(t_min, sc(df$Mean.ch1), col = "dodgerblue", lwd = 1.5)
}
mtext("GFP (green) vs BFP (blue) — scaled 0–1 per cell",
      outer = TRUE, cex = 0.9, font = 2)
dev.off()
cat("Saved plot4_bfp_vs_gfp_trajectories.png\n")

cat("Saved plot4_bfp_vs_gfp_trajectories.png\n")

# ── Plot 5: Population mean normalised GFP (raw vs corrected) ─────────────────
# Normalise each cell's trajectory to its own [0,1] range, then average per frame.

all_frames <- sort(unique(spots$Frame))

norm_list <- lapply(split(spots, spots$Track.ID), function(df) {
  df <- df[order(df$Frame), ]

  raw_range  <- range(df$Mean.ch2,      na.rm = TRUE)
  corr_range <- range(df$ch2_corrected, na.rm = TRUE)

  raw_span  <- diff(raw_range)
  corr_span <- diff(corr_range)

  df$raw_norm  <- if (raw_span  > 1e-9) (df$Mean.ch2      - raw_range[1])  / raw_span  else NA_real_
  df$corr_norm <- if (corr_span > 1e-9) (df$ch2_corrected - corr_range[1]) / corr_span else NA_real_
  df[, c("Track.ID", "Frame", "T..sec.", "raw_norm", "corr_norm")]
})

norm_df <- do.call(rbind, norm_list)

pop <- do.call(rbind, lapply(split(norm_df, norm_df$Frame), function(d) {
  data.frame(
    Frame     = d$Frame[1],
    t_min     = d[["T..sec."]][1] / 60,
    raw_mean  = mean(d$raw_norm,  na.rm = TRUE),
    raw_sem   = sd(d$raw_norm,  na.rm = TRUE) / sqrt(sum(!is.na(d$raw_norm))),
    corr_mean = mean(d$corr_norm, na.rm = TRUE),
    corr_sem  = sd(d$corr_norm, na.rm = TRUE) / sqrt(sum(!is.na(d$corr_norm))),
    n         = sum(!is.na(d$raw_norm))
  )
}))
pop <- pop[order(pop$Frame), ]

png(file.path(out_dir, "plot5_population_mean_gfp.png"),
    width = 900, height = 550, type = "cairo")

ylim <- range(c(pop$raw_mean  - pop$raw_sem,  pop$raw_mean  + pop$raw_sem,
                pop$corr_mean - pop$corr_sem, pop$corr_mean + pop$corr_sem),
              na.rm = TRUE)

plot(pop$t_min, pop$raw_mean,
     type = "l", col = "darkgreen", lwd = 2,
     ylim = ylim,
     xlab = "Time (min)", ylab = "Normalised GFP (mean ± SEM across cells)",
     main = "Population mean GFP: raw vs BFP-corrected\n(each cell normalised 0–1 to its own range)")

polygon(c(pop$t_min, rev(pop$t_min)),
        c(pop$raw_mean + pop$raw_sem, rev(pop$raw_mean - pop$raw_sem)),
        col = rgb(0, 0.5, 0, 0.15), border = NA)

lines(pop$t_min, pop$corr_mean, col = "chartreuse3", lwd = 2, lty = 2)
polygon(c(pop$t_min, rev(pop$t_min)),
        c(pop$corr_mean + pop$corr_sem, rev(pop$corr_mean - pop$corr_sem)),
        col = rgb(0.4, 0.8, 0, 0.15), border = NA)

abline(h = 0, col = "grey80", lty = 3)
legend("topleft", bty = "n",
       legend = c("Raw GFP", "Corrected GFP"),
       col = c("darkgreen", "chartreuse3"), lwd = 2, lty = c(1, 2))
dev.off()
cat("Saved plot5_population_mean_gfp.png\n")

cat("\nDone.  α =", round(alpha, 6), "\n")

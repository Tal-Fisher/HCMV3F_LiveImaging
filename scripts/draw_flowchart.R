options(bitmapType = "cairo")

base_dir <- "/home/labs/ginossar/talfis/LiveImaging"
fig_dir  <- file.path(base_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

png(file.path(fig_dir, "pipeline_flowchart.png"),
    width = 1800, height = 2400, res = 150, type = "cairo")

par(mar = c(0.5, 0.5, 1.5, 0.5), bg = "white")
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1))

# ── helpers ───────────────────────────────────────────────────────────────────
box <- function(x, y, w, h, label, sub = NULL,
                fill = "#DDEEFF", border = "#336699", lwd = 1.5,
                cex = 0.72, cex_sub = 0.6, col = "black", radius = 0.012) {
  # rounded-corner approximation with rect + clipping
  rect(x - w/2, y - h/2, x + w/2, y + h/2,
       col = fill, border = border, lwd = lwd)
  if (is.null(sub)) {
    text(x, y, label, cex = cex, col = col, font = 2, adj = c(0.5, 0.5))
  } else {
    text(x, y + h * 0.18, label, cex = cex, col = col, font = 2, adj = c(0.5, 0.5))
    text(x, y - h * 0.12, sub,   cex = cex_sub, col = "#333333", adj = c(0.5, 0.5))
  }
}

box_multi <- function(x, y, w, h, lines, fill = "#DDEEFF",
                      border = "#336699", lwd = 1.5, cex = 0.6, col = "black") {
  rect(x - w/2, y - h/2, x + w/2, y + h/2,
       col = fill, border = border, lwd = lwd)
  n  <- length(lines)
  ys <- seq(y + h/2 - h/(n+1), y - h/2 + h/(n+1), length.out = n)
  for (i in seq_along(lines)) {
    f <- if (i == 1) 2 else 1
    text(x, ys[i], lines[i], cex = cex, col = col, font = f, adj = c(0.5, 0.5))
  }
}

arr <- function(x0, y0, x1, y1, col = "#336699", lwd = 1.5) {
  arrows(x0, y0, x1, y1, length = 0.08, angle = 20,
         col = col, lwd = lwd)
}

dbox <- function(x, y, w, h, label,
                 fill = "#FFF3CC", border = "#CC9900", lwd = 1.5, cex = 0.62) {
  # diamond for cache/data nodes
  rect(x - w/2, y - h/2, x + w/2, y + h/2,
       col = fill, border = border, lwd = lwd, lty = 2)
  text(x, y, label, cex = cex, col = "#664400", font = 3, adj = c(0.5, 0.5))
}

removed_box <- function(x, y, w, h, label, cex = 0.58) {
  rect(x - w/2, y - h/2, x + w/2, y + h/2,
       col = "#FFE0E0", border = "#CC3333", lwd = 1.5, lty = 2)
  text(x, y + h * 0.2, label, cex = cex, col = "#CC0000", font = 2, adj = c(0.5, 0.5))
  text(x, y - h * 0.2, "REMOVED", cex = cex * 0.9, col = "#CC0000", font = 1, adj = c(0.5, 0.5))
}

# ── layout constants ──────────────────────────────────────────────────────────
bw  <- 0.30   # standard box width
bh  <- 0.055  # standard box height
dw  <- 0.22   # data node width
dh  <- 0.040  # data node height
rw  <- 0.20   # removed-box width
rh  <- 0.042

title_col  <- "#1A3A5C"
script_col <- "#EEF4FF"
script_brd <- "#336699"
data_fill  <- "#FFFBE6"
data_brd   <- "#B8860B"
rm_fill    <- "#FFE8E8"
rm_brd     <- "#CC3333"
out_fill   <- "#E8F5E9"
out_brd    <- "#2E7D32"

# ── title ─────────────────────────────────────────────────────────────────────
text(0.5, 0.985, "HCMV Live Imaging Analysis Pipeline",
     cex = 1.1, font = 2, col = title_col, adj = c(0.5, 0.5))

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1: Input CSVs  (y = 0.93)
# ─────────────────────────────────────────────────────────────────────────────
y_in <- 0.93
box(0.28, y_in, 0.24, dh + 0.01,
    "A2 raw CSVs", "(spots + nuclei)",
    fill = "#E8F0FE", border = "#3355AA", cex = 0.72, cex_sub = 0.60)
box(0.72, y_in, 0.24, dh + 0.01,
    "A3 raw CSVs", "(spots + nuclei)",
    fill = "#E8F0FE", border = "#3355AA", cex = 0.72, cex_sub = 0.60)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2: 01_preprocess.R  (y = 0.845)
# ─────────────────────────────────────────────────────────────────────────────
y_pre <- 0.845
bh_pre <- 0.085
box_multi(0.5, y_pre, 0.56, bh_pre,
          c("01_preprocess.R  (run per dataset)",
            "Filter tracks < 10 frames; exclude early-GFP cells (green > 1.5 at t₀)",
            "Fit BFP→GFP bleedthrough α per movie (LM, first 5 frames, origin)",
            "ch2_corrected = Mean.ch2 − α × Mean.ch1"),
          fill = script_col, border = script_brd, cex = 0.64)

arr(0.28, y_in - (dh+0.01)/2, 0.40, y_pre + bh_pre/2)
arr(0.72, y_in - (dh+0.01)/2, 0.60, y_pre + bh_pre/2)

# outputs of preprocess
y_pre_out <- 0.772
dbox(0.30, y_pre_out, dw, dh, "spots_clean.rds", fill = data_fill, border = data_brd)
dbox(0.70, y_pre_out, dw, dh, "nuc_raw.rds",     fill = data_fill, border = data_brd)
arr(0.37, y_pre - bh_pre/2, 0.34, y_pre_out + dh/2)
arr(0.63, y_pre - bh_pre/2, 0.66, y_pre_out + dh/2)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3: 02_nuclei.R  (y = 0.700)
# ─────────────────────────────────────────────────────────────────────────────
y_nuc <- 0.700
bh_nuc <- 0.068
box_multi(0.70, y_nuc, 0.42, bh_nuc,
          c("02_nuclei.R  (per dataset)",
            "Match nucleus to cell track by proximity",
            "Keep best nucleus per cell per frame"),
          fill = script_col, border = script_brd, cex = 0.64)

arr(0.70, y_pre_out - dh/2, 0.70, y_nuc + bh_nuc/2)

y_nuc_out <- 0.636
dbox(0.70, y_nuc_out, dw, dh, "nuc_assigned.rds", fill = data_fill, border = data_brd)
arr(0.70, y_nuc - bh_nuc/2, 0.70, y_nuc_out + dh/2)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4: 03_onset.R  (y = 0.557)
# ─────────────────────────────────────────────────────────────────────────────
y_ons <- 0.557
bh_ons <- 0.090
box_multi(0.40, y_ons, 0.58, bh_ons,
          c("03_onset.R  (per dataset)",
            "Smooth fluorescence ±1 frame",
            "GFP onset: thresh=0.2, 3 consec frames, no Δ constraint",
            "BFP onset: thresh=0.5, 3 consec frames, no Δ constraint",
            "mCherry onset: thresh=2.25, 5 consec frames, Δ>0.15"),
          fill = script_col, border = script_brd, cex = 0.63)

arr(0.30, y_pre_out - dh/2, 0.30, y_ons + bh_ons/2)   # spots_clean -> onset
arr(0.70, y_nuc_out - dh/2, 0.57, y_ons + bh_ons/2)   # nuc_assigned -> onset

# outputs
y_ons_out <- 0.483
dbox(0.30, y_ons_out, dw, dh, "onset_df.rds", fill = data_fill, border = data_brd)
arr(0.35, y_ons - bh_ons/2, 0.33, y_ons_out + dh/2)

# flagged (removed)
removed_box(0.72, y_ons_out, rw, rh,
            "onset_flagged.rds\n(red without green or blue)")
arr(0.53, y_ons - bh_ons/2, 0.70, y_ons_out + rh/2,
    col = "#CC3333")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5: 04_features.R  (y = 0.400)
# ─────────────────────────────────────────────────────────────────────────────
y_feat <- 0.400
bh_feat <- 0.100
box_multi(0.40, y_feat, 0.60, bh_feat,
          c("04_features.R  (per dataset)",
            "Window: 16 frames from GFP onset (fallback: track start)",
            "Exclude onset_flagged cells",
            "27 features: corrected GFP, nuclear BFP, nucleus morphology,",
            "nuc/cell ratio, cell area, solidity, shape index, SNR/contrast,",
            "delay_green_to_blue"),
          fill = script_col, border = script_brd, cex = 0.62)

arr(0.30, y_ons_out - dh/2,  0.26, y_feat + bh_feat/2)  # spots_clean (re-used)
arr(0.70, y_nuc_out - dh/2,  0.60, y_feat + bh_feat/2)  # nuc_assigned (re-used)
arr(0.30, y_ons_out - dh/2,  0.35, y_feat + bh_feat/2)  # onset_df

y_feat_out <- 0.318
dbox(0.30, y_feat_out, dw + 0.02, dh, "feat_df.rds / model_df.rds",
     fill = data_fill, border = data_brd)
arr(0.37, y_feat - bh_feat/2, 0.32, y_feat_out + dh/2)

removed_box(0.72, y_feat_out, rw + 0.02, rh,
            "no_nucleus_16.rds\n(no nucleus in feature window)")
arr(0.53, y_feat - bh_feat/2, 0.70, y_feat_out + rh/2,
    col = "#CC3333")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 6: 05 + 06 (parallel)  (y = 0.235)
# ─────────────────────────────────────────────────────────────────────────────
y_en  <- 0.242
bh_en <- 0.082
box_multi(0.22, y_en, 0.36, bh_en,
          c("05_elasticnet.R  (per dataset)",
            "Z-normalize; median impute NAs",
            "ElasticNet α=0.5, 5-fold CV",
            "Outcome: delay_green_to_red"),
          fill = script_col, border = script_brd, cex = 0.63)

box_multi(0.66, y_en, 0.36, bh_en,
          c("06_cox.R  (per dataset)",
            "Cox ElasticNet α=0.5",
            "Event: red onset (1=yes, 0=censored)",
            "C-statistic, KM curves by tertile"),
          fill = script_col, border = script_brd, cex = 0.63)

arr(0.28, y_feat_out - dh/2, 0.22, y_en + bh_en/2)
arr(0.36, y_feat_out - dh/2, 0.66, y_en + bh_en/2)

y_en_out <- 0.168
dbox(0.22, y_en_out, dw - 0.02, dh, "en_results.rds",
     fill = data_fill, border = data_brd)
dbox(0.66, y_en_out, dw - 0.02, dh, "cox_results.rds",
     fill = data_fill, border = data_brd)
arr(0.22, y_en - bh_en/2, 0.22, y_en_out + dh/2)
arr(0.66, y_en - bh_en/2, 0.66, y_en_out + dh/2)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 7: 07_combine.R  (y = 0.100)
# ─────────────────────────────────────────────────────────────────────────────
y_comb <- 0.100
bh_comb <- 0.068
box_multi(0.50, y_comb, 0.72, bh_comb,
          c("07_combine.R",
            "Prefix Track.IDs (A2_ / A3_); merge datasets; combined ElasticNet + Cox"),
          fill = script_col, border = script_brd, cex = 0.65)

arr(0.36, y_feat_out - dh/2,   0.42, y_comb + bh_comb/2)  # feat_df
arr(0.22, y_en_out  - dh/2,    0.42, y_comb + bh_comb/2)  # en_results
arr(0.66, y_en_out  - dh/2,    0.58, y_comb + bh_comb/2)  # cox_results

# ─────────────────────────────────────────────────────────────────────────────
# ROW 8: outputs of combine  (y = 0.030)
# ─────────────────────────────────────────────────────────────────────────────
y_out <- 0.030
ow <- 0.27
oh <- 0.050
xs <- c(0.14, 0.38, 0.63, 0.87)
labels <- c(
  "Combined EN + Cox\n(all cells)",
  "Green-normalized Cox\n(t₀ = IE gene expression)",
  "Early-green subgroup\n(GFP onset ≤ 1000 min)",
  "Population dynamics\n(60-min bins, rolling mean)"
)
for (i in seq_along(xs)) {
  rect(xs[i] - ow/2, y_out - oh/2, xs[i] + ow/2, y_out + oh/2,
       col = out_fill, border = out_brd, lwd = 1.5)
  text(xs[i], y_out, labels[i], cex = 0.60, col = "#1B5E20", font = 2, adj = c(0.5, 0.5))
  arr(0.50, y_comb - bh_comb/2, xs[i], y_out + oh/2, col = out_brd)
}

dev.off()
cat("Saved figures/pipeline_flowchart.png\n")

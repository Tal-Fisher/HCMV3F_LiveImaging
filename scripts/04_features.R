options(bitmapType = "cairo")

dataset  <- "A2"   # change to "A3" or "combined"

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", dataset)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# ── load upstream caches ───────────────────────────────────────────────────────
for (f in c("spots_clean.rds","nuc_assigned.rds","onset_df.rds")) {
  if (!file.exists(file.path(cache_dir, f)))
    stop(sprintf("Missing cache/%s/%s — run the corresponding script first", dataset, f))
}
spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))
onset_df     <- readRDS(file.path(cache_dir, "onset_df.rds"))
flagged_path <- file.path(cache_dir, "onset_flagged.rds")
flagged_ids  <- if (file.exists(flagged_path)) readRDS(flagged_path)$Track.ID else character(0)
cat(sprintf("=== %s: feature extraction (%d cells, %d with nucleus, %d red events, %d flagged/excluded) ===\n",
            dataset,
            length(unique(spots$Track.ID)),
            length(unique(nuc_assigned$Track.ID)),
            sum(!is.na(onset_df$red_onset_min)),
            length(flagged_ids)))

# ── helpers ───────────────────────────────────────────────────────────────────
safe_slope <- function(x, t) {
  keep <- is.finite(x) & is.finite(t)
  if (sum(keep) < 2) return(NA_real_)
  unname(coef(lm(x[keep] ~ t[keep]))[2])
}
safe_sd    <- function(x) { if (sum(!is.na(x)) < 2) NA_real_ else sd(x, na.rm=TRUE) }
safe_first <- function(x) { x <- x[is.finite(x)]; if (!length(x)) NA_real_ else x[1] }

# ── feature extraction (first 16 frames per cell) ─────────────────────────────
n_feat   <- 16
cell_ids <- setdiff(intersect(unique(spots$Track.ID), unique(nuc_assigned$Track.ID)), flagged_ids)

feat_list <- lapply(cell_ids, function(tid) {
  cs_all <- spots[spots$Track.ID == tid, ]
  cs_all <- cs_all[order(cs_all$T..sec.), ]
  t0     <- cs_all$T..sec.[1]

  # start window at GFP onset; fall back to track start if onset not detected
  g_min   <- onset_df$green_onset_min[onset_df$Track.ID == tid]
  g_min   <- if (length(g_min) == 1 && !is.na(g_min)) g_min else 0
  g_sec   <- g_min * 60 + t0
  onset_i <- which(cs_all$T..sec. >= g_sec)
  onset_i <- if (length(onset_i) == 0) 1L else onset_i[1]

  # cap window at red onset to prevent leakage; exclude if < n_feat clean frames
  r_min  <- onset_df$red_onset_min[onset_df$Track.ID == tid]
  r_sec  <- if (length(r_min) == 1 && !is.na(r_min)) r_min * 60 + t0 else Inf
  red_i  <- which(cs_all$T..sec. >= r_sec)
  red_i  <- if (length(red_i) == 0) nrow(cs_all) + 1L else red_i[1]
  cs <- cs_all[onset_i:min(onset_i + n_feat - 1L, red_i - 1L, nrow(cs_all)), ]
  if (nrow(cs) < n_feat) return(NULL)

  ns <- nuc_assigned[nuc_assigned$Track.ID == tid, ]
  ns <- ns[order(ns$T..sec.), ]

  merged <- merge(
    cs[, c("Frame","ch2_corrected","Mean.ch2","Mean.ch1","Mean.ch3",
            "Area","Solidity","Shape.index","SNR.ch2","SNR.ch4","Ctrst.ch4")],
    ns[, c("Frame","Mean.ch1","Area","Circ.")],
    by="Frame", suffixes=c("_cell","_nuc"), all.x=TRUE)
  merged <- merged[order(merged$Frame), ]
  nf     <- seq_len(nrow(merged)) - 1
  nuc_ratio <- ifelse(merged$Area_cell > 0,
                      merged$Area_nuc / merged$Area_cell, NA_real_)
  # GFP / nuclear BFP: normalises expression level differences (MOI, cell volume)
  gfp_bfp_ratio <- ifelse(!is.na(merged$Mean.ch1_nuc) & merged$Mean.ch1_nuc > 0,
                          merged$ch2_corrected / merged$Mean.ch1_nuc, NA_real_)

  data.frame(
    Track.ID = tid,

    # corrected GFP
    gfp_corr_start = safe_first(merged$ch2_corrected),
    gfp_corr_mean  = mean(merged$ch2_corrected, na.rm=TRUE),
    gfp_corr_sd    = safe_sd(merged$ch2_corrected),
    gfp_corr_slope = safe_slope(merged$ch2_corrected, nf),

    # nuclear BFP
    nuc_bfp_start  = safe_first(merged$Mean.ch1_nuc),
    nuc_bfp_mean   = mean(merged$Mean.ch1_nuc,  na.rm=TRUE),
    nuc_bfp_sd     = safe_sd(merged$Mean.ch1_nuc),
    nuc_bfp_slope  = safe_slope(merged$Mean.ch1_nuc, nf),

    # nucleus morphology
    nuc_area_mean  = mean(merged$Area_nuc,  na.rm=TRUE),
    nuc_area_slope = safe_slope(merged$Area_nuc, nf),
    nuc_circ_mean  = mean(merged$Circ.,     na.rm=TRUE),
    nuc_circ_sd    = safe_sd(merged$Circ.),

    # nucleus/cell area ratio
    nuc_ratio_mean  = mean(nuc_ratio, na.rm=TRUE),
    nuc_ratio_slope = safe_slope(nuc_ratio, nf),

    # cell area
    area_start = safe_first(merged$Area_cell),
    area_mean  = mean(merged$Area_cell, na.rm=TRUE),
    area_sd    = safe_sd(merged$Area_cell),
    area_slope = safe_slope(merged$Area_cell, nf),

    # morphology
    solidity_mean  = mean(merged$Solidity,    na.rm=TRUE),
    solidity_sd    = safe_sd(merged$Solidity),
    shape_idx_mean = mean(merged$Shape.index, na.rm=TRUE),

    # SNR / contrast
    gfp_snr_mean  = mean(merged$SNR.ch2,   na.rm=TRUE),
    gfp_snr_sd    = safe_sd(merged$SNR.ch2),
    bf_snr_mean   = mean(merged$SNR.ch4,   na.rm=TRUE),
    bf_ctrst_mean = mean(merged$Ctrst.ch4, na.rm=TRUE),
    bf_ctrst_sd   = safe_sd(merged$Ctrst.ch4),

    # GFP / nuclear BFP ratio: normalises MOI and cell-level expression variation
    gfp_ratio_start = safe_first(gfp_bfp_ratio),
    gfp_ratio_mean  = mean(gfp_bfp_ratio,  na.rm=TRUE),
    gfp_ratio_sd    = safe_sd(gfp_bfp_ratio),
    gfp_ratio_slope = safe_slope(gfp_bfp_ratio, nf),
    gfp_ratio_max   = if (sum(is.finite(gfp_bfp_ratio)) > 0) max(gfp_bfp_ratio, na.rm=TRUE) else NA_real_
  )
})

n_excluded_red <- sum(vapply(feat_list, is.null, logical(1)))
feat_list <- Filter(Negate(is.null), feat_list)
cat(sprintf("Excluded %d cells (< %d frames before red onset)\n", n_excluded_red, n_feat))
feat_df <- do.call(rbind, feat_list)

# ── flag and remove cells with no nucleus in first 16 frames ──────────────────
no_nuc_mask <- is.na(feat_df$nuc_bfp_mean) | is.nan(feat_df$nuc_bfp_mean)
no_nuc_df   <- feat_df[no_nuc_mask, ]
feat_df     <- feat_df[!no_nuc_mask, ]
cat(sprintf("Flagged %d cells (no nucleus in first 16 frames) — saved to no_nucleus_16.rds\n",
            nrow(no_nuc_df)))
saveRDS(no_nuc_df, file.path(cache_dir, "no_nucleus_16.rds"))

# ── add delay_green_to_blue from onset_df ─────────────────────────────────────
feat_df <- merge(feat_df,
                 onset_df[, c("Track.ID","delay_green_to_blue")],
                 by="Track.ID", all.x=TRUE)

model_df <- merge(feat_df,
                  onset_df[, c("Track.ID","delay_green_to_red")],
                  by="Track.ID", all.x=TRUE)
cat(sprintf("Feature matrix: %d cells x %d features\n",
            nrow(model_df), ncol(feat_df) - 1))

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(feat_df,  file.path(cache_dir, "feat_df.rds"))
saveRDS(model_df, file.path(cache_dir, "model_df.rds"))
cat(sprintf("Saved cache/%s/feat_df.rds + model_df.rds\n", dataset))

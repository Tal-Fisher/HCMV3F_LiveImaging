options(bitmapType = "cairo")

dataset  <- "A2"   # change to "A3" for the other experiment

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
data_dir  <- file.path(base_dir, "CompleteImage")
cache_dir <- file.path(base_dir, "cache", dataset)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# ── load ───────────────────────────────────────────────────────────────────────
cat(sprintf("=== %s: preprocess ===\n", dataset))
spots <- read.csv(file.path(data_dir, sprintf("%s_Merged_spots.csv",  dataset)))
nuc   <- read.csv(file.path(data_dir, sprintf("%s_nuclei_spots.csv",  dataset)))

spots <- spots[!is.na(spots$Track.ID), ]
nuc   <- nuc[!is.na(nuc$Track.ID), ]
spots <- spots[order(spots$Track.ID, spots$T..sec.), ]
nuc   <- nuc[order(nuc$Track.ID, nuc$T..sec.), ]

# ── B3 nucleus columns use verbose names; normalise to A2/A3 convention ───────
if (dataset == "B3") {
  rename_map <- c("Mean.intensity.ch1"    = "Mean.ch1",
                  "Mean.intensity.ch2"    = "Mean.ch2",
                  "Mean.intensity.ch3"    = "Mean.ch3",
                  "Circularity"           = "Circ.",
                  "Probability.stuck"     = "P.stuck",
                  "Probability.diffusive" = "P.diffusive")
  for (old in names(rename_map)) {
    idx <- which(names(nuc) == old)
    if (length(idx)) names(nuc)[idx] <- rename_map[[old]]
  }
  cat("B3: nucleus columns renamed to A2/A3 convention\n")
}

# ── remove split tracks (GFP on at track start) ───────────────────────────────
first_gfp <- tapply(spots$Mean.ch2, spots$Track.ID, function(x) x[1])
keep_ids  <- as.integer(names(first_gfp)[first_gfp < 2.5])
spots     <- spots[spots$Track.ID %in% keep_ids, ]

# ── bleedthrough correction (Pacific Blue → GFP, theoretical α) ───────────────
# Simultaneous 405/488 nm acquisition; Pacific Blue emission at 500-550 nm
# is ~3-8% of its peak (455 nm) → α ≈ 0.05 (physically motivated upper bound).
# The data-fitted α (~3.25) was ~65× too large because it captured biological
# co-expression of GFP and BFP, not optical bleedthrough.
alpha <- 0.05
spots$ch2_corrected <- spots$Mean.ch2 - alpha * spots$Mean.ch1
cat(sprintf("Bleedthrough alpha: %.5f  (fixed theoretical value)\n", alpha))

# ── remove cells with corrected GFP already high at track start ───────────────
# Threshold matches filter 1 (Mean.ch2 < 2.5); with α=0.05 ch2_corrected ≈ Mean.ch2
first_corr <- vapply(split(spots, spots$Track.ID),
                     function(df) df$ch2_corrected[1], numeric(1))
spots <- spots[spots$Track.ID %in% as.integer(names(first_corr)[first_corr < 2.5]), ]
cat(sprintf("Cells after early-GFP filter: %d\n", length(unique(spots$Track.ID))))

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(spots, file.path(cache_dir, "spots_clean.rds"))
saveRDS(nuc,   file.path(cache_dir, "nuc_raw.rds"))
cat(sprintf("Saved cache/%s/spots_clean.rds + nuc_raw.rds\n", dataset))

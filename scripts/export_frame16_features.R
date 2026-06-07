options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs", .libPaths()))

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
export_dir <- file.path(base_dir, "cache", "python_export")

compute_frame16 <- function(dataset) {
  cache_dir    <- file.path(base_dir, "cache", dataset)
  spots        <- readRDS(file.path(cache_dir, "spots_clean.rds"))
  nuc_assigned <- readRDS(file.path(cache_dir, "nuc_assigned.rds"))

  # merge GFP (ch2_corrected) and BFP (Mean.ch1) per frame per cell
  spots_m <- merge(
    spots[, c("Track.ID", "Frame", "T..sec.", "ch2_corrected")],
    nuc_assigned[, c("Track.ID", "Frame", "Mean.ch1")],
    by = c("Track.ID", "Frame"), all.x = TRUE
  )
  spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

  results <- lapply(split(spots_m, spots_m$Track.ID), function(d) {
    d <- d[order(d$T..sec.), ]

    # take first 16 frames
    n    <- nrow(d)
    f16  <- min(n, 16)
    d16  <- d[seq_len(f16), ]

    # feature 1: GFP at frame 16 (last of first 16)
    gfp_at_f16 <- d16$ch2_corrected[f16]

    # feature 2: BFP at frame 16
    bfp_at_f16 <- d16$Mean.ch1[f16]

    # feature 3: mean GFP change per frame (mean of consecutive differences)
    gfp_vals <- d16$ch2_corrected
    gfp_delta_mean <- if (length(gfp_vals) >= 2) mean(diff(gfp_vals), na.rm = TRUE) else NA_real_

    # feature 4: mean BFP change per frame
    bfp_vals <- d16$Mean.ch1
    bfp_delta_mean <- if (length(bfp_vals) >= 2 && sum(!is.na(bfp_vals)) >= 2)
                        mean(diff(bfp_vals), na.rm = TRUE) else NA_real_

    data.frame(
      Track.ID       = d$Track.ID[1],
      gfp_at_f16     = gfp_at_f16,
      bfp_at_f16     = bfp_at_f16,
      gfp_delta_mean = gfp_delta_mean,
      bfp_delta_mean = bfp_delta_mean
    )
  })

  out <- do.call(rbind, results)
  # prefix Track.ID with dataset to match proximity format (e.g. "A2_1000")
  out$Track.ID <- paste0(dataset, "_", out$Track.ID)
  cat(sprintf("%s: %d cells, frame-16 features computed\n", dataset, nrow(out)))
  out
}

f16_a2 <- compute_frame16("A2")
f16_a3 <- compute_frame16("A3")
f16_all <- rbind(f16_a2, f16_a3)

# also export category labels (early/mid/late/non-productive)
cat_df <- readRDS(file.path(base_dir, "cache", "combined", "category_df.rds"))
# Track.ID in category_df already has dataset prefix (check)
cat(sprintf("category_df Track.ID sample: %s\n", paste(head(cat_df$Track.ID, 3), collapse=", ")))

write.csv(f16_all, file.path(export_dir, "frame16_features.csv"), row.names = FALSE)
write.csv(cat_df,  file.path(export_dir, "category_df.csv"),      row.names = FALSE)
cat(sprintf("Saved frame16_features.csv (%d rows) and category_df.csv\n", nrow(f16_all)))

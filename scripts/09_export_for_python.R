options(bitmapType = "cairo")

base_dir   <- "/home/labs/ginossar/talfis/LiveImaging"
cache_comb <- file.path(base_dir, "cache", "combined")
export_dir <- file.path(base_dir, "cache", "python_export")
dir.create(export_dir, recursive = TRUE, showWarnings = FALSE)

model_df <- readRDS(file.path(cache_comb, "model_df.rds"))
onset_df <- readRDS(file.path(cache_comb, "onset_df.rds"))

# movie duration per dataset (max T..sec. across all tracked cells)
dur_min <- sapply(c("A2","A3"), function(ds) {
  spots <- readRDS(file.path(base_dir, "cache", ds, "spots_clean.rds"))
  max(spots$T..sec., na.rm = TRUE) / 60
})
cat(sprintf("Movie durations:  A2=%.0f min  A3=%.0f min\n", dur_min["A2"], dur_min["A3"]))

# track start time (absolute minutes from movie start) per cell
track_start_min <- do.call(c, lapply(c("A2","A3"), function(ds) {
  spots <- readRDS(file.path(base_dir, "cache", ds, "spots_clean.rds"))
  ts    <- tapply(spots$T..sec., spots$Track.ID, min) / 60
  setNames(ts, paste0(ds, "_", names(ts)))
}))

# attach green_onset_min, track_start_min, and movie half-duration to model_df
onset_slim <- onset_df[, c("Track.ID", "green_onset_min", "dataset")]
model_df   <- merge(model_df, onset_slim, by = "Track.ID", all.x = TRUE,
                    suffixes = c("", "_onset"))
# resolve dataset column collision
if ("dataset_onset" %in% names(model_df)) {
  model_df$dataset_onset <- NULL
}
model_df$track_start_min  <- track_start_min[model_df$Track.ID]
model_df$abs_gfp_onset_min <- model_df$track_start_min + model_df$green_onset_min
model_df$movie_half_min   <- dur_min[model_df$dataset] / 2

write.csv(model_df,    file.path(export_dir, "model_df.csv"),    row.names = FALSE)
write.csv(data.frame(dataset=names(dur_min), duration_min=dur_min, row.names=NULL),
          file.path(export_dir, "movie_durations.csv"), row.names = FALSE)

cat(sprintf("Exported %d cells to cache/python_export/\n", nrow(model_df)))
cat(sprintf("Columns: %s\n", paste(names(model_df), collapse=", ")))

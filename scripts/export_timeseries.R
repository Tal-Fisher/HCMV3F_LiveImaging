options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))

BASE    <- "/home/labs/ginossar/talfis/LiveImaging"
OUT_CSV <- file.path(BASE, "cache/python_export/timeseries_data.csv")

cat("Loading spots_clean.rds ...\n")
spots <- readRDS(file.path(BASE, "cache/combined/spots_clean.rds"))

# rename ambiguous columns
names(spots)[names(spots) == "Area"]  <- "Area_cell"
names(spots)[names(spots) == "Circ."] <- "Circ_cell"

cat("Loading nuc_assigned.rds ...\n")
nuc <- readRDS(file.path(BASE, "cache/combined/nuc_assigned.rds"))

# select only nuclear columns and rename to avoid collision
nuc_sel <- nuc[, c("Track.ID", "Frame", "Mean.ch1", "Area", "Circ.")]
names(nuc_sel) <- c("Track.ID", "Frame", "Mean.ch1_nuc", "Area_nuc", "Circ_nuc")

# if multiple nuclei per (cell, frame), take the mean
nuc_agg <- aggregate(
  cbind(Mean.ch1_nuc, Area_nuc, Circ_nuc) ~ Track.ID + Frame,
  data = nuc_sel,
  FUN  = mean,
  na.action = na.pass
)

cat("Merging cell + nucleus ...\n")
df <- merge(spots, nuc_agg, by = c("Track.ID", "Frame"), all.x = TRUE)

# nucleus / cell area ratio
df$nuc_ratio <- ifelse(df$Area_cell > 0, df$Area_nuc / df$Area_cell, NA_real_)

cat("Loading onset_df.rds ...\n")
onset <- readRDS(file.path(BASE, "cache/combined/onset_df.rds"))
onset_sel <- onset[, c("Track.ID", "red_onset_min")]

cat("Loading model_df.csv ...\n")
md <- read.csv(file.path(BASE, "cache/python_export/model_df.csv"),
               stringsAsFactors = FALSE)
md_sel <- md[, c("Track.ID", "abs_gfp_onset_min", "movie_half_min")]

cat("Joining timing info ...\n")
df <- merge(df, onset_sel, by = "Track.ID", all.x = TRUE)
df <- merge(df, md_sel,   by = "Track.ID", all.x = TRUE)

# first-half filter: keep only cells that turned GFP before the movie midpoint
before <- !is.na(df$abs_gfp_onset_min) & !is.na(df$movie_half_min) &
          df$abs_gfp_onset_min <= df$movie_half_min
df <- df[before, ]
cat("After first-half filter:", nrow(df), "rows\n")

# absolute time in minutes
df$T_min <- df$T..sec. / 60

# keep only the columns the Python script needs
keep_cols <- c(
  "Track.ID", "dataset", "Frame", "T_min",
  "ch2_corrected",          # GFP (corrected for BFP bleedthrough)
  "Mean.ch1",               # cell BFP (cytoplasmic ch1)
  "Mean.ch3",               # mCherry (will rise at red onset)
  "Mean.ch1_nuc",           # nuclear BFP
  "Area_cell",              # cell area
  "Area_nuc",               # nucleus area
  "Circ_nuc",               # nucleus circularity
  "nuc_ratio",              # nucleus / cell area ratio
  "Solidity",               # cell solidity
  "Shape.index",            # cell shape index
  "El..long.axis",          # cell long axis
  "P.stuck",                # immobility probability
  "Ctrst.ch4",              # brightfield contrast
  "abs_gfp_onset_min",      # absolute GFP onset time (minutes from movie start)
  "red_onset_min",          # time from GFP onset to red onset (NA = non-productive)
  "movie_half_min"          # movie midpoint (for reference)
)
df_out <- df[, keep_cols]

# safe column name
names(df_out)[names(df_out) == "El..long.axis"] <- "El_long_axis"
names(df_out)[names(df_out) == "Shape.index"]   <- "Shape_index"

cat("Writing", nrow(df_out), "rows to", OUT_CSV, "\n")
write.csv(df_out, OUT_CSV, row.names = FALSE)
cat("Done.\n")

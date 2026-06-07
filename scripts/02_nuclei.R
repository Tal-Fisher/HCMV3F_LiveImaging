options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
library(dplyr)
library(purrr)

dataset              <- "A2"   # change to "A3"
reverse_checkpoints  <- TRUE   # TRUE = anchor checkpoints at last frame, check backward (last 50%)
out_suffix           <- if (reverse_checkpoints) "_reverse" else ""

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", dataset)

# ── load upstream cache ────────────────────────────────────────────────────────
if (!file.exists(file.path(cache_dir, "spots_clean.rds"))) stop("Run 01_preprocess.R first")
spots_table <- readRDS(file.path(cache_dir, "spots_clean.rds"))
nuc_table   <- readRDS(file.path(cache_dir, "nuc_raw.rds"))
cat(sprintf("=== %s: nucleus assignment (%d cells) ===\n",
            dataset, length(unique(spots_table$Track.ID))))

spots_table <- spots_table %>%
  mutate(X = as.numeric(as.character(X)),
         Y = as.numeric(as.character(Y))) %>%
  arrange(`Track.ID`, `T..sec.`)

nuc_table <- nuc_table %>%
  mutate(X = as.numeric(as.character(X)),
         Y = as.numeric(as.character(Y))) %>%
  arrange(`Track.ID`, `T..sec.`)

# ── 1. frame-wise nearest matches ─────────────────────────────────────────────
make_frame_matches_nearest <- function(t_now, nuc_table, spots_table, max_dist = 100) {
  nuc_sub <- nuc_table %>%
    filter(`T..sec.` == t_now) %>%
    transmute(nuc_track = `Track.ID`, t = `T..sec.`, nuc_x = X, nuc_y = Y)
  cell_sub <- spots_table %>%
    filter(`T..sec.` == t_now) %>%
    transmute(cell_track = `Track.ID`, t = `T..sec.`, cell_x = X, cell_y = Y)
  if (nrow(nuc_sub) == 0 || nrow(cell_sub) == 0) return(NULL)
  all_pairs <- merge(cell_sub, nuc_sub, by = "t") %>%
    mutate(centroid_dist = sqrt((cell_x - nuc_x)^2 + (cell_y - nuc_y)^2)) %>%
    filter(centroid_dist <= max_dist)
  if (nrow(all_pairs) == 0) return(NULL)
  all_pairs %>%
    group_by(cell_track, t) %>%
    mutate(is_nearest_for_cell = centroid_dist == min(centroid_dist, na.rm = TRUE)) %>%
    ungroup()
}

cat("Computing frame-wise matches...\n")
all_times <- intersect(unique(nuc_table$`T..sec.`), unique(spots_table$`T..sec.`))
frame_matches <- map_dfr(all_times, make_frame_matches_nearest,
                          nuc_table = nuc_table, spots_table = spots_table, max_dist = 100)

# ── 2. checkpoint setup ────────────────────────────────────────────────────────
checkpoint_props <- if (reverse_checkpoints) seq(1, 0.5, by = -0.1) else seq(0, 1, by = 0.1)
cell_track_times <- spots_table %>%
  group_by(cell_track = `Track.ID`) %>%
  summarise(t_min = min(`T..sec.`, na.rm = TRUE),
            t_max = max(`T..sec.`, na.rm = TRUE), .groups = "drop")
checkpoint_targets <- cell_track_times %>%
  tidyr::crossing(check_prop = checkpoint_props) %>%
  mutate(target_t   = t_min + check_prop * (t_max - t_min),
         checkpoint = paste0("cp_", sprintf("%02d", round(check_prop * 100))))

# ── 3. checkpoint-level consistency ───────────────────────────────────────────
cat("Scoring checkpoint consistency...\n")
checkpoint_scores <- checkpoint_targets %>%
  inner_join(frame_matches, by = "cell_track", relationship = "many-to-many") %>%
  mutate(dt = abs(t - target_t)) %>%
  group_by(cell_track, nuc_track, checkpoint) %>%
  slice_min(dt, n = 1, with_ties = FALSE) %>%
  ungroup()

pair_checkpoint_summary <- checkpoint_scores %>%
  group_by(cell_track, nuc_track) %>%
  summarise(n_checkpoints_found = n(),
            cp_nearest_count    = sum(is_nearest_for_cell, na.rm = TRUE),
            cp_frac_nearest     = mean(is_nearest_for_cell, na.rm = TRUE),
            cp_mean_dist        = mean(centroid_dist, na.rm = TRUE),
            cp_median_dist      = median(centroid_dist, na.rm = TRUE),
            .groups = "drop")

# ── 4. full movie consistency ──────────────────────────────────────────────────
pair_all_frame_summary <- frame_matches %>%
  group_by(cell_track, nuc_track) %>%
  summarise(n_overlap_frames  = n(),
            nearest_count_all = sum(is_nearest_for_cell, na.rm = TRUE),
            frac_nearest_all  = mean(is_nearest_for_cell, na.rm = TRUE),
            mean_dist_all     = mean(centroid_dist, na.rm = TRUE),
            median_dist_all   = median(centroid_dist, na.rm = TRUE),
            max_dist_all      = max(centroid_dist, na.rm = TRUE),
            t_start           = min(t, na.rm = TRUE),
            t_end             = max(t, na.rm = TRUE),
            .groups = "drop")

pair_summary <- pair_all_frame_summary %>%
  left_join(pair_checkpoint_summary, by = c("cell_track", "nuc_track")) %>%
  mutate(n_checkpoints_found = coalesce(n_checkpoints_found, 0L),
         cp_nearest_count    = coalesce(cp_nearest_count, 0),
         cp_frac_nearest     = coalesce(cp_frac_nearest, 0),
         cp_mean_dist        = coalesce(cp_mean_dist, Inf),
         cp_median_dist      = coalesce(cp_median_dist, Inf))

# ── 5-6. best nucleus per cell, resolve conflicts ──────────────────────────────
sort_cols <- c("cp_nearest_count", "cp_frac_nearest", "frac_nearest_all",
               "cp_mean_dist", "mean_dist_all")
best_nucleus_per_cell <- pair_summary %>%
  group_by(cell_track) %>%
  arrange(desc(cp_nearest_count), desc(cp_frac_nearest),
          desc(frac_nearest_all), cp_mean_dist, mean_dist_all, .by_group = TRUE) %>%
  slice(1) %>% ungroup()

best_unique_assignments <- best_nucleus_per_cell %>%
  group_by(nuc_track) %>%
  arrange(desc(cp_nearest_count), desc(cp_frac_nearest),
          desc(frac_nearest_all), cp_mean_dist, mean_dist_all, .by_group = TRUE) %>%
  slice(1) %>% ungroup()

# ── 7. confidence filtering ────────────────────────────────────────────────────
good_assignments <- best_unique_assignments %>%
  filter(n_overlap_frames >= 5, frac_nearest_all >= 0.6, cp_frac_nearest >= 0.6)

one_to_one_ok <- nrow(good_assignments) == n_distinct(good_assignments$cell_track) &&
                 nrow(good_assignments) == n_distinct(good_assignments$nuc_track)
cat(sprintf("Assigned: %d / %d cells  |  1-to-1 clean: %s\n",
            nrow(good_assignments), length(unique(spots_table$Track.ID)), one_to_one_ok))

# ── 8. relabel nucleus table with cell Track.IDs ───────────────────────────────
nuc_table_filt <- nuc_table %>%
  inner_join(good_assignments %>% select(cell_track, nuc_track),
             by = c("Track.ID" = "nuc_track")) %>%
  mutate(`Track.ID` = cell_track) %>%
  select(-cell_track)

# ── save ───────────────────────────────────────────────────────────────────────
saveRDS(nuc_table_filt,   file.path(cache_dir, paste0("nuc_assigned",           out_suffix, ".rds")))
saveRDS(good_assignments, file.path(cache_dir, paste0("nuc_assignments_summary", out_suffix, ".rds")))
cat(sprintf("Saved cache/%s/nuc_assigned%s.rds\n", dataset, out_suffix))

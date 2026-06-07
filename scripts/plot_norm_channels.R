options(bitmapType = "cairo")

data_dir <- "/home/labs/ginossar/talfis/LiveImaging/CompleteImage"
out_dir  <- "/home/labs/ginossar/talfis/LiveImaging"

# ── load ───────────────────────────────────────────────────────────────────────
spots <- read.csv(file.path(data_dir, "A2_Merged_spots.csv"))
nuc   <- read.csv(file.path(data_dir, "A2_nuclei_spots.csv"))

spots <- spots[!is.na(spots$Track.ID), ]
nuc   <- nuc[!is.na(nuc$Track.ID), ]
spots <- spots[order(spots$Track.ID, spots$T..sec.), ]

# ── remove split tracks (GFP already on) ──────────────────────────────────────
first_gfp <- tapply(spots$Mean.ch2, spots$Track.ID, function(x) x[1])
keep_ids  <- as.integer(names(first_gfp)[first_gfp < 2.5])
spots     <- spots[spots$Track.ID %in% keep_ids, ]

# ── bleedthrough correction ───────────────────────────────────────────────────
n_baseline  <- 5
baseline_df <- do.call(rbind, lapply(split(spots, spots$Track.ID), function(df)
  head(df[order(df$T..sec.), ], n_baseline)))
baseline_df <- baseline_df[is.finite(baseline_df$Mean.ch1) & baseline_df$Mean.ch1 > 0 &
                            is.finite(baseline_df$Mean.ch2), ]
alpha <- unname(coef(lm(Mean.ch2 ~ Mean.ch1 - 1, data = baseline_df))[1])
spots$ch2_corrected <- spots$Mean.ch2 - alpha * spots$Mean.ch1
cat(sprintf("alpha = %.5f\n", alpha))

# ── remove cells with GFP already on ──────────────────────────────────────────
first_corr <- vapply(split(spots, spots$Track.ID),
                     function(df) df$ch2_corrected[1], numeric(1))
spots <- spots[spots$Track.ID %in% as.integer(names(first_corr)[first_corr < 1.5]), ]
cat(sprintf("Cells: %d\n", length(unique(spots$Track.ID))))

# ── assign nearest nucleus per cell (majority vote) ───────────────────────────
max_dist   <- 100
all_frames <- intersect(unique(spots$Frame), unique(nuc$Frame))

frame_matches <- do.call(rbind, lapply(all_frames, function(fr) {
  cs <- spots[spots$Frame == fr, c("Track.ID","X","Y")]
  ns <- nuc[nuc$Frame == fr,    c("Track.ID","X","Y")]
  if (!nrow(cs) || !nrow(ns)) return(NULL)
  pairs <- merge(
    data.frame(cell_id=cs$Track.ID, cx=cs$X, cy=cs$Y),
    data.frame(nuc_id =ns$Track.ID, nx=ns$X, ny=ns$Y), by=NULL)
  pairs$dist <- sqrt((pairs$cx-pairs$nx)^2 + (pairs$cy-pairs$ny)^2)
  pairs <- pairs[pairs$dist <= max_dist, ]
  if (!nrow(pairs)) return(NULL)
  pairs[ave(pairs$dist, pairs$cell_id, FUN=min) == pairs$dist,
        c("cell_id","nuc_id","dist")]
}))

best_nuc <- do.call(rbind, lapply(split(frame_matches, frame_matches$cell_id), function(df) {
  tab  <- sort(table(df$nuc_id), decreasing=TRUE)
  data.frame(cell_id=df$cell_id[1], nuc_id=as.integer(names(tab)[1]),
             frac=as.numeric(tab[1]/nrow(df)), n_fr=nrow(df))
}))
best_nuc <- best_nuc[best_nuc$frac >= 0.6 & best_nuc$n_fr >= 5, ]
best_nuc <- best_nuc[order(-best_nuc$frac), ]
best_nuc <- best_nuc[!duplicated(best_nuc$nuc_id), ]

nuc_assigned <- merge(nuc, best_nuc[, c("cell_id","nuc_id")],
                      by.x="Track.ID", by.y="nuc_id", all=FALSE)
nuc_assigned$Track.ID <- nuc_assigned$cell_id
nuc_assigned$cell_id  <- NULL

# ── merge nuclear BFP into spots by Track.ID + Frame ─────────────────────────
spots_m <- merge(
  spots[, c("Track.ID","Frame","T..sec.","ch2_corrected","Mean.ch3")],
  nuc_assigned[, c("Track.ID","Frame","Mean.ch1")],
  by = c("Track.ID","Frame"), all.x = TRUE
)
spots_m <- spots_m[order(spots_m$Track.ID, spots_m$T..sec.), ]

# ── per-cell min-max normalize, then average per frame ────────────────────────
sc01 <- function(x) {
  r <- range(x, na.rm = TRUE)
  if (diff(r) > 1e-9) (x - r[1]) / diff(r) else rep(NA_real_, length(x))
}

norm_list <- lapply(split(spots_m, spots_m$Track.ID), function(df) {
  df <- df[order(df$Frame), ]
  data.frame(
    norm_frame = seq_len(nrow(df)) - 1L,
    gfp_norm   = sc01(df$ch2_corrected),
    bfp_norm   = sc01(df$Mean.ch1),
    red_norm   = sc01(df$Mean.ch3)
  )
})
norm_df <- do.call(rbind, norm_list)

pop <- do.call(rbind, lapply(split(norm_df, norm_df$norm_frame), function(d) {
  data.frame(
    norm_frame = d$norm_frame[1],
    mean_gfp   = mean(d$gfp_norm, na.rm=TRUE),
    sem_gfp    = sd(d$gfp_norm, na.rm=TRUE) / sqrt(sum(!is.na(d$gfp_norm))),
    mean_bfp   = mean(d$bfp_norm, na.rm=TRUE),
    sem_bfp    = sd(d$bfp_norm, na.rm=TRUE) / sqrt(sum(!is.na(d$bfp_norm))),
    mean_red   = mean(d$red_norm, na.rm=TRUE),
    sem_red    = sd(d$red_norm, na.rm=TRUE) / sqrt(sum(!is.na(d$red_norm))),
    n          = sum(!is.na(d$gfp_norm))
  )
}))
pop <- pop[order(pop$norm_frame), ]

# ── plot ───────────────────────────────────────────────────────────────────────
png(file.path(out_dir, "plot_norm_channels.png"), width=900, height=550, type="cairo")

ylim <- range(c(
  pop$mean_gfp - pop$sem_gfp, pop$mean_gfp + pop$sem_gfp,
  pop$mean_bfp - pop$sem_bfp, pop$mean_bfp + pop$sem_bfp,
  pop$mean_red - pop$sem_red, pop$mean_red + pop$sem_red
), na.rm=TRUE)

plot(pop$norm_frame, pop$mean_gfp,
     type="l", col="green3", lwd=2, ylim=ylim,
     xlab="Normalized frame (from track start)",
     ylab="Normalized intensity (0–1, per cell)",
     main="Population mean normalized channel dynamics\n(each cell scaled 0–1 per channel; mean ± SEM)")

polygon(c(pop$norm_frame, rev(pop$norm_frame)),
        c(pop$mean_gfp + pop$sem_gfp, rev(pop$mean_gfp - pop$sem_gfp)),
        col=rgb(0, 0.7, 0, 0.15), border=NA)

lines(pop$norm_frame, pop$mean_bfp, col="dodgerblue", lwd=2)
polygon(c(pop$norm_frame, rev(pop$norm_frame)),
        c(pop$mean_bfp + pop$sem_bfp, rev(pop$mean_bfp - pop$sem_bfp)),
        col=rgb(0, 0.4, 1, 0.15), border=NA)

lines(pop$norm_frame, pop$mean_red, col="tomato", lwd=2)
polygon(c(pop$norm_frame, rev(pop$norm_frame)),
        c(pop$mean_red + pop$sem_red, rev(pop$mean_red - pop$sem_red)),
        col=rgb(1, 0.2, 0.2, 0.15), border=NA)

legend("topleft", bty="n",
       legend=c("Corrected GFP", "BFP (nuclear)", "mCherry"),
       col=c("green3","dodgerblue","tomato"), lwd=2)

dev.off()
cat("Saved plot_norm_channels.png\n")

options(bitmapType = "cairo")
.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs",
            "/home/labs/ginossar/talfis/Rlibs",
            .libPaths()))
library(glmnet)

base_dir  <- "/home/labs/ginossar/talfis/LiveImaging"
cache_dir <- file.path(base_dir, "cache", "combined")
fig_dir   <- file.path(base_dir, "figures", "combined")

# ── load data (identical setup) ───────────────────────────────────────────────
model_df <- readRDS(file.path(cache_dir, "model_df.rds"))
en_res   <- readRDS(file.path(cache_dir, "en_results.rds"))
meta     <- read.csv(file.path(base_dir, "Forecast", "cell_metadata.csv"),
                     stringsAsFactors = FALSE)

red_df <- model_df[is.finite(model_df$delay_green_to_red), ]
df     <- merge(red_df, meta[, c("Track.ID", "group")], by = "Track.ID")
df$group <- factor(df$group, levels = c("early", "medium", "late"))

en_feats <- rownames(en_res$coefs_nz)
en_feats <- intersect(en_feats, names(df))

X_raw <- as.matrix(df[, en_feats])
for (j in seq_len(ncol(X_raw))) {
  na_j <- is.na(X_raw[, j])
  if (any(na_j)) X_raw[na_j, j] <- median(X_raw[!na_j, j])
}
X_sc  <- scale(X_raw)
y_log <- log(df$delay_green_to_red)

# GMM cutoffs in log-minutes
cut_low  <- log(911)
cut_high <- log(2163)

# ── same 75/25 stratified split ───────────────────────────────────────────────
set.seed(42)
train_idx <- unlist(lapply(levels(df$group), function(g) {
  idx <- which(df$group == g)
  sample(idx, size = floor(0.75 * length(idx)))
}))
test_idx <- setdiff(seq_len(nrow(df)), train_idx)

X_tr <- X_sc[train_idx, ];  X_te <- X_sc[test_idx, ]
y_tr <- y_log[train_idx];   y_te <- y_log[test_idx]
grp_te <- df$group[test_idx]

# ── Stage 1: combined ElasticNet on all training cells ────────────────────────
cat("--- Stage 1: combined ElasticNet (router) ---\n")
set.seed(42)
cv_combined <- cv.glmnet(X_tr, y_tr, alpha = 0.5, nfolds = 10)
pred_tr_s1  <- as.numeric(predict(cv_combined, newx = X_tr, s = "lambda.min"))
pred_te_s1  <- as.numeric(predict(cv_combined, newx = X_te, s = "lambda.min"))

r2_s1 <- 1 - sum((y_te - pred_te_s1)^2) / sum((y_te - mean(y_te))^2)
r_s1  <- cor(y_te, pred_te_s1)
mae_s1 <- mean(abs(exp(y_te) - exp(pred_te_s1)))
cat(sprintf("Combined EN test:  R²=%.3f  r=%.3f  MAE=%.0f min\n", r2_s1, r_s1, mae_s1))

# route training cells by their Stage 1 *predictions* (not true labels)
route_tr <- cut(pred_tr_s1,
                breaks = c(-Inf, cut_low, cut_high, Inf),
                labels = c("early", "medium", "late"))
cat("\nTraining routing by Stage 1 prediction:\n")
print(table(route_tr))
cat("True group distribution:\n")
print(table(df$group[train_idx]))

# ── Stage 2: per-group ElasticNet on routed training cells ────────────────────
cat("\n--- Stage 2: per-group regressors (routed by Stage 1) ---\n")
reg_models <- list()
for (g in c("early", "medium", "late")) {
  idx_g <- which(route_tr == g)
  if (length(idx_g) < 5) {
    cat(sprintf("  %s: only %d cells — skipping, will use combined EN\n", g, length(idx_g)))
    reg_models[[g]] <- NULL
    next
  }
  set.seed(42)
  cv_g <- cv.glmnet(X_tr[idx_g, ], y_tr[idx_g],
                    alpha = 0.5, nfolds = min(10, length(idx_g)))
  r2_cv <- 1 - min(cv_g$cvm) / var(y_tr[idx_g])
  cat(sprintf("  %-7s  n=%d  CV R²≈%.3f\n", g, length(idx_g), r2_cv))
  reg_models[[g]] <- cv_g
}

# ── Test-set prediction: route by Stage 1, refine with Stage 2 ────────────────
route_te <- cut(pred_te_s1,
                breaks = c(-Inf, cut_low, cut_high, Inf),
                labels = c("early", "medium", "late"))
cat("\nTest routing distribution:\n"); print(table(route_te))

pred_te_s2 <- pred_te_s1   # default to Stage 1 prediction
for (g in c("early", "medium", "late")) {
  if (is.null(reg_models[[g]])) next
  m <- route_te == g
  if (!any(m)) next
  pred_te_s2[m] <- as.numeric(
    predict(reg_models[[g]], newx = X_te[m, , drop=FALSE], s = "lambda.min"))
}

# ── metrics ───────────────────────────────────────────────────────────────────
r2_s2  <- 1 - sum((y_te - pred_te_s2)^2) / sum((y_te - mean(y_te))^2)
r_s2   <- cor(y_te, pred_te_s2)
mae_s2 <- mean(abs(exp(y_te) - exp(pred_te_s2)))

cat("\n--- Final comparison ---\n")
cat(sprintf("Stage 1 only  (combined EN):   R²=%.3f  r=%.3f  MAE=%.0f min\n",
            r2_s1, r_s1, mae_s1))
cat(sprintf("Stage 1 + 2   (routed EN):     R²=%.3f  r=%.3f  MAE=%.0f min\n",
            r2_s2, r_s2, mae_s2))

cat("\nPer-group test metrics:\n")
for (g in c("early", "medium", "late")) {
  m <- grp_te == g
  for (lbl in c("S1", "S2")) {
    p   <- if (lbl == "S1") pred_te_s1[m] else pred_te_s2[m]
    r2g <- 1 - sum((y_te[m]-p)^2) / sum((y_te[m]-mean(y_te[m]))^2)
    rg  <- cor(y_te[m], p)
    cat(sprintf("  %-7s %s  n=%-3d  R²=%.3f  r=%.3f\n", g, lbl, sum(m), r2g, rg))
  }
}

# ── plot ───────────────────────────────────────────────────────────────────────
GROUP_COLS <- c(early = "#e67e22", medium = "#2980b9", late = "#27ae60")
actual_h   <- exp(y_te) / 60
s1_h       <- exp(pred_te_s1) / 60
s2_h       <- exp(pred_te_s2) / 60
all_lim    <- range(c(actual_h, s1_h, s2_h), finite = TRUE) * c(0.9, 1.1)

png(file.path(fig_dir, "twostage_en_v2.png"),
    width = 1400, height = 430, res = 115, type = "cairo")
par(mfrow = c(1, 4), mar = c(4.5, 4.5, 4, 1.5), oma = c(0, 0, 2.5, 0))

# left panel: combined EN (Stage 1 only), all cells
r2_all <- 1 - sum((y_te - pred_te_s1)^2) / sum((y_te - mean(y_te))^2)
r_all  <- cor(y_te, pred_te_s1)
plot(actual_h, s1_h,
     col  = adjustcolor("grey40", 0.5), pch = 16, cex = 0.75,
     xlim = all_lim, ylim = all_lim,
     xlab = "Actual delay (h)", ylab = "Predicted delay (h)",
     main = sprintf("Combined EN\n(all cells, R²=%.2f  r=%.2f)", r2_all, r_all))
abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
abline(lm(s1_h ~ actual_h), col = "grey20", lwd = 1.8)

# right 3 panels: Stage 1+2, one per true group
for (g in c("early", "medium", "late")) {
  mask <- grp_te == g
  x_g  <- actual_h[mask]
  y_g  <- s2_h[mask]
  r2_g <- 1 - sum((y_te[mask] - pred_te_s2[mask])^2) /
              sum((y_te[mask] - mean(y_te[mask]))^2)
  r_g  <- cor(y_te[mask], pred_te_s2[mask])
  col  <- GROUP_COLS[g]
  plot(x_g, y_g,
       col  = adjustcolor(col, 0.6), pch = 16, cex = 0.85,
       xlim = all_lim, ylim = all_lim,
       xlab = "Actual delay (h)", ylab = "Predicted delay (h)",
       main = sprintf("%s  (n=%d)\nR²=%.2f   r=%.2f", g, sum(mask), r2_g, r_g))
  abline(0, 1, lty = 2, col = "grey50", lwd = 1.2)
  if (sum(mask) > 2) abline(lm(y_g ~ x_g), col = col, lwd = 1.8)
}

mtext(sprintf("Two-stage EN v2: Stage1=combined EN router  →  Stage2=per-group EN  |  S1: R²=%.2f  S1+S2: R²=%.2f",
              r2_s1, r2_s2),
      outer = TRUE, cex = 0.82, font = 2, line = 0.8)
dev.off()
cat(sprintf("\nSaved figures/combined/twostage_en_v2.png\n"))

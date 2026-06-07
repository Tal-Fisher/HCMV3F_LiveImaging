.libPaths(c("/home/labs/ginossar/talfis/LiveImaging/Rlibs", .libPaths()))
suppressPackageStartupMessages(library(glmnet))

EXPORT  <- "/home/labs/ginossar/talfis/LiveImaging/cache/python_export"
RESULTS <- "/home/labs/ginossar/talfis/LiveImaging/results/elasticnet_productive"
SEED    <- 42

res <- readRDS(file.path(RESULTS, "en_productive_results.rds"))
cat("CV R2 (stored):", round(res$r2_cv, 4), "\n")
cat("Feature cols:  ", length(res$feat_cols), "\n")

# ── rebuild X and y the same way as script 19 ─────────────────────────────────
model_df <- read.csv(file.path(EXPORT, "model_df.csv"))
df       <- model_df[is.finite(model_df$delay_green_to_red), ]   # productive only
fc       <- res$feat_cols
fc       <- fc[vapply(fc, function(c) sum(is.finite(df[[c]])) >= 5, logical(1))]

X_raw <- as.matrix(df[, fc])
for (j in seq_len(ncol(X_raw))) {
  med_j <- median(X_raw[, j], na.rm = TRUE)
  X_raw[is.na(X_raw[, j]), j] <- if (is.finite(med_j)) med_j else 0
}
X <- scale(X_raw)
y <- df$delay_green_to_red
cat("n cells:", length(y), "  features:", ncol(X), "\n")

# ── fit on full data with lambda from inner CV ─────────────────────────────────
set.seed(SEED)
cv_en    <- cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
en_model <- glmnet(X, y, alpha = 0.5, lambda = cv_en$lambda.min)

y_hat  <- as.numeric(predict(en_model, newx = X, s = cv_en$lambda.min))
r2_ins <- 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)
r_ins  <- cor(y, y_hat)

cat("\nlambda.min:    ", round(cv_en$lambda.min, 2), "\n")
cat("In-sample R2:  ", round(r2_ins, 4), "\n")
cat("In-sample r:   ", round(r_ins,  4), "\n")
cat("CV R2:         ", round(res$r2_cv, 4), "\n")
cat("Overfit gap:   ", round(r2_ins - res$r2_cv, 4), "\n")

args <- commandArgs(trailingOnly = TRUE)

suppressPackageStartupMessages(library(nlme))

input_path <- if (length(args) >= 1) args[[1]] else file.path(
  "..", "MIRCID_dataset", "moa", "derived", "moa_table1_auc001.csv"
)
output_path <- if (length(args) >= 2) args[[2]] else file.path(
  "outputs", "moa", "moa_mixed_effects.csv"
)
diagnostic_path <- if (length(args) >= 3) args[[3]] else file.path(
  "outputs", "moa", "moa_mixed_effects_diagnostics.txt"
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(diagnostic_path), recursive = TRUE, showWarnings = FALSE)

d <- read.csv(input_path, check.names = FALSE)
d$algorithm <- factor(d$algorithm)
d$cell_line <- factor(d$cell_line)
d$setting <- factor(d$setting)
d$use_miRNA <- as.numeric(d$use_miRNA)
d$use_TF <- as.numeric(d$use_TF)

fit <- lme(
  fixed = value ~ use_miRNA * use_TF + algorithm + cell_line,
  random = ~ 1 | setting,
  data = d,
  method = "REML",
  control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)
)

coef_table <- summary(fit)$tTable
intervals_table <- intervals(fit, level = 0.95)$fixed
terms <- rownames(coef_table)
result <- data.frame(
  term = terms,
  estimate = coef_table[, "Value"],
  standard_error = coef_table[, "Std.Error"],
  degrees_freedom = coef_table[, "DF"],
  t_value = coef_table[, "t-value"],
  p_value = coef_table[, "p-value"],
  ci_low = intervals_table[terms, "lower"],
  ci_high = intervals_table[terms, "upper"],
  model = "AUC0.01 ~ use_miRNA * use_TF + algorithm + cell_line; random intercept: setting",
  statistical_unit = "algorithm-cell-line setting",
  stringsAsFactors = FALSE
)
write.csv(result, output_path, row.names = FALSE)

res <- residuals(fit, type = "normalized")
diag_lines <- c(
  capture.output(summary(fit)),
  "",
  sprintf("N observations: %d", nrow(d)),
  sprintf("N settings: %d", length(unique(d$setting))),
  sprintf("Normalized residual mean: %.8f", mean(res)),
  sprintf("Normalized residual SD: %.8f", sd(res)),
  sprintf("Max absolute normalized residual: %.8f", max(abs(res))),
  sprintf("Random-intercept SD: %.8f", as.numeric(VarCorr(fit)[1, "StdDev"])),
  sprintf("Residual SD: %.8f", as.numeric(VarCorr(fit)[2, "StdDev"]))
)
writeLines(diag_lines, diagnostic_path)
print(summary(fit))

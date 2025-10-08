# Takes a list of csv files with soft budget evaluation results and combines them into a single csv file.
# Every csv file must follow the format described in docs/dataframe_formats.md (soft budget case).
# Usage: Rscript combine_soft_budget_results.R file1.csv file2.csv ... > combined.csv

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript combine_soft_budget_results.R file1.csv file2.csv ... > combined.csv")
}

# Required columns for soft budget results
required_cols <- c(
  "method",
  "training_seed",
  "cost_parameter",
  "dataset",
  "sample",
  "features_chosen",
  "predicted_label_builtin",
  "predicted_label_external"
)

dfs <- lapply(args, function(f) {
  df <- read.csv(f, stringsAsFactors = FALSE)
  missing <- setdiff(required_cols, colnames(df))
  if (length(missing) > 0) {
    stop(sprintf("File %s is missing columns: %s", f, paste(missing, collapse = ", ")))
  }
  df
})

combined <- do.call(rbind, dfs)
write.csv(combined, file = "", row.names = FALSE)

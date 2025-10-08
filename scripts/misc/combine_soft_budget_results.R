# Takes a list of csv files with soft budget evaluation results and combines them into a single csv file.
# Every csv file must follow the format described in docs/dataframe_formats.md (soft budget case).
# Usage: Rscript combine_soft_budget_results.R file1.csv file2.csv combined.csv

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript combine_soft_budget_results.R file1.csv file2.csv ... combined.csv")
}

library(dplyr)

output_path <- args[length(args)]
input_files <- args[-length(args)]

# Required columns for soft budget results
required_cols <- c(
  "method",
  "training_seed",
  "cost_parameter",
  "dataset",
  "sample",
  "features_chosen",
  "predicted_label_builtin",
  "predicted_label_external",
  "true_label"
)

dfs <- lapply(input_files, function(f) {
  df <- read.csv(f, stringsAsFactors = FALSE)
  missing <- setdiff(required_cols, colnames(df))
  if (length(missing) > 0) {
    stop(sprintf("File %s is missing columns: %s", f, paste(missing, collapse = ", ")))
  }
  df
})

# Stack by rows
combined <- do.call(rbind, dfs)

# FIX: remove
write.csv(combined, file = paste(output_path, "~"), row.names = FALSE)

# We want to ensure that for a given (dataset, sample) combination, there
# is only one true_label. If this is not the case, something is wrong!
conflicting <- combined %>%
  group_by(dataset, sample) %>%
  summarise(n_labels = n_distinct(true_label), .groups = "drop") %>%
  filter(n_labels > 1)

if (nrow(conflicting) > 0) {
  print("Conflicting true_label values found for the following (dataset, sample) pairs:")
  print(conflicting)
  stop("There are observations where dataset and sample are the same but true_label is different!")
}

write.csv(combined, file = output_path, row.names = FALSE)

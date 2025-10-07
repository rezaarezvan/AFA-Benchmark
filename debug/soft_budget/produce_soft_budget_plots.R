#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript plot_accuracy_vs_features.R dataset.csv eval_results.csv output_plot.png")
}

library(ggplot2)
library(dplyr)
library(readr)

dataset_path <- args[1]
results_path <- args[2]
plot_path <- args[3]

# Read the data
dataset <- read_csv(dataset_path, col_types = cols(
  Dataset = col_factor(),
  Sample = col_integer(),
  `True label` = col_integer()
))
results <- read_csv(results_path, col_types = cols(
  Method = col_factor(),
  `Training seed` = col_integer(),
  `Cost parameter` = col_double(),
  Dataset = col_factor(),
  Sample = col_integer(),
  `Features chosen` = col_integer(),
  `Predicted label (builtin)` = col_integer(),
  `Predicted label (external)` = col_integer(),
))

# Join results with true labels
df <- results %>%
  left_join(dataset, by = c("Dataset", "Sample"))

# Summarize accuracy and features chosen
accuracy_by_method_df1 <- df %>%
  mutate(
    correct = (`Predicted label (external)` == `True label`)
  ) %>%
  group_by(Method, Dataset, `Cost parameter`, `Training seed`) %>%
  summarize(
    Accuracy = mean(correct),
    `Avg. features chosen` = mean(`Features chosen`),
    .groups = "drop_last"
  ) %>%
  summarize(
    `Avg. accuracy` = mean(Accuracy),
    `Sd. accuracy` = sd(Accuracy),
    `Mean avg. features chosen` = mean(`Avg. features chosen`),
    `Sd. avg. features chosen` = sd(`Avg. features chosen`),
    .groups = "drop_last"
  )

# Create the plot
p <- ggplot(accuracy_by_method_df1, aes(
  x = `Mean avg. features chosen`,
  y = `Avg. accuracy`,
  color = Method
)) +
  geom_point() +
  geom_line() +
  geom_errorbar(
    aes(
      ymin = `Avg. accuracy` - `Sd. accuracy`,
      ymax = `Avg. accuracy` + `Sd. accuracy`
    ),
    width = 0
  ) +
  geom_errorbarh(
    aes(
      xmin = `Mean avg. features chosen` - `Sd. avg. features chosen`,
      xmax = `Mean avg. features chosen` + `Sd. avg. features chosen`
    ),
    height = 0
  ) +
  facet_wrap(~Dataset, scales = "free_x") +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Accuracy vs Avg. Features Chosen",
    x = "Mean avg. features chosen",
    y = "Avg. accuracy"
  ) +
  theme_bw()

# Save the plot
ggsave(plot_path, p, width = 10, height = 6, dpi = 300)
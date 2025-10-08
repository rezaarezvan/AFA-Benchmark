#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript plot_accuracy_vs_features.R dataset.csv eval_results.csv output_plot.png")
}

library(ggplot2)
library(dplyr)
library(readr)

# Which datasets we want to plot F1 for instead of accuracy
f1_datasets <- c("physionet")

results_path <- args[1]
plot_path <- args[2]

results <- read_csv(results_path, col_types = cols(
    method = col_factor(),
    training_seed = col_integer(),
    cost_parameter = col_double(),
    dataset = col_factor(),
    features_chosen = col_integer(),
    predicted_label_builtin = col_integer(),
    predicted_label_external = col_integer()
))

# For now, only care about the external predictions
df <- results %>%
    rename(predicted_label = predicted_label_external) %>%
    select(-predicted_label_builtin)


# Summarize accuracy and features chosen
df_summarized <- df %>%
    mutate(
        correct = (predicted_label == true_label),
        tp = (predicted_label == 1 & true_label == 1),
        fp = (predicted_label == 1 & true_label == 0),
        tn = (predicted_label == 0 & true_label == 0),
        fn = (predicted_label == 0 & true_label == 1)
    ) %>%
    group_by(method, dataset, cost_parameter, training_seed) %>%
    summarize(
        accuracy = mean(correct),
        avg_features_chosen = mean(features_chosen),
        tp = sum(tp),
        fp = sum(fp),
        tn = sum(tn),
        fn = sum(fn),
        precision = ifelse(tp + fp == 0, 0, tp / (tp + fp)),
        recall = ifelse(tp + fn == 0, 0, tp / (tp + fn)),
        f1 = ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall)),
        .groups = "drop_last"
    )

# Some datasets use accuracy, others use F1
df_summarized <- df_summarized %>%
    mutate(
        metric_type = ifelse(dataset %in% f1_datasets, "f1", "accuracy"),
        metric_value = ifelse(metric_type == "f1", f1, accuracy)
    )

# Mean and sd over training seeds
df_summary <- df_summarized %>%
    group_by(method, dataset, cost_parameter, metric_type) %>%
    summarize(
        avg_metric = mean(metric_value),
        sd_metric = sd(metric_value),
        mean_avg_features_chosen = mean(avg_features_chosen),
        sd_avg_features_chosen = sd(avg_features_chosen),
        .groups = "drop"
    )

df_labels <- df_summary %>%
  group_by(dataset) %>%
  summarize(
    metric_label = ifelse(first(metric_type) == "f1", "F1", "Accuracy"),
    x = min(mean_avg_features_chosen),
    y = 1
  )

# Create the plot
p <- ggplot(df_summary, aes(
    x = mean_avg_features_chosen,
    y = avg_metric,
    color = method
)) +
    geom_point() +
    geom_line() +
    geom_errorbar(
        aes(
            ymin = avg_metric - sd_metric,
            ymax = avg_metric + sd_metric
        ),
        width = 0
    ) +
    geom_errorbarh(
        aes(
            xmin = mean_avg_features_chosen - sd_avg_features_chosen,
            xmax = mean_avg_features_chosen + sd_avg_features_chosen
        ),
        height = 0
    ) +
    geom_text(
      data = df_labels,
      aes(x = x, y = y, label = metric_label),
      inherit.aes = FALSE,
      hjust = 0, vjust = 1, fontface = "bold"
    ) +
    facet_wrap(~dataset, scales = "free_x") +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
        title = "Metric vs Avg. Features Chosen",
        x = "Avg. features chosen",
        y = "Metric",
    ) +
    theme_bw()

# Save the plot
ggsave(plot_path, p, width = 10, height = 6, dpi = 300)
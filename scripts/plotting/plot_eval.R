#!/usr/bin/env Rscript
options(device = "null")

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(yardstick)
  library(stringr)
})

read_csv_safe <- function(path) {
  read_csv(path, col_types = list(
    afa_method = col_factor(),
    classifier = col_factor(),
    dataset = col_factor(),
    selections_performed = col_integer(),
    features_observed = col_integer(),
    predicted_class = col_factor(),
    true_class = col_factor(),
    train_seed = col_integer(),
    eval_seed = col_integer(),
    hard_budget = col_integer(),
    soft_budget_param = col_number(),
  ))
}

generate_dummy_data <- function(n = 100) {
  set.seed(42)
  methods <- c("ODIN-MFRL", "EDDI", "DIME")
  classifiers <- c(NA, "fully_connected")
  datasets <- c("cube", "MNIST")
  train_seeds <- 1:2
  eval_seeds <- 1:2
  hard_budgets <- c(NA_integer_, 5, 10, 15)
  soft_budget_params <- c(NA, 0.1, 1.0)
  n_classes <- 5
  n_features <- 15



  df <- tibble(
    afa_method = factor(sample(methods, n, replace = TRUE)),
    classifier = factor(sample(classifiers, n, replace = TRUE)),
    dataset = factor(sample(datasets, n, replace = TRUE)),
    selections_performed = sample(1:n_features, n, replace = TRUE),
    features_observed = selections_performed,
    predicted_class = factor(sample(0:(n_classes - 1), n, replace = TRUE)),
    true_class = factor(sample(0:(n_classes - 1), n, replace = TRUE)),
    train_seed = sample(train_seeds, n, replace = TRUE),
    eval_seed = sample(eval_seeds, n, replace = TRUE),
    hard_budget = as.integer(sample(hard_budgets, n, replace = TRUE)),
    soft_budget_param = sample(soft_budget_params, n, replace = TRUE),
  )

  return(df)
}


# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  print("Proceeding with plotting using dummy data...")
  df <- generate_dummy_data(n = 10000)
  output_path <- NA
} else if (length(args) == 1) {
  print("Proceeding with plotting using dummy data...")
  df <- generate_dummy_data(n = 10000)
  output_path <- args[1]
} else if (length(args) == 2) {
  df <- read_csv_safe(args[1])
  output_path <- args[2]
} else {
  stop("Usage: Rscript plot.R [eval_results.csv] output_folder")
}

class_metrics <- metric_set(
  accuracy,
  kap,
  f_meas
)

# First type of plot: hard budget
df_hard_budget <- suppressWarnings(
  df %>%
    # We filter results to only include methods that were not trained/evaluated with a soft budget parameter.
    filter(
      !is.na(hard_budget),
      is.na(soft_budget_param),
    ) %>%
    group_by(
      afa_method,
      classifier,
      dataset,
    ) %>%
    # We don't care about soft budget param (which we know is NA anyway)
    select(
      -soft_budget_param,
    ) %>%
    # We only care about the prediction made once we reach the evaluation hard budget
    filter(
      features_observed == hard_budget
    ) %>%
    select(
      -features_observed,
      -selections_performed
    ) %>%
    # Also group over train_seed, eval_seed and hard_budget before calculating metrics.
    # hard_budget will become the x-axis
    group_by(
      train_seed,
      eval_seed,
      hard_budget,
      .add = TRUE
    ) %>%
    # Now we can calculate metrics
    class_metrics(truth = true_class, estimate = predicted_class) %>%
    # tibble is automatically ungrouped after calculating metrics
    # Calculate mean and std of metrics
    group_by(
      afa_method,
      classifier,
      dataset,
      hard_budget,
      .metric,
      .estimator
      # leaves train_seed, eval_seed and .estimate
    ) %>%
    summarize(
      estimate_mean = mean(.estimate),
      estimate_sd = sd(.estimate),
      .groups = "drop"
    )
)

# Example: plot with builtin classifier accuracy
hard_budget_plot <- df_hard_budget %>%
  filter(
    .metric == "accuracy",
    is.na(classifier)
  ) %>%
  ggplot(aes(x = hard_budget, y = estimate_mean, color = afa_method, fill = afa_method)) +
  geom_line() +
  geom_ribbon(aes(ymin = estimate_mean - estimate_sd, ymax = estimate_mean + estimate_sd), alpha = 0.2, linetype = "blank") +
  facet_wrap(vars(dataset))

if (!is.na(output_path)) {
  ggsave(str_c(output_path, "/hard_budget.png"), plot = hard_budget_plot, create.dir = TRUE)
} else {
  print(hard_budget_plot)
}

# Second type of plot: soft budget
df_soft_budget <- suppressWarnings(
  df %>%
    # We filter results to only include methods that were trained or evaluated with a soft budget parameter.
    filter(
      is.na(hard_budget),
      !is.na(soft_budget_param)
    ) %>%
    group_by(
      afa_method,
      classifier,
      dataset,
    ) %>%
    # We don't care about hard budget (which we know is NA anyway)
    select(
      -hard_budget
    ) %>%
    # This time we care about predictions at every step, but we focus on the number of features observed instead of selections performed
    select(
      -selections_performed
    ) %>%
    # Also group over (train_seed, eval_seed, train_soft_budget_param, eval_soft_budget_param, features_observed) before calculating metrics.
    # eval_hard_budget will become the x-axis
    group_by(
      train_seed,
      eval_seed,
      soft_budget_param,
      features_observed,
      .add = TRUE
    ) %>%
    # Now we can calculate metrics
    class_metrics(truth = true_class, estimate = predicted_class) %>%
    # tibble is automatically ungrouped after calculating metrics
    # Calculate mean and std of metrics
    group_by(
      afa_method,
      classifier,
      dataset,
      features_observed,
      .metric,
      .estimator
      # leaves train_seed, eval_seed, soft_budget_param and .estimate
    ) %>%
    summarize(
      estimate_mean = mean(.estimate),
      estimate_sd = sd(.estimate),
      .groups = "drop"
    )
)

# Example: plot with builtin classifier accuracy
soft_budget_plot <- df_soft_budget %>%
  filter(
    .metric == "accuracy",
    is.na(classifier)
  ) %>%
  ggplot(aes(x = features_observed, y = estimate_mean, color = afa_method, fill = afa_method)) +
  geom_line() +
  geom_ribbon(aes(ymin = estimate_mean - estimate_sd, ymax = estimate_mean + estimate_sd), alpha = 0.2, linetype = "blank") +
  facet_wrap(vars(dataset))

if (!is.na(output_path)) {
  ggsave(str_c(output_path, "/soft_budget.png"), plot = soft_budget_plot, create.dir = TRUE)
} else {
  print(soft_budget_plot)
}

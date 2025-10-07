# Load required libraries
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)

# Read the data
dataset <- read_csv("dataset.csv", col_types = cols(
  Dataset = col_factor(),
  `Sample` = col_integer(),
  `True label` = col_integer()
))
results <- read_csv("results.csv", col_types = cols(
  Method = col_factor(),
  `Training seed` = col_integer(),
  `Cost parameter` = col_double(),
  Dataset = col_factor(),
  Sample = col_integer(),
  `Features chosen` = col_integer(),
  `Predicted label (builtin)` = col_integer(),
  `Predicted label (external)` = col_integer(),
))

# ---- Join results with true labels ----
df <- results %>%
    left_join(dataset, by = c("Dataset", "Sample"))

# First alternative
accuracy_by_method_df1 = df %>%
  mutate(
    correct = (`Predicted label (external)` == `True label`)
  ) %>%
  group_by(Method, Dataset, `Cost parameter`, `Training seed`) %>%
  summarize( # over samples in dataset
    Accuracy = mean(correct),
    `Avg. features chosen` = mean(`Features chosen`),
  .groups = "drop_last") %>% 
  summarize(
    `Avg. accuracy` = mean(Accuracy),
    `Sd. accuracy` = sd(Accuracy),
    `Mean avg. features chosen` = mean(`Avg. features chosen`),
    `Sd. avg. features chosen` = sd(`Avg. features chosen`)
  , .groups = "drop_last")


ggplot(accuracy_by_method_df1, aes(
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
  ylim(0, 1) +
  labs(
    title = "Accuracy vs Avg. Features Chosen",
    x = "Mean avg. features chosen",
    y = "Avg. accuracy"
  ) +
  theme_bw()

# Second alternative
accuracy_by_method_df2 = df %>%
  mutate(
    correct = (`Predicted label (external)` == `True label`)
  ) %>%
  group_by(Method, Dataset, `Cost parameter`) %>%
  summarize( # over training seed and samples in dataset
    `Avg. features chosen` = mean(`Features chosen`),
    `Sd. features chosen` = sd(`Features chosen`),
    `Avg. accuracy` = mean(correct),
    `Sd. accuracy` = sd(correct),
  .groups = "drop_last")

ggplot(accuracy_by_method_df2, aes(
  x = `Avg. features chosen`,
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
      xmin = `Avg. features chosen` - `Sd. features chosen`,
      xmax = `Avg. features chosen` + `Sd. features chosen`
    ),
    height = 0
  ) +
  facet_wrap(~Dataset) +
  coord_cartesian(ylim = c(0, 1)) +
  # ylim(0, 1) +
  labs(
    title = "Accuracy vs avg. features Chosen",
    x = "Avg. features chosen",
    y = "Avg. accuracy"
  ) +
  theme_bw()

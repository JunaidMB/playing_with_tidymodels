library(tidyverse)
library(tidymodels)
library(xgboost)
library(vip)

# Load Data
vb_matches <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-19/vb_matches.csv', guess_max = 76000)

# Sort data and omit NAs
vb_parsed <- vb_matches %>%
  transmute(
    circuit,
    gender,
    year,
    w_attacks = w_p1_tot_attacks + w_p2_tot_attacks,
    w_kills = w_p1_tot_kills + w_p2_tot_kills,
    w_errors = w_p1_tot_errors + w_p2_tot_errors,
    w_aces = w_p1_tot_aces + w_p2_tot_aces,
    w_serve_errors = w_p1_tot_serve_errors + w_p2_tot_serve_errors,
    w_blocks = w_p1_tot_blocks + w_p2_tot_blocks,
    w_digs = w_p1_tot_digs + w_p2_tot_digs,
    l_attacks = l_p1_tot_attacks + l_p2_tot_attacks,
    l_kills = l_p1_tot_kills + l_p2_tot_kills,
    l_errors = l_p1_tot_errors + l_p2_tot_errors,
    l_aces = l_p1_tot_aces + l_p2_tot_aces,
    l_serve_errors = l_p1_tot_serve_errors + l_p2_tot_serve_errors,
    l_blocks = l_p1_tot_blocks + l_p2_tot_blocks,
    l_digs = l_p1_tot_digs + l_p2_tot_digs
  ) %>%
  na.omit()

# Create separate dataframes for winners and losers and bind them together
winners <- vb_parsed %>%
  dplyr::select(circuit, gender, year,
         w_attacks:w_digs) %>%
  rename_with(~ str_remove_all(., "w_"), w_attacks:w_digs) %>%
  mutate(win = "win")

losers <- vb_parsed %>%
  dplyr::select(circuit, gender, year,
         l_attacks:l_digs) %>%
  rename_with(~ str_remove_all(., "l_"), l_attacks:l_digs) %>%
  mutate(win = "lose")

vb_df <- bind_rows(winners, losers) %>%
  mutate_if(is.character, factor)

# Plot to explore data
vb_df %>%
  pivot_longer(attacks:digs, names_to = "stat", values_to = "value") %>%
  ggplot(aes(gender, value, fill = win, color = win)) +
  geom_boxplot(alpha = 0.4) +
  facet_wrap(~stat, scales = "free_y", nrow = 2) +
  labs(y = NULL, color = NULL, fill = NULL)

# Data splitting
set.seed(123)
vb_split <- initial_split(vb_df, strata = win)

vb_train <- training(vb_split)
vb_test <- testing(vb_split)

# Setup XGBoost model and tune hyperparameters
xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_spec

## Setup a hyperparameter grid
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), vb_train),
  learn_rate(),
  size = 30
)

# Make a workflow with the model
## Since we have no recipe, we can use add formula
xgb_wf <- workflow() %>% 
  add_formula(win ~ .) %>% 
  add_model(xgb_spec)

xgb_wf

# Create cross-validation resamples for tuning the model
set.seed(123)
vb_folds <- vfold_cv(vb_train, strata = win)

vb_folds

# Tune the model - Even with parallelisation, it takes a long time!
doParallel::registerDoParallel()

set.seed(234)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)

# View the metrics for the models
collect_metrics(xgb_res)

# Finalise the workflow with the best of the model
final_xgb <- xgb_wf %>% 
  finalize_workflow(select_best(xgb_res, "roc_auc"))

# Variable Importance
final_xgb %>% 
  fit(data = vb_train) %>% 
  pull_workflow_fit() %>% 
  vip(geom = "point")

# Fit the model on the training data and check on the test data
final_res <- final_xgb %>% 
  last_fit(vb_split)

## Metrics on test set
final_res %>% 
  collect_metrics()

## Predictions on test set
final_res %>% 
  collect_predictions() %>%
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(vb_test) 

## Confusion Matrix on test set
final_res %>% 
  collect_predictions() %>% 
  conf_mat(win, .pred_class)

# Plot the ROC curve
final_res %>%
  collect_predictions() %>%
  roc_curve(win, .pred_win) %>%
  ggplot(aes(x = specificity, y = 1 - sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )


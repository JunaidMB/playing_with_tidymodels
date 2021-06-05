library(tidyverse)
library(tidymodels)
library(workflowsets)
library(modeldata)
library(parallel)
library(doParallel)
library(xgboost)
library(vip)


# Read Data ----
data(cells, package = "modeldata")
head(cells)

# Data Splitting ----
## Training and Test Split
set.seed(123)
cell_split <- initial_split(cells)

cell_train <- training(cell_split)
cell_test <- testing(cell_split)

## Cross Validation Folds
cell_folds <- vfold_cv(cell_train)

# Define Recipes ----
base_recipe <- recipe(class ~ ., data = cell_train) %>% 
  step_rm(case)

normalize_recipe <- base_recipe %>% 
  step_normalize(all_numeric_predictors())

# Define Models ----
# 2 Models: Random Forest and XGBoost
rf_spec <- rand_forest(trees = 100, mtry = tune()) %>% 
  set_engine("ranger", importance = "permutation") %>% 
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = 100,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

# Define Workflowset ----
cells_models <- workflow_set(
  preproc = list(unnormalized = base_recipe,
                 normalized = normalize_recipe),
  models = list(random_forest = rf_spec,
                xgboost = xgb_spec),
  cross = TRUE
)

# Train and Tune Models ----
## Some recipes have tuning parameters we want to have more control over than default. This is done via the parameters and update options.
rf_param <- 
  rf_spec %>% 
  parameters() %>% 
  update(mtry = mtry() %>% finalize(cells %>% dplyr::select(-case, -class)))

xgb_param <- 
  xgb_spec %>% 
  parameters() %>% 
  update(mtry = mtry() %>% finalize(cells %>% dplyr::select(-case, -class)))

## Add the parameters as options to workflowset, specify the id so we know which model's parameters are being changed
cells_models <- cells_models %>% 
  option_add(param_info = rf_param, id = "unnormalized_random_forest") %>% 
  option_add(param_info = rf_param, id = "normalized_random_forest") %>% 
  option_add(param_info = xgb_param, id = "unnormalized_xgboost") %>% 
  option_add(param_info = xgb_param, id = "normalized_xgboost")

## Parallelised tuning - use a random grid
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1)
doParallel::registerDoParallel(cl) # Register Backend

set.seed(123)
cells_models <- cells_models %>%
  workflow_map(resamples = cell_folds,
               grid = 5,
               control = control_grid(save_pred = TRUE),
               metrics = metric_set(roc_auc, accuracy),
               verbose = TRUE)

# Inspect Results ----
## Inspect CV Metrics
cells_models %>% 
  collect_metrics()

## Inspect CV predictions
cells_models %>% 
  collect_predictions()

## Ranked results
cells_models %>% 
  rank_results(rank_metric = "roc_auc")

## Plot results
autoplot(cells_models, metric = "roc_auc")

# Select Best Model ----
best_results <- cells_models %>% 
  pull_workflow_set_result("unnormalized_random_forest") %>% 
  select_best(metric = "roc_auc")

# Finalize Model ----
rf_workflow <- cells_models %>% 
  pull_workflow("unnormalized_random_forest")

rf_workflow_fit <- rf_workflow %>% 
  finalize_workflow(best_results) %>% 
  fit(data = cell_train)

# Evaluate on Test Set ----
## Make a table of Model Predictions
test_pred <- rf_workflow_fit %>% 
  predict(cell_test, type = "prob") %>% 
  bind_cols(cell_test)

## Compute Confusion Matrix on Test Set
test_conf_mat <- rf_workflow_fit %>% 
  predict(cell_test) %>% 
  bind_cols(cell_test) %>% 
  conf_mat(class, .pred_class)

# Variable Importance ----
## Computed on baked training data - the model is fit to all the training data once again
set.seed(123)

rf_workflow %>% 
  finalize_workflow(best_results) %>% 
  fit(data = cells) %>%
  pull_workflow_fit() %>% 
  vip(geom = "point", all_permutations = TRUE)




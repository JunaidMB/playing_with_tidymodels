library(tidyverse)
library(tidymodels)
library(themis)
library(vip)
library(finetune)
library(doParallel)
library(butcher)
library(here)

# Blog Link: https://juliasilge.com/blog/mario-kart/
# Predict if a Mario Kart World Record was achieved using a shortcut or not. Will use a Decision Tree Model for prediction.


# Load Data ----
records <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-25/records.csv")

# Split Data ----
set.seed(123)
records_split <- initial_split(records)

records_train <- training(records_split)
records_test <- testing(records_split)

# Create Recipe ----
## Define Recipe
records_recipe <- recipe(shortcut ~ ., data = records_train) %>% 
  step_rm(c("player", "system_played", "time_period", "record_duration"))

## Prepare and Bake recipe on training and test data
records_recipe_prep <- prep(records_recipe, training = records_train)

## Bake Recipe for Training and Testing sets
records_train_bake <- bake(records_recipe_prep, new_data = NULL)
records_test_bake <- bake(records_recipe_prep, new_data = records_test)

# Define Model ----
## Initialise model with tunable hyperparameters
tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# Define a workflow to connect Recipe and Model ----
tree_wf <- workflow() %>% 
  add_recipe(records_recipe) %>% 
  add_model(tree_spec)

# Train and Tune Model ----
## Define a grid for hyperparameters to vary over
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 7)

## Make Cross Validation Folds - Use bootstrapping due to low sample size
set.seed(123)

records_folds <- bootstraps(records_train, strata = shortcut)

## Tune model with parallel processing
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1)
doParallel::registerDoParallel(cl)

set.seed(123)
tree_tuned <- tree_wf %>% 
  tune_grid(resamples = records_folds,
            grid = tree_grid, 
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, accuracy))

tree_tuned %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  arrange(desc(mean))

# Finalise Model ----
## Update workflow with the best model
tree_best_model <- tree_tuned %>% 
  select_best(metric = "roc_auc")

final_tree_wf <- tree_wf %>% 
  finalize_workflow(tree_best_model)

## Fit the final model to all the training data
final_tree_model <- final_tree_wf %>% 
  fit(data = records_train)

## Predict 
final_tree_model %>% 
  predict(records_train, type = "prob")

# Fit Model to test data ----
## Fit finalised model on all training data and evaluate on test data
tree_fit_final <- final_tree_model %>% 
  last_fit(records_split)

# Model Evaluation ----
## Metrics on test set
tree_fit_final %>% 
  collect_metrics()

## Predictions on test set
tree_fit_final %>% 
  collect_predictions() %>% 
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(records_test)

## Confusion Matrix on test set
tree_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(shortcut, .pred_class)

# Variable Importance ----
## Fit on baked training data
set.seed(123)

decision_tree(
  cost_complexity = tree_best_model$cost_complexity,
  tree_depth = tree_best_model$tree_depth
) %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>% 
  fit(shortcut ~ ., data = records_train_bake) %>% 
  vip(geom = "point", all_permutations = TRUE)


# Save Model and Metrics ----
tree_wf_model <- tree_fit_final$.workflow[[1]]

#predict(tree_wf_model, records_test[5,]) # Feed test data and it generates predictions

## Save metrics
collect_metrics(tree_fit_final) %>% 
  write_csv(here::here("saved_models", "tree_model_metrics_mariokart.csv"))

## Save Model
saveRDS(tree_wf_model, here::here("saved_models", "tree_wf_model_mariokart.rds"))

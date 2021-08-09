library(tidyverse)
library(devtools)
devtools::install_github("tidymodels/dials", force = TRUE)
library(tidymodels)
library(xgboost)
library(doParallel)
library(finetune)
library(lubridate)
library(vroom)
library(vip)

# Blog Link: https://juliasilge.com/blog/shelter-animals/
# Data Link: https://www.kaggle.com/c/sliced-s01e09-playoffs-1/data?select=train.csv

# Load Data ----
train_raw <- vroom(file = "./data/animal_shelter_train.csv")
train_cleaned <- train_raw %>% 
  mutate(age_upon_outcome = as.period(as.Date(datetime) - date_of_birth),
         age_upon_outcome = time_length(age_upon_outcome, unit = "weeks"))

# Split Data ----
set.seed(123)
shelter_split <- initial_split(train_cleaned, strata = outcome_type)

shelter_train <- training(shelter_split)
shelter_test <- testing(shelter_split)

# Create Recipe ----
shelter_rec <- recipe(outcome_type ~ age_upon_outcome + animal_type + datetime + sex + spay_neuter, data = shelter_train) %>% 
  step_date(datetime, features = c("year", "week", "dow"), keep_original_cols = FALSE) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_zv(all_predictors())

## Prepare and Bake recipe on training and test data
shelter_rec_prep <- prep(shelter_rec, training = shelter_train)

shelter_train_bake <- bake(shelter_rec_prep, new_data = NULL)
shelter_test_bake <- bake(shelter_rec_prep, new_data = shelter_test)

# Define Model ----
stopping_spec <- boost_tree(
  trees = 500,
  mtry = tune(),
  learn_rate = tune(),
  stop_iter = tune()
) %>% 
  set_engine("xgboost", validation = 0.2) %>% 
  set_mode("classification")

# Define Workflow to connect Recipe and Model ----
early_stop_wf <- workflow() %>% 
  add_recipe(shelter_rec) %>% 
  add_model(stopping_spec)

# Train and Tune Model ----
## Define a grid for hyperparameters to vary over
set.seed(123)

stopping_grid <- grid_latin_hypercube(
  mtry(range = c(5L, 20L)),
  learn_rate(range = c(-5, -1)),
  stop_iter(range = c(10L, 50L)),
  size = 10
)

## Make Cross Validation Folds
set.seed(234)
shelter_folds <- vfold_cv(shelter_train, strata = outcome_type)

## Tune Model using Parallel Processing
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1) 
doParallel::registerDoParallel(cl) # Register Backend

set.seed(345)
stopping_rs <- tune_grid(
  early_stop_wf,
  shelter_folds,
  grid = stopping_grid,
  metrics = metric_set(accuracy, roc_auc, mn_log_loss)
)

## View performance metrics across all hyperparameter permutations
stopping_rs %>% 
  collect_metrics()

# Finalise Model ----
## Update workflow with the best model found
xgb_best_model <- stopping_rs %>% 
  select_best(metric = "mn_log_loss")

final_xgb_workflow <- early_stop_wf %>% 
  finalize_workflow(xgb_best_model)

## Fit the final model to all the training data
final_xgb_model <- final_xgb_workflow %>% 
  fit(data = shelter_train)

# Fit model to test data ----
## Fit finalised model on all training data and evaluate on test data
xgb_fit_final <- final_xgb_model %>% 
  last_fit(shelter_split)

# Model Evaluation ----
## Metrics on test set
xgb_fit_final %>% 
  collect_metrics()

## Predictions on test set
xgb_fit_final %>% 
  collect_predictions() %>%
  dplyr::select(starts_with(".pred")) %>%
  bind_cols(shelter_test)

## Confusion Matrix on test set
xgb_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(outcome_type, .pred_class)

# Variable Importance ----
xgb_fit_final %>% 
  extract_workflow() %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 15, geom = "point", all_permutations = TRUE)

# Save Model and Metrics ----
## Extract final fitted workflow
xgb_wf_model <- xgb_fit_final$.workflow[[1]]

## Save Model 
saveRDS(xgb_wf_model, file = "shelter_xgb_saved_model.rds")

library(AmesHousing)
library(janitor)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(finetune)
library(vip)
library(knitr)
library(remotes)
remotes::install_github("curso-r/treesnip")
library(treesnip)
library(devtools)
devtools::install_github("curso-r/rightgbm")
library(rightgbm)
library(lightgbm)


# Load Data ----
set.seed(123)
ames_data <- make_ames() %>% 
  janitor::clean_names()

# Split Data ----
ames_split <- initial_split(ames_data, prop = 0.8, strata = sale_price)

ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# Create Recipe ----
ames_recipe <- recipe(sale_price ~ ., data = ames_train) %>% 
  step_other(all_nominal_predictors(), threshold = 0.01) %>% 
  step_nzv(all_nominal_predictors())

# Define Model -----
lightgbm_spec <- boost_tree(
  trees = 1000,
  min_n = tune(),
  learn_rate = tune(),
  tree_depth = tune()) %>% 
  set_engine("lightgbm", loss_function = "reg:squarederror", nthread = 6) %>% 
  set_mode("regression")


# Define Workflow to connect Recipe and Model ----
lightgbm_workflow <- workflow() %>% 
  add_recipe(ames_recipe) %>% 
  add_model(lightgbm_spec)

# Train and Tune Model ----
## Define a random grid for hyperparameters to vary over
set.seed(123)

lightgbm_grid <- grid_max_entropy(
  min_n(),
  learn_rate(),
  tree_depth(),
  size = 10
)

## Make Cross Validation Folds
set.seed(123)
ames_folds <- vfold_cv(data = ames_train, v = 5)

## Tune Model using Parallel Processing
cores <- parallel::detectCores(logical = FALSE)
cl <- makeForkCluster(cores - 1)  
doParallel::registerDoParallel(cl) # Register Backend

set.seed(123)
lightgbm_tuned <- lightgbm_workflow %>% 
  tune_grid(resamples = ames_folds, 
            grid = lightgbm_grid,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(rmse, rsq, mae))


## View performance metrics across all hyperparameter permutations
lightgbm_tuned %>% 
  collect_metrics()

# Plot for tuning parameter performance ----
## RMSE
lightgbm_tuned %>% 
  show_best(metric = "rmse", n = 10) %>% 
  tidyr::pivot_longer(min_n:learn_rate, names_to="variable",values_to="value" ) %>% 
  ggplot(mapping = aes(value, mean)) +
  geom_line(alpha=1/2)+
  geom_point()+
  facet_wrap(~variable, scales = "free")+
  ggtitle("Best parameters for RMSE")

## MAE
lightgbm_tuned %>% 
  show_best(metric = "mae", n = 10) %>% 
  tidyr::pivot_longer(min_n:learn_rate, names_to="variable", values_to="value" ) %>% 
  ggplot(mapping = aes(value,mean)) +
  geom_line(alpha=1/2)+
  geom_point()+
  facet_wrap(~variable, scales = "free")+
  ggtitle("Best parameters for MAE")

# Finalise Model ----
## Update workflow with the best model found
lightgbm_best_model <- lightgbm_tuned %>% 
  select_best(metric = "rmse")

final_lightgbm_workflow <- lightgbm_workflow %>% 
  finalize_workflow(lightgbm_best_model)

## Fit the final model to all the training data
final_lightgbm_model <- final_lightgbm_workflow %>% 
  fit(data = ames_train)

# Fit model to test data ----
lightgbm_fit_final <- final_lightgbm_model %>% 
  last_fit(ames_split)

# Model Evaluation ----
## Metrics on test set
lightgbm_fit_final %>% 
  collect_metrics()

## Predictions on test set
lightgbm_fit_final %>% 
  collect_predictions() %>% 
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(ames_test)

## Residuals on training set
final_lightgbm_model %>% 
  predict(new_data = ames_train) %>% 
  bind_cols(ames_train) %>% 
  mutate(residuals = sale_price - .pred) %>% 
  ggplot(mapping = aes(x = .pred, y = residuals)) + 
  geom_point()

## Residuals on test set
lightgbm_fit_final %>% 
  collect_predictions() %>% 
  mutate(residuals = sale_price - .pred) %>% 
  ggplot(mapping = aes(x = .pred, y = residuals)) + 
  geom_point()

# Yardstick prediction metrics ----
## Training set
final_lightgbm_model %>% 
  predict(new_data = ames_train) %>% 
  bind_cols(ames_train) %>% 
  yardstick::metrics(sale_price, .pred) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ",")) %>%
  knitr::kable()

## Test set
lightgbm_fit_final %>% 
  collect_predictions() %>% 
  yardstick::metrics(sale_price, .pred) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ",")) %>%
  knitr::kable()


# Stop Cluster
parallel::stopCluster(cl)

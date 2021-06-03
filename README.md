# Repository of Tidymodels template scripts in R

I will be using this repository to store scripts in R that demonstrate features using the tidymodels framework. Since the tidymodels framework has multiple variations of what can be done, this repository is where I will keep completed scripts so I can refer to them in future when building models using the tidymodels framework. 

Although there is no one archetype of the tidymodels workflow, there are some repeating patterns worth recording in the README for a quick tidymodels script. Those patterns are recorded in a code block in this README.

## Tidymodels - General Workflow (Single Workflow)

Note: This is pseudocode, don't try to run this directly! The example model is xgboost.

```
library(tidyverse)
library(tidymodels)
library(themis)
library(vip)
library(finetune)
library(doParallel)
library(xgboost)

# Load Data - From the internet/CSV/Database ----
df <- read_csv("{some filepath}")

# Split Data ----
set.seed(123)
df_split <- initial_split(df)

df_train <- training(df_split)
df_test <- testing(df_split)

# Create Recipe ----
## Define Recipe to be applied to the dataset. Note: Y denotes the outcome variable
df_recipe <- recipe(Y ~ ., data = df_train) %>%
	update_role(..., new_role = ...) %>%
	step_foobar(...)

## Prepare and Bake recipe on training and test data
df_recipe_prep <- prep(df_recipe, training = df_train)

df_train_bake <- bake(df_recipe_prep, new_data = NULL)
df_test_bake <- bake(df_recipe_prep, new_data = df_test)

# Define Model ----
## Initialise model with tunable hyperparameters
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

# Define Workflow to connect Recipe and Model ----
xgb_workflow <- workflow() %>%
	add_recipe(df_recipe) %>% 
	add_model(xgb_spec)

# Train and Tune Model ----
## Define a random grid for hyperparameters to vary over
set.seed(123)

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), vb_train),
  learn_rate(),
  size = 10
)

## Make Cross Validation Folds
set.seed(123)
folds <- vfold_cv(data = df_train, v = 10)

## Tune Model using Parallel Processing
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1) 
doParallel::registerDoParallel(cl) # Register Backend

set.seed(123)
xgb_tuned <- xgb_workflow %>% 
	tune_grid(resamples = folds,
			  grid = xgb_grid,
			  control = control_grid(save_pred = TRUE),
			  metrics = metric_set(roc_auc, accuracy))

## View performance metrics across all hyperparameter permutations
xgb_tuned %>% 
  collect_metrics()

# Finalise Model ----
## Update workflow with the best model found
xgb_best_model <- xgb_tuned %>% 
	select_best(metric = "roc_auc")

final_xgb_workflow <- xgb_workflow %>%
	finalize_workflow(xgb_best_model)

## Fit the final model to all the training data
final_xgb_model <- final_xgb_workflow %>% 
	fit(data = df_train)

# Fit model to test data ----
## Fit finalised model on all training data and evaluate on test data
xgb_fit_final <- final_xgb_model %>% 
	last_fit(df_split)

# Model Evaluation ----
## Metrics on test set
xgb_fit_final %>%
	collect_metrics()

## Predictions on test set
xgb_fit_final %>% 
	collect_predictions() %>%
	dplyr::select(starts_with(".pred")) %>%
	bind_cols(df_test)

## Confusion Matrix on test set
xgb_fit_final %>% 
	collect_predictions() %>% 
	conf_mat(Y, .pred_class)

# Variable Importance
## Computed on baked training data - the model is fit to all the training data once again

set.seed(123)
boost_tree(
  trees = 1000,
  tree_depth = xgb_best_model$tree_depth,
  min_n = xgb_best_model$min_n,
  loss_reduction = xgb_best_model$loss_reduction,
  sample_size = xgb_best_model$sample_size,
  mtry = xgb_best_model$mtry,
  learn_rate = xgb_best_model$learn_rate
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  fit(red_density_state ~ .,
      data = df_train_bake %>% dplyr::select({ID columns}) # Remove ID columns
  ) %>% 
  vip(geom = "point", all_permutations = TRUE)


```

## Useful Links
* [TidyModels Docs](https://www.tidymodels.org/start/)
* [Tidymodels Book](https://www.tmwr.org/)
* [Julia Silge Blog](https://juliasilge.com/blog/)
* [Tidyverse Blog](https://www.tidyverse.org/blog/)

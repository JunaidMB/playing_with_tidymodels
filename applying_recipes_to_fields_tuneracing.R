library(tidymodels)
library(randomForest)
library(discrim)
library(klaR)
library(tidyverse)
library(ROSE)
library(themis)
library(vip)
library(parallel)
library(doParallel)
library(finetune)
library(ranger)

# Create Fake Data ----
## Generate column values
set.seed(123)

fields <- sort(rep.int(c(1:5), 100))
quads <- rep(c(1:5), 100)
values <- rnorm(n = 500, mean = 3)
crop <- sample(x = c("linseed", "barley", "olive", "potato", "cauliflower"), size = 500, replace = TRUE, prob = c(0.5, 0.2, 0.05, 0.20, 0.05))
red_density_state <- sample(x = c(1,2,3), size = 500, replace = TRUE, prob = c(0.5, 0.4, 0.1))
junk_col1 <- rnorm(n = 500, mean = 50)
junk_col2 <- rnorm(n = 500, mean = 40)
junk_col3 <- rnorm(n = 500, mean = 30)

## Join columns into a tibble
df <- tibble(fields, quads, values, crop, red_density_state, junk_col1, junk_col2, junk_col3)

## Add some junk rows with NA and NaN values
df <- bind_rows(df, tibble(fields = c(5, 5), quads = c(6, 7), values = c(NaN, 3.4), crop = c("olive", NA), red_density_state = (c(2, 1))) )
df <- df %>%
  mutate(.row = row_number())

# Data Cleaning ----
## Convert Outcome Variable to Factor
df <- df %>% 
  mutate(red_density_state = factor(red_density_state))

## Drop NAs and NaNs from the dataset, Random Forest doesn't like this
df <- df %>% 
  drop_na()

# Create Training and Test Datasets ----
## Split the data while keeping group ids separate, groups will not be split up across training and testing tests 
set.seed(127)
holdout_field_id <- sample(unique(df$fields), size = 1)

indices <- list(
  analysis = df %>% filter(!(fields %in% holdout_field_id)) %>% pull(.row),
  assessment = df %>% filter(fields %in% holdout_field_id) %>% pull(.row)
)

split <- make_splits(indices, df)
df_train <- training(split)
df_test <- testing(split)

# Create Recipe ----
## Define a recipe to be applied to the data
df_recipe <- recipe(red_density_state ~ ., data = df_train) %>% 
  update_role(fields, quads, .row, new_role = "ID") %>% 
  step_rm("junk_col1", "junk_col2", "junk_col3") %>% 
  themis::step_downsample(all_outcomes(), skip = TRUE) %>% 
  step_center(all_predictors(), -all_nominal()) %>% 
  step_scale(all_predictors(), -all_nominal()) %>% 
  step_other(crop, threshold = 1, other = "other") %>% 
  step_dummy(all_nominal(), -all_outcomes())

summary(df_recipe)

# Prepare recipe and bake to obtain cleaned data frames: This will be used for variable importance ----
## Prepare Recipe
df_recipe_prep <- prep(df_recipe, training = df_train)

## Bake Recipe for Training and Testing sets
df_train_bake <- bake(df_recipe_prep, new_data = NULL)
df_test_bake <- bake(df_recipe_prep, new_data = df_test)

# Define Random Forest model via Parsnip ---
## Initialise Model with hyperparameters to be tunes
rf_spec <- rand_forest(trees = tune(), mtry = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Define a workflow to connect the recipe and model ----
rf_workflow <- 
  workflow() %>% 
  add_recipe(df_recipe) %>% 
  add_model(rf_spec)


# Train and Tune model ----
## Define a random grid for hyper parameters to vary over
set.seed(10)
rf_grid <- grid_random(mtry() %>% finalize(df %>% dplyr::select(values, crop)), # Predictor Values to be used from Raw data
                       trees(), 
                       size = 20) %>% 
  arrange(mtry, trees)

## Fit Model with resampling - Resampling is always used with the training set.
set.seed(234)
folds <- group_vfold_cv(data = df_train, group = "fields")
folds


## Tune candidate models in the grid
# Via parallel processing
# Parallel processing setup
cores <- parallel::detectCores(logical = FALSE)

# Register backend
cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)

## Parallel racing
set.seed(345)
rf_tuned <- rf_workflow %>% 
  tune_race_anova(resamples = folds,
            grid = rf_grid, 
            control = control_race(save_pred = TRUE),
            metrics = metric_set(roc_auc, accuracy))

## View performance metrics across all hyperparameter permutations
rf_tuned %>% 
  collect_metrics()

## Select the best model according to AUC
rf_best_model <- rf_tuned %>% 
  select_best(metric = "roc_auc")


# Finalise the Model: Select best model ----

## Update the workflow with the model with the best hyperparameters (obtained from select_best())
final_rf_workflow <- rf_workflow %>% 
  finalize_workflow(rf_best_model)

## Fit the final model to the training data
final_rf_model <- final_rf_workflow %>% 
  fit(data = df_train)

## Pull model from the workflow
final_rf_model %>% 
  pull_workflow_fit()

## Predict from final model
final_rf_model %>% 
  predict(df_train, type = "prob")


# Fit the model to the test data ----
## Use last_fit() this function fits the finalised model on the full training dataset and evaluates the finalised model on the testing data
rf_fit_final <- final_rf_model %>% 
  last_fit(split)

## Metrics on test set
rf_fit_final %>% 
  collect_metrics()

## Predictions on test set
rf_fit_final %>% 
  collect_predictions() %>%
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(df_test) 

## Confusion Matrix on test set
rf_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(red_density_state, .pred_class)

# Variable Importance ----
## Computed on baked training data - the model is fit to all the training data once again

set.seed(123)
rand_forest(trees = rf_best_model$trees, mtry = rf_best_model$mtry) %>% # Best parameters from the finalised workflow
  set_engine("ranger", importance = "permutation") %>% 
  set_mode("classification") %>% 
  fit(red_density_state ~ .,
      data = df_train_bake %>% 
        dplyr::select(-fields, -quads, -.row) # Remove ID columns
  ) %>% 
  vip(geom = "point", all_permutations = TRUE)




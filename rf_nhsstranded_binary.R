library(tidymodels)
library(readr)
library(broom)
library(broom.mixed)
library(skimr)
library(remotes)
library(dplyr)
library(magrittr)
library(parallel)
library(doParallel)
library(vip)
library(themis)
library(plotly)
library(ConfusionTableR)
library(finetune)
library(ranger)
library(randomForest)
library(butcher)
library(lobstr)
library(lubridate)
library(NHSRdatasets)

# Read in data ----
##  a stranded patient is a patient that has been in hospital for longer than 7 days and we also call these Long Waiters.
strand_pat <- NHSRdatasets::stranded_data %>% 
  setNames(c("stranded_class", "age", "care_home_ref_flag", "medically_safe_flag", 
             "hcop_flag", "needs_mental_health_support_flag", "previous_care_in_last_12_month", "admit_date", "frail_descrip")) %>% 
  mutate(stranded_class = factor(stranded_class),
         admit_date = as.Date(admit_date, format = "%d/%m/%Y")) %>% 
  drop_na()

# Explore data ----
## Analyse Class Imbalance
class_bal_table <- table(strand_pat$stranded_class)
prop_tab <- prop.table(class_bal_table)
upsample_ratio <- class_bal_table[2]/ sum(class_bal_table)

# Partition into training and test data splits ----
set.seed(123)
split <- initial_split(strand_pat)
train_data <- training(split)
test_data <- testing(split)  

# Create Recipe ----
## Define Recipe to be applied to the dataset
stranded_rec <- 
  recipe(stranded_class ~ ., data = train_data) %>% 
  # Make a day of week and month feature from admit date and remove raw admit date
  step_date(admit_date, features = c("dow", "month")) %>% 
  step_rm(admit_date) %>% 
  # Upsample minority (positive) class
  themis::step_upsample(stranded_class, over_ratio = as.numeric(upsample_ratio)) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

## Prepare and Bake recipe on training and test data
stranded_recipe_prep <- prep(stranded_rec, training = train_data)

stranded_train_bake <- bake(stranded_recipe_prep, new_data = NULL)
stranded_test_bake <- bake(stranded_recipe_prep, new_data = test_data)

# Create Model ----
rf_spec <- rand_forest(trees = tune(),
                       mtry = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Define a workfloe to connect the recipe and model ----
rf_workflow <- 
  workflow() %>% 
  add_recipe(stranded_rec) %>% 
  add_model(rf_spec)

# Train and Tune model ----
## Define a random grid for hyperparameters to vary over
set.seed(345)
rf_grid <- 
  grid_random(mtry() %>% finalize(strand_pat %>% select(-stranded_class)),
              trees(), 
              size = 10) %>% 
  arrange(mtry, trees)

## Fit model with resampling on training data
set.seed(456)
folds <- vfold_cv(train_data, v = 5)

## Tune candidate models in the grid
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1)
doParallel::registerDoParallel(cl) # Register Backend


set.seed(567)
rf_tuned <- 
  rf_workflow %>% 
  tune_grid(resamples = folds,
            grid = rf_grid,
            control = control_grid(),
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
  finalize_workflow(rf_tuned %>% 
                      select_best(metric = "roc_auc"))

## Fit the final model to the training data
final_rf_model <- final_rf_workflow %>% 
  fit(data = train_data)

## Pull model from the workflow
final_rf_model %>% 
  pull_workflow_fit()

## Predict from final model
final_rf_model %>% 
  predict(train_data, type = "prob")

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
  bind_cols(test_data) 

## Confusion Matrix on test set
rf_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(stranded_class, .pred_class)

## Generate ROC Curve
roc_plot <- 
  rf_fit_final %>% 
  collect_predictions() %>% 
  roc_curve(stranded_class, '.pred_Not Stranded') %>% 
  autoplot()

# Variable Importance ----
## Computed on baked training data - the model is fit to all the training data once again

set.seed(123)
rand_forest(trees = rf_best_model$trees, mtry = rf_best_model$mtry) %>% # Best parameters from the finalised workflow
  set_engine("ranger", importance = "permutation") %>% 
  set_mode("classification") %>% 
  fit(stranded_class ~ .,
      data = stranded_train_bake
  ) %>% 
  vip(geom = "point", all_permutations = TRUE)

# Save Model and Metrics ----
## Extract final fitted workflow
rf_wf_model <- rf_fit_final$.workflow[[1]]

## Save Metrics
collect_metrics(rf_fit_final) %>% 
  write_csv(file = "./saved_models/random_forest_model_metrics.csv")

## Save Model
### Measure object size of workflow
obj_size(rf_wf_model)

### Weigh in the workflow, the objects that are taking up the most memory
weigh(rf_wf_model)

### Butcher workflow to take up less space
rf_wf_model_reduced <- butcher::butcher(rf_wf_model)

### Check size difference
print(obj_size(rf_wf_model))
print(obj_size(rf_wf_model_reduced))
obj_size(rf_wf_model) - obj_size(rf_wf_model_reduced) 

### Save model object as an RDS object
saveRDS(rf_wf_model_reduced, file = "./saved_models/random_forest_stranded.rds")

# Reading in workflow and predicting ----
## rm(rf_wf_model)
rf_wf_model <- readRDS(file = "./saved_models/random_forest_stranded.rds")

## Predict on test data with loaded workflow ----
test_sample <- test_data %>% 
  slice_sample(n = 50)

rf_wf_model %>% 
  predict(test_sample) %>% 
  cbind(stranded_class = test_sample$stranded_class)




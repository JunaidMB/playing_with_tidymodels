library(tidyverse)
library(tidymodels)
library(doParallel)
library(earth)

# We want to predict the weight of giant pumpkins from other characteristics

# Read Data ----
pumpkins_raw <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-10-19/pumpkins.csv")

pumpkins <-
  pumpkins_raw %>%
  separate(id, into = c("year", "type")) %>%
  mutate(across(c(year, weight_lbs, ott, place), parse_number)) %>%
  filter(type == "P") %>%
  select(weight_lbs, year, place, ott, gpc_site, country)

# Split Data ----
## Training and Test Split
set.seed(123)

pumpkin_split <- pumpkins %>% 
  filter(ott > 20, ott < 1e3) %>% 
  initial_split(strata = weight_lbs)

pumpkin_train <- training(pumpkin_split)
pumpkin_test <- testing(pumpkin_split)

## Cross Validation Folds
set.seed(234)
pumpkin_folds <- vfold_cv(pumpkin_train, strate = weight_lbs)

# Define Recipes ----
'''
Create 3 data preprocessing recipes:
1. One that pools infrequenlty factor levels
2. One that pools factors AND creates indicator variables
3. One that creates spline terms for over-the-stop inches (as well as recipe steps 1 and 2)
'''

base_rec <- 
  recipe(weight_lbs ~ ott + year + country + gpc_site, data = pumpkin_train) %>% 
  step_other(country, gpc_site, threshold = 0.02)

ind_rec <- 
  base_rec %>% 
  step_dummy(all_nominal_predictors())

spline_rec <- 
  ind_rec %>% 
  step_bs(ott)

# Define Models ----
'''
Define 3 models:
1. Random Forest
2. MARS model
3. Linear model
'''

rf_spec <- 
  rand_forest(trees = 1e3) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")

mars_spec <- 
  mars() %>% 
  set_mode("regression") %>% 
  set_engine("earth")

lm_spec <- 
  linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm")
  

# Define Workflowset ----
pumpkin_set <- 
  workflow_set(
    preproc = list(base_rec, ind_rec, spline_rec),
    models = list(rf_spec, mars_spec, lm_spec),
    cross = FALSE
  )

# Train and Tune Models ----
## No hyperparameter tuning
doParallel::registerDoParallel()
set.seed(2021)

pumpkin_rs <- 
  workflow_map(
    pumpkin_set,
    "fit_resamples",
    resamples = pumpkin_folds
  )

# Inspect Results ----
## Inspect Regression Metrics
collect_metrics(pumpkin_rs)

## Ranked results
pumpkin_rs %>% 
  rank_results(rank_metric = "rmse")

## Plot results
autoplot(pumpkin_rs)

# Select Best Model ----
best_model <- 
  pumpkin_rs %>% 
  rank_results(rank_metric = "rmse") %>% 
  filter(.metric == "rmse", rank == 1) %>% 
  pull(wflow_id)

best_results <- 
  pumpkin_rs %>% 
  extract_workflow_set_result(best_model) %>% 
  select_best(metric = "rmse")

# Finalize Model ----
best_workflow <- 
  pumpkin_rs %>% 
  pull_workflow(best_model)

best_workflow_fit <- 
  best_workflow %>% 
  finalize_workflow(best_results) %>% 
  fit(data = pumpkin_train)

## Examine model parameters
tidy(best_workflow_fit) %>% 
  arrange(-abs(estimate))

# Evaluate on Test Set ----
final_best_workflow_fit <- best_workflow_fit %>%
  last_fit(pumpkin_split)

## Make a table of Model Predictions 
test_pred <- 
  final_best_workflow_fit %>% 
  pull(.workflow[[1]]) %>% 
  '[['(1) %>% 
  predict(pumpkin_test) %>% 
  bind_cols(pumpkin_test)

## Test RMSE
rmse(test_pred, truth = weight_lbs, estimate = .pred)


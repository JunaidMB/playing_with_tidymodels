library(tidyverse)
library(tidytext)
library(tidylo)
library(tidymodels)
library(themis)
library(textrecipes)
library(here)
library(glmnet)
library(parallel)

# Read Data ----
papers <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-09-28/papers.csv")
programs <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-09-28/programs.csv")
paper_authors <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-09-28/paper_authors.csv")
paper_programs <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-09-28/paper_programs.csv")

# Join to create dataset ----
papers_joined <-
  paper_programs %>%
  left_join(programs) %>%
  left_join(papers) %>%
  filter(!is.na(program_category)) %>%
  distinct(paper, program_category, year, title)

## View Target variable distribution
papers_joined %>%
  count(program_category)

# Split Data ----
set.seed(123)
nber_split <- initial_split(papers_joined, strata = program_category)
nber_train <- training(nber_split)
nber_test <- testing(nber_split)

## Make Cross Validation Folds
set.seed(234)
nber_folds <- vfold_cv(nber_train, strata = program_category)

# Create Recipe ----
nber_rec <- 
  recipe(program_category ~ year + title, data = nber_train) %>% 
  step_tokenize(title) %>% 
  step_tokenfilter(title, max_tokens = 200) %>% 
  step_tfidf(title) %>% 
  step_downsample(program_category)

nber_prep <- prep(nber_rec)
nber_train_baked <- bake(nber_prep, new_data = NULL)
nber_test_baked <- bake(nber_prep, new_data = nber_test)

# Create Model ----
multi_spec <- 
  multinom_reg(penalty = tune(), mixture = 1) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

# Create workflow to connect recipe and model ----
nber_wf <- workflow() %>% 
  add_recipe(nber_rec) %>% 
  add_model(multi_spec)

# Train and Tune model ----
## Create Hyperparameter grid
nber_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 20)

## Tune Model in Parallell
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1) 
doParallel::registerDoParallel()

set.seed(2021)
nber_rs <- nber_wf %>% 
  tune_grid(
    resamples = nber_folds,
    grid = nber_grid,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc, accuracy)
  )

## View performance metrics across all hyperparameter permutations
nber_rs %>% 
  collect_metrics()

# Plot Model performance ----
autoplot(nber_rs)

# Finalise Model ----
## Update workflow with the best model found
nber_best_model <- nber_rs %>% 
  select_by_one_std_err(metric = "roc_auc", desc(penalty))

final_penalty_workflow <- nber_wf %>% 
  finalize_workflow(nber_best_model)

## Fit the final model to all the training data
final_penalty_model <- final_penalty_workflow %>% 
  fit(data = nber_train)

# Fit model to test data ----
## Fit finalised model on all training data and evaluate on test data
penalty_fit_final <- final_penalty_model %>% 
  last_fit(nber_split)

# Model Evaluation ----
## Metrics on test set
penalty_fit_final %>% 
  collect_metrics()

## Predictions on test set
penalty_fit_final %>% 
  collect_predictions() %>% 
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(nber_test)

## Confusion Matrix on test set
penalty_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(program_category, .pred_class)

## ROC curves by class
collect_predictions(penalty_fit_final) %>%
  roc_curve(truth = program_category, .pred_Finance:.pred_Micro) %>%
  ggplot(aes(1 - specificity, sensitivity, color = .level)) +
  geom_abline(slope = 1, color = "gray50", lty = 2, alpha = 0.8) +
  geom_path(size = 1.5, alpha = 0.7) +
  labs(color = NULL) +
  coord_fixed()

# Save Model and Metrics ----
## Extract final fitted workflow
final_fitted <- penalty_fit_final$.workflow[[1]]

## make up new paper titles and see how our model classifies them.
predict(final_fitted, tibble(year = 2021, title = "Pricing Models for Corporate Responsibility"), type = "prob")


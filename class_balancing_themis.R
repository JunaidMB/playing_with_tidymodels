library(modeldata)
library(recipes)
library(themis)

# Load Data ----
data("okc")
table(okc$Class)

# Hybrid Sampling: Adding observations to the minority class ----
hybrid_rec <- recipe(Class ~ age + diet + height, data = okc) %>% 
  # turns NAs to "unknown"
  step_unknown(diet) %>% 
  # One Hot Encodes the diet variable
  step_dummy(diet) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  # Impute numeric predictors with their column mean
  step_impute_mean(all_numeric_predictors()) %>% 
  # Balance classes with SMOTE 
  step_smote(Class)

## Prepare and Bake recipe 
hybrid_recipe_prep <- prep(hybrid_rec, training = okc)

## Bake Recipe 
hybrid_baked_data <- bake(hybrid_recipe_prep, new_data = NULL)

hybrid_baked_data %>% 
  pull(Class) %>% 
  table()

# Undersampling: Removing observations from the majority class ----
undersample_rec <- recipe(Class ~ age + diet + height, data = okc) %>% 
  # turns NAs to "unknown"
  step_unknown(diet) %>% 
  # One Hot Encodes the diet variable
  step_dummy(diet) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  # Impute numeric predictors with their column mean
  step_impute_mean(all_numeric_predictors()) %>% 
  step_downsample(Class)

## Prepare and Bake recipe 
undersample_recipe_prep <- prep(undersample_rec, training = okc)

## Bake Recipe 
undersample_baked_data <- bake(undersample_recipe_prep, new_data = NULL)

undersample_baked_data %>% 
  pull(Class) %>% 
  table()



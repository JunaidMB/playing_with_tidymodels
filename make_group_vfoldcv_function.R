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

# Develop cross validation function that splits according to groups ----
group_vfold_cv2 <- function(data, group_id, group_prop, num_folds = 10) {

if (num_folds <= 0 ){
  stop("num_folds must be greater than 0")
} else
if (num_folds < 10) {
  id <- c(paste0("Fold0", 1:num_folds))
} else 
  if (num_folds >= 10) {
id <- c(paste0("Fold0", 1:9), paste0("Fold", 10:num_folds))
  }
  
if (!between(group_prop, 0, 1)) {
  stop("group_prop must be between 0 and 1")
}

splits_list <- list()

for (i in 1:num_folds) {
  
  holdout_field_id <- sample(unique(data[[group_id]]), size = ceiling(group_prop*(length(unique(data[[group_id]])))) )
  
  indices <- list(
    analysis = data %>% filter(!(.data[[group_id]] %in% holdout_field_id)) %>% pull(.row),
    assessment = data %>% filter(.data[[group_id]] %in% holdout_field_id) %>% pull(.row)
  )
  
  split <- make_splits(indices, data)
  splits_list[[i]] <- split
  
}

df_folds <- tibble(splits = splits_list, id = id)

return(df_folds)

}

group_vfold_cv2(data = df, group_id = "fields", group_prop = 0.1, num_folds = 5)



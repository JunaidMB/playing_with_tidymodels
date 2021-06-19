library(tidyverse)
library(tidymodels)
library(butcher)


# Load Data ----
records <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-25/records.csv")

# Choose random subset of data to mimic test data ----
records_sample <- records %>% 
  slice_sample(prop = 0.1) %>% 
  mutate(time = time + rnorm(n= 1, mean = 0, sd = 5))

# Read in fitted workflow ----
pretrained_model <- readRDS(file = "saved_models/tree_wf_model_mariokart.rds")

# Predict on mimicked test data with loaded workflow ----
pretrained_model %>% 
  predict(records_sample)

# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)
library(dbarts)

# Read in Data
train <- vroom("train.csv")
test <- vroom("test.csv")


# Recipe
my_recipe <- recipe(type ~ ., data = train) %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())


## Model

## or BART
bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

#Workflow
bart_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(bart_model)

## CV tune, finalize

## Set up a grid of tuning values
tuning_grid <- grid_regular(trees(),
                            levels = 6)

## Set up K-fold CV
folds <- vfold_cv(train, v = 10, repeats = 1)

## Run the CV
CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))


## Find best tuning parameters
bestTune <- CV_results %>% 
  select_best(metric = "accuracy")

## Finalize workflow and predict 
final_wf <-
  boosted_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = train)

## Predict
# make predictions
predictions <- predict(final_wf,
                       new_data = test,
                       type = "class")

submission <- predictions |>
  rename(type = .pred_class) %>% 
  select(type) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, type) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission, file = "./bartggg.csv", delim = ",")

# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(kknn)
library(themis)
library(discrim)

# Read in Data
train <- vroom("train.csv")
test <- vroom("test.csv")

## RAN FOREST 
# Recipe

# Model
## Define a model
my_recipe <- recipe(type ~ ., data = train) %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)%>% 
  step_smote(all_outcomes(), neighbors = 3)
  

## nb model
nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

## work flow
nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_model)

## Tuning
## Set up a grid of tuning values
grid_of_tuning_params_nb <- grid_regular(Laplace(),
                                         smoothness(),
                                         levels = 20)

## Set up K-fold CV
folds_nb <- vfold_cv(train, v = 10, repeats = 1)

## Find best tuning parameters
CV_results_nb <- nb_wf %>% 
  tune_grid(resamples = folds_nb,
            grid = grid_of_tuning_params_nb,
            metrics = metric_set(accuracy, roc_auc))

bestTune_nb <- CV_results_nb %>% 
  select_best(metric = "roc_auc")

## Finalize workflow and predict 
final_wf_nb <-
  nb_wf %>% 
  finalize_workflow(bestTune_nb) %>% 
  fit(data = train)

# make predictions
predictions <- predict(final_wf_nb,
                       new_data = test,
                       type = "class")

submission_nb <- predictions |>
  rename(type = .pred_class) %>% 
  select(type) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, type) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_nb, file = "./NBayesGGG5.csv", delim = ",")

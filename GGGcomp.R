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
my_recipe <- recipe(type ~ ., data = train) %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_normalize(all_numeric_predictors())
bake(prep(my_recipe), new_data=train)

# Model
## Define a model
my_mod_rf <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

## Create a workflow w/ model and recipe

#Workflow
randfor_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod_rf)
# 
#  prepped_recipe <- prep(bike_recipe)
#  final <- bake(prepped_recipe, new_data = new_train)


## Set up a grid of tuning values
grid_of_tuning_params_rf <- grid_regular(mtry(range = c(1, 20)),
                                         min_n(),
                                         levels = 4)

## Set up K-fold CV
folds_rf <- vfold_cv(train, v = 10, repeats = 1)

## Find best tuning parameters
CV_results_rf <- randfor_wf %>% 
  tune_grid(resamples = folds_rf,
            grid = grid_of_tuning_params_rf,
            metrics = metric_set(accuracy))

bestTune_rf <- CV_results_rf %>% 
  select_best(metric = "accuracy")

## Finalize workflow and predict 
final_wf_rf <-
  randfor_wf %>% 
  finalize_workflow(bestTune_rf) %>% 
  fit(data = train)

## Predict
# make predictions
predictions <- predict(final_wf_rf,
                       new_data = test,
                       type = "class")

submission <- predictions |>
  rename(type = .pred_class) %>% 
  select(type) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, type) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission, file = "./gggranfor3.csv", delim = ",")



# KNN

# KNN Model
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

## workflow
knn_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_model)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 10)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- knn_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

## Find best tuning parameters
bestTune <- CV_results %>% 
  select_best(metric = "accuracy")

## Finalize workflow
final_wf <-
  knn_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = train)

# make predictions
knn_predictions <- predict(final_wf,
                              new_data = test,
                              type = "class")

submission_knn <- knn_predictions |>
  rename(type = .pred_class) %>% 
  select(type) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, type) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_knn, file = "./GGGKnnPred.csv", delim = ",")

### Naive Bayes
# Recipe


# Model
## Define a model
my_recipe <- recipe(type ~ ., data = train) %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_normalize(all_numeric_predictors())

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
                                         levels = 5)

## Set up K-fold CV
folds_nb <- vfold_cv(train, v = 5, repeats = 1)

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
vroom_write(x= submission_nb, file = "./NBayesGGG4.csv", delim = ",")


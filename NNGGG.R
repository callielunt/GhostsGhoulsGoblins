# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(kknn)
library(themis)
library(discrim)
library(keras)

# Read in Data
train <- vroom("train.csv")
test <- vroom("test.csv")

##
# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")
# reticulate::install_python()
# 
# keras::install_keras()
# 
# library(reticulate)
# py_config()

library(keras)
# install_keras()

nn_recipe <- recipe(type ~ ., data = train) %>% 
  # update_role(id, new_role = "id") %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]


nn_model <- mlp(hidden_units = tune(),
                epochs = 100) %>%
  set_engine("keras") %>% #verbose = 0 prints off less9
  set_mode("classification")

nn_wf <- workflow() %>% 
  add_recipe(nn_recipe) %>% 
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)), levels = 5)

## Set up K-fold CV
folds_nn <- vfold_cv(train, v = 5, repeats = 1)

tuned_nn <- nn_wf %>% 
  tune_grid(resamples = folds_nn,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy, roc_auc))

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="roc_auc") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()
  

  
bestTune_nn <- tuned_nn %>% 
  select_best(metric = "roc_auc")

## Finalize workflow and predict 
final_wf_nn <-
  nn_wf %>% 
  finalize_workflow(bestTune_nn) %>% 
  fit(data = train)

# make predictions
predictions <- predict(final_wf_nn,
                       new_data = test,
                       type = "class")

submission_nn <- predictions |>
  rename(type = .pred_class) %>% 
  select(type) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, type) # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission_nn, file = "./nnGGG100.csv", delim = ",")


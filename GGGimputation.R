# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)

# Read in Data
trainSet <- vroom("train.csv")
missSet <- vroom("trainWithMissingValues.csv")
missSet$type <- as.factor(missSet$type)
missSet$color <- as.factor(missSet$color)

colMeans(is.na(missSet))

DataExplorer::plot_missing(missSet) # see which one has most missing


recipe <- 
  recipe(type ~ ., data = missSet) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(color, has_soul, color, id),
                  neighbors = 7) %>% 
  step_impute_knn(rotting_flesh, impute_with = imp_vars(color, has_soul, color, id, hair_length), 
                  neighbors = 7) %>% 
  step_impute_knn(bone_length, impute_with = imp_vars(color, has_soul, color, id, hair_length, rotting_flesh), 
                  neighbors = 7)

prepped <- prep(recipe)  
imputedSet <- bake(prepped, new_data=missSet)

rmse_vec(trainSet[is.na(missSet)],
         imputedSet[is.na(missSet)])



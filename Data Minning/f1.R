library(ggplot2)
library(grid)
library(gridExtra)
library(dplyr)
library(rpart)
library(pROC)
library(survival)
library(pROC)
library(DMwR)
library(scales)
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm

library(caret)        # an aggregator package for performing many machine learning models

library(h2o)          # a java-based platform

data<-read.csv("C:/Users/dhruv/Desktop/Final Project/IBM-Health Analytics/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data
set.seed(123)
n <- nrow(data)
rnd <- sample(n,  n*0.7) # get sample dataset, use 70% of data
train <- data[rnd,]
test <- data[-rnd,]

set.seed(3433)

# Setting the basic train control used in all GBM models

ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

# Simple GBM

gbmfit <- train(Attrition ~., 
                data = train, 
                method = "gbm", 
                verbose = FALSE, 
                metric = "ROC", 
                trControl = ctrl)
gbm.fit <- gbm(
  formula = Attrition ~ .,
  distribution = "gaussian",
  data = train,
  n.trees=5000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
  )
model_gbm <- caret::train(Attrition ~ .,
                          data = train,
                          method = "gbm",
                          trControl = trainControl(method = "repeatedcv", 
                                                  number = 5, 
                                                  repeats = 3, 
                                                  verboseIter = FALSE),
                          verbose = 0)
model_gbm
print(gbm.fit)

gbmpreds <- predict(gbm.fit, test)
gbmpreds
gbm.perf(gbm.fit, method = "cv")

sqrt(min(gbm.fit$cv.error))
rocgbm <- roc(as.numeric(test$Attrition), as.numeric(gbmpreds))
rocgbm$auc

# create hyperparameter grid

hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0      
)




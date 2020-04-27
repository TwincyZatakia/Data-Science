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
print(gbm.fit)

gbmpreds <- predict(gbm.fit, test)
gbmpreds
rocgbm <- roc(as.numeric(test$Attrition), as.numeric(gbmpreds))
rocgbm
rocgbm$auc
plot(rocgbm, ylim = c(0,1), print.thres = T, print.thres.cex = 0.8, col = "burlywood", add = T)
gbm.perf(gbm.fit, method = "cv")
sqrt(min(gbm.fit$cv.error))

gbm.fit2 <- gbm(
  formula = Attrition ~ .,
  distribution = "gaussian",
  data = train,
  n.trees=1000,
  interaction.depth = 2,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
  )
gbmpreds <- predict(gbm.fit2, test)
gbmpreds

gbm.perf(gbm.fit2, method = "cv")
sqrt(min(gbm.fit2$cv.error))

rocgbm <- roc(as.numeric(test$Attrition), as.numeric(gbmpreds))
plot(rocgbm)
rocgbm$auc



# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit2$cv.error)
min_MSE
# get MSE and compute RMSE
sqrt(gbm.fit2$cv.error[min_MSE])
# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit2, method = "cv")

# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0      
)

nrow(hyper_grid)
rocgbm <- roc(as.numeric(test$Attrition), as.numeric(gbmpreds))
plot(rocgbm, ylim = c(0,1), print.thres = T, print.thres.cex = 0.8, col = "burlywood", add = T)


# randomize data
random_index <- sample(1:nrow(train), nrow(train))
random_train <- train[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = Attrition ~ .,
    distribution = "gaussian",
    data = train,
    n.trees = 3000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )

hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}
hyper_grid$min_RMSE[i]
hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)


# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = Attrition ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 69,
  interaction.depth = 5,
  shrinkage = 0.1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
  )  
print(gbm.fit.final)

gbmpreds <- predict(gbm.fit.final, test)
gbmpreds
rocgbm <- roc(as.numeric(test$Attrition), as.numeric(gbmpreds))
rocgbm$auc
gbm.perf(gbm.fit.final, method = "cv")
sqrt(min(gbm.fit$cv.error))

par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
  )


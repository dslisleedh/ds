library(e1071)
library(caret)
library(xgboost)
library(tidyverse)
library(skimr)
library(MLmetrics)
library(Metrics)
library(ggplot2)

data <- tibble(read.table('pima-indians-diabetes.data', sep = ','))
test <- read.table('pima-indians-diabetes_test.data', sep = ',')
train <- read.table('pima-indians-diabetes_train.data', sep = ',')
colname <- c('pregnant', 'glucose', 'bloodpressure', 'skin', 'insulin', 'bmi' ,'diabetes_faimily','age', 'class')
colnames(data) <- colname
data %>% glimpse()
data %>% skim()
data %>% correlate()

#k-nn
X <- scale(train[,-9])
y <- as.factor(train[,9])
test_x <- scale(test[,-9])

trControl <- trainControl(method = 'cv',
                          number = 10)

start <- Sys.time()
set.seed(201600177)
fit <- caret::train(x = X,
                    y = y,
                    method = 'knn',
                    tuneGrid = expand.grid(k = seq(1,39,2)),
                    trControl = trControl,
                    metric = 'Accuracy')
end <- Sys.time()
start - end
fit$bestTune
fit$results[fit$results$k == as.numeric(a),]

Accuracy(predict(fit, test_x), test[,9])

# accuacy = 0.76 , 20 comb, 1.09sec

ggplot(fit, aes(x = k, y = Accuracy)) + geom_area(color = '#118874', fill = '#ABE7DD') + labs(title = 'K의 변화에 따른 정확도 변화') + 
  coord_cartesian(ylim = c(0.7,0.76))











#SVM, C-classification, rbf kernel
X <- scale(train[,-9])
y <- as.factor(train[,9])
test_x <- scale(test[,-9])

trControl <- tune.control(cross = 10)
start <- Sys.time()
set.seed(201600177)
svmtune <- tune.svm(x = X, y = y,type = 'C-classification', kernel = 'radial', cost = 2^(-7:7), gamma = 10^(-4:1), tunecontrol = trControl)
end <- Sys.time()
svmtune$best.parameters
start - end
svmtune
svmtune$performances
ggplot(svmtune$performances, aes(x = factor(gamma), factor(cost))) + geom_tile(aes(fill = error)) + scale_fill_gradient(low = 'blue', high = 'red') +
  labs(title = 'C와 gamma의 변화에 따른 오류율의 변화')


set.seed(201600177)
obj <- svm(X,y, type = 'C-classification',
           kernel = 'radial', 
           gamma = svmtune$best.parameters[1],
           cost = svmtune$best.parameters[2])
result <- predict(obj, newdata = test_x[,-9])
Accuracy(as.numeric(as.character(result)), test[,9])
start - end
table(as.numeric(as.character(result)), test[,9])
#accuacy : 0.77, 90 comb, 26.06sec












#xgboost, gbtree, binary:logistic
xgb.train <- xgb.DMatrix(data = as.matrix(train[,-9]), label = as.matrix(train[,9]))
xgb.test <- xgb.DMatrix(data = as.matrix(test[,-9]), label = as.matrix(test[,9]))



hp <- expand.grid(eta = 10^(-4:-1),
            max_depth = seq(5,10,1),
            subsample = seq(0.5,1,0.1),
            colsample_bytree = seq(0.625,1,0.125),
            gamma = seq(0,5,1),
            min_child_weight = seq(1,7,2))
nrow(hp)

make_parameter <- function(i){
  return(list(booster = 'gbtree', objective = 'binary:logistic', eval_metric = 'error',
              eta = as.numeric(hp[i,1]), 
              max_depth = as.numeric(hp[i,2]),
              subsample = as.numeric(hp[i,3]),
              colsample_bytree = as.numeric(hp[i,4]),
              gamma = as.numeric(hp[i,5]),
              min_child_weight = as.numeric(hp[i,6])))
}
a <- make_parameter(1)

hp
grid_search <- function(train_data){
  start <- Sys.time()
  error <- c()
  for(i in 1:nrow(hp)){
    set.seed(201600177)
    xgb_cv <- xgb.cv(params = make_parameter(i), nfold = 10, data = train_data, nrounds = 100000, early_stopping_rounds = 15, verbose = F, nthread = 16)
    error[i] <- as.numeric(xgb_cv$evaluation_log[xgb_cv$best_iteration,2])
    print((i/nrow(hp))*100)
  }
  end <- Sys.time()
  print(difftime(start, end, units = 'secs'))
  return(error)
}
result <- grid_search(xgb.train)
result
hp[840,]
result[match(min(result[,1]), result),]
set.seed(201600177)
xgb.fit <- xgb.train(params = make_parameter(match(min(result), result)),
                     data = xgb.train,
                     nround = 500,
                     nthread = 16)
xgb.fit
xgb.result <- predict(xgb.fit, xgb.test)
Accuracy(round(xgb.result), test[,9])
#accuacy : 0.78, 13824 comb,  7193.309sec
xgb.fit$params
imp <- xgb.importance(feature_names = colname, model = xgb.fit)
imp

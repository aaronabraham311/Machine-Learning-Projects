library(tidyverse)
library(caret)

# Reading in data
data <- read.csv("data.csv")
knnModel <- readRDS("knnmodel.rds")
svmModel <- readRDS("svmmodel.rds")
xgModel <- readRDS("xgmodel.rds")

# Training and testing data
indices <- createDataPartition(data$diagnosis, p = 0.7, list = FALSE) #70% of data will be for training
train <- data[indices,]
test <- data[-indices,]

# Creating ensemble training and testing data
knnPredictions <- predict(knnModel, train)
svmPredictions <- predict(svmModel, train)
xgPredictions <- predict(xgModel, train)

test$knnPredictions <- predict(knnModel, test)
test$svmPredictions <- predict(svmModel, test)
test$xgPredictions <- predict(xgModel, test)

preddf <- data.frame(knnPredictions, svmPredictions, xgPredictions, train$diagnosis)

# Accuracy metrics functions
accuracyMetric <- function (conMatrix) {
  tp <- conMatrix[1,1]
  fp <- conMatrix[1,2]
  fn <- conMatrix[2,1]
  tn <- conMatrix[2,2]
  
  metric <- (tp + tn)/(tp + fp + fn + tn)
  return(metric)
}

mccMetric <- function (conMatrix) {
  tp <- conMatrix[1,1]
  fp <- conMatrix[1,2]
  fn <- conMatrix[2,1]
  tn <- conMatrix[2,2]
  
  metric <- ((tp * tn) - (fp * fn))/(sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
  return(metric)
}

# Training ensemble
controlParameters <- trainControl(
  method = "cv",
  number = 10,
  savePrediction = TRUE,
  classProbs = TRUE
)

ensemble <- train(train.diagnosis ~ ., 
                  method = "rf", 
                  data = preddf,
                  trControl = controlParameters)

# Testing model
ensemblePredictions <- predict(ensemble, test)
confMatrixEnsemble <- table(predictions = ensemblePredictions, actual = test$diagnosis)
ensembleAccuracy <- accuracyMetric(confMatrixEnsemble) # Same as xgmodel
ensembleMCC <- mccMetric(confMatrixEnsemble)

# Writing model
write_rds(ensemble, "ensemble.rds")

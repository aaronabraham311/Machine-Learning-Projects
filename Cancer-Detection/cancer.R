library(tidyverse)
library(caret)
library(reshape2)
library(beeswarm)
library(ellipse)
library(RColorBrewer)

# Reading in data
data <- read.csv("data.csv")
data <- subset(data, select = -c(X))

ncol(data) # 32 different features

# Finding missing features.
apply(data, 2, function(x) any(is.na(x))) # No missing values

# FEATURE SELECTION: Looking at the data, there is a lot on ...._se.
# This is standard error so we should drop it

data <- subset(data, select = -c(13:22))

# There isn't much feature engineering that we can do. Interested to look at the proportion of malignant and benign.

count(data, diagnosis) # Distribution isn't bad. Benign: 357 instances. Malignant: 212. 

# VISUALIZATION:

## Violin plots
ggplot(data, aes(x = diagnosis, y = radius_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Significant difference

ggplot(data, aes(x = diagnosis, y = texture_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Not very significant. Outliers on both 

ggplot(data, aes(x = diagnosis, y = perimeter_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Significant difference

ggplot(data, aes(x = diagnosis, y = area_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Range is smaller in benign data. Malignant has outliers

ggplot(data, aes(x = diagnosis, y = smoothness_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Not very significant. Outliers present

ggplot(data, aes(x = diagnosis, y = compactness_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Not very significant

ggplot(data, aes(x = diagnosis, y = concavity_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Lots of outliers in benign

ggplot(data, aes(x = diagnosis, y = concave.points_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) # Significant difference. Lots of outliers

ggplot(data, aes(x = diagnosis, y = symmetry_mean, fill = diagnosis)) + geom_violin(draw_quantiles = TRUE) + 
  geom_boxplot(width = 0.1) #Not very significant. Outliers!

## Heatmap:
heatmap_data <- data %>% filter(data$diagnosis == "M") %>% subset(select = c(3:22))
cormat <- round(cor(heatmap_data),2) #Correlation matrix

get_lower_tri <- function(cormat) {
  cormat[upper.tri(cormat)] <- NA
  return (cormat)
}

get_upper_tri <- function(cormat) {
  cormat[lower.tri(cormat)] <- NA
  return (cormat)
}

upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "#FFFC99", high = "#49227F", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed() +
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  ggtitle("Correlation Heatmap of Factors")

# Ellipse heatmap
elData <- cor(data[,3:12])

my_color <- brewer.pal(6, "Spectral")
my_color = colorRampPalette(my_color)(100)

ord <- order(elData[1, ])
data_ord = elData[ord, ord]
plotcorr(data_ord , col= my_color[data_ord*50+50] , mar=c(1,1,1,1)  )

## Swarm plots to experiment:
beeswarm(area_mean ~ diagnosis, data = data, method = "swarm", col = 2:3)

# Random scatter plots:
ggplot(data, aes(x = perimeter_worst, y = smoothness_worst, color = diagnosis)) +
  geom_point()

ggplot(data, aes(x = area_worst, y = smoothness_worst, color = diagnosis)) +
  geom_point()

featurePlot(x = data[,3:7], y = data[,2], plot = "pairs")

# MACHINE LEARNING: XGBOOST (http://xgboost.readthedocs.io/en/latest/model.html)

## Metric functions
mccMetric <- function (conMatrix) {
  tp <- conMatrix[1,1]
  fp <- conMatrix[1,2]
  fn <- conMatrix[2,1]
  tn <- conMatrix[2,2]
  
  metric <- ((tp * tn) - (fp * fn))/(sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
  return(metric)
}

accuracyMetric <- function (conMatrix) {
  tp <- conMatrix[1,1]
  fp <- conMatrix[1,2]
  fn <- conMatrix[2,1]
  tn <- conMatrix[2,2]
  
  metric <- (tp + tn)/(tp + fp + fn + tn)
  return(metric)
}

## Splitting data and shuffling
indices <- createDataPartition(data$diagnosis, p = 0.7, list = FALSE) #70% of data will be for training
train <- data[indices,]
test <- data[-indices,]

train <- train[sample(nrow(train)),]

## Cross-validation
controlParameters <- trainControl(
  method = "cv",
  number = 10,
  savePrediction = TRUE,
  classProbs = TRUE
)

## Training model
xgmodel <- train(diagnosis ~.,
               method = "xgbTree",
               data = train,
               trControl = controlParameters )

## Variable importance
plot(varImp(xgmodel), main = "Variable Importance - xgboost")

## Predicting
xgPredictions <- predict(xgmodel, test)

## Confusion matrix and metrics
xgmatrix <- table(predictions = xgPredictions, actual = test$diagnosis)

xgmccScore <- mccMetric(xgmatrix)
xgaccuracyScore <- accuracyMetric(xgmatrix)

## Training KNN model
knnmodel <- train(diagnosis ~.,
               method = "knn",
               data = train,
               trControl = controlParameters )

## Variable importance
plot(varImp(knnmodel), main = "Variable Importance - KNN")

## Predicting
knnPredictions <- predict(knnmodel, test)

## Confusion matrix and metrics
knnmatrix <- table(predictions = knnPredictions, actual = test$diagnosis)

knnmccScore <- mccMetric(knnmatrix)
knnaccuracyScore <- accuracyMetric(knnmatrix)

## Training SVM model
svmmodel <- train(diagnosis ~.,
               method = "svmRadial",
               data = train,
               trControl = controlParameters )

## Variable importance
plot(varImp(svmmodel), main = "Variable Importance - SVM")

## Predicting
svmPredictions <- predict(svmmodel, test)

## Confusion matrix and metrics
svmmatrix <- table(predictions = svmPredictions, actual = test$diagnosis)

svmmccScore <- mccMetric(svmmatrix)
svmaccuracyScore <- accuracyMetric(svmmatrix)

# Writing models
write_rds(xgmodel,"xgmodel.rds")
write_rds(knnmodel, "knnmodel.rds")
write_rds(svmmodel,"svmmodel.rds")


# ADD SE AND ALSO TRY PCA. USE ONLY TOP FEATURES TO MAKE SIMILAR ALGORITHM
# https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
#https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/

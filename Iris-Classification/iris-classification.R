# Including libraries
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(corrplot)
library(grid)
library(gridExtra)
library(rpart)
library(caret)

# Reading data
iris <- read_csv("iris.csv")
View(iris)

# From data, we can see that there are 4 values that we can use to classify the irises.
# Note that the data already classified the flowers. We can use this to visualize the dataset

# Question one: Can we use univariate analysis to distinguish the different petals

# Visualize sepal length
sepalLength <- iris %>% select(SepalLengthCm, Species)
ggplot(sepalLength) + geom_point(aes(SepalLengthCm, color = Species), stat = "count") +
  ggtitle("Sepal Length")
# Notes: Sepal length order: setosa < versicolor < virginica

# Visualize sepal width
sepalWidth <- iris %>% select(SepalWidthCm, Species)
ggplot(sepalWidth) + geom_point(aes(SepalWidthCm, color = Species), stat = "count") +
  ggtitle("Sepal Width")
# Notes: Versicolor sepal width is on the lower side, while setosa has bigger width.
# Virginica is more spread out

# Visualize petal length
petalLength <- iris %>% select(PetalLengthCm, Species)
ggplot(petalLength) + geom_point(aes(PetalLengthCm, color = Species), stat = "count") +
  ggtitle("Petal Length")
# Notes: Petal length order: setosa < versicolor < virginca. Same as sepal length (correlation)

# Visualize petal width
petalWidth <- iris %>% select(PetalWidthCm, Species)
ggplot(petalWidth) + geom_point(aes(PetalWidthCm, color = Species), stat = "count") +
  ggtitle("Petal Width")
# Notes: Petal width order: setosa < versicolor < virginica. Same as before!

# Conclusion: Univariate analysis could be used to distinguish between the different flowers. 

# Question 2: Can bivariate analysis be used to distinguish between the different species
# Approach: use scatterplots and then use a correlation plot based on finidings from univariate analysis

# Sepal length vs Petal length
sepalPetalLength <- iris %>% select(SepalLengthCm, PetalLengthCm, Species)
ggplot(sepalPetalLength) + geom_point(aes(SepalLengthCm, PetalLengthCm, color = Species)) +
  ggtitle("Petal Length vs Sepal Length")
# Comment: CLUSTERING CAN BE ACHIEVED. Very good diffrentiation

# Sepal width vs petal width
sepalPetalWidth <- iris %>% select(SepalWidthCm, PetalWidthCm, Species)
ggplot(sepalPetalWidth) + geom_point(aes(SepalWidthCm, PetalWidthCm, color = Species)) + 
  ggtitle("Petal Width vs Sepal Width")
# Comment: Clusters are apparent but cannot be used for classification

# Experimenting with violin plots for all variables
data_summary <- function(x)
{
  m <- mean(x)
  ymin <- m - sd(x)
  ymax <- m + sd(x)
  return (c(y = m, ymin = ymin, ymax = ymax))
}
violinPlot1 <- ggplot(iris, aes(Species, SepalLengthCm, fill = Species)) + geom_violin(trim = FALSE) +
  stat_summary(fun.data = data_summary)
violinPlot2 <- ggplot(iris, aes(Species, SepalWidthCm, fill = Species)) + geom_violin(trim = FALSE) +
  stat_summary(fun.data = data_summary)
violinPlot3 <- ggplot(iris, aes(Species, PetalLengthCm, fill = Species)) + geom_violin(trim = FALSE) +
  stat_summary(fun.data = data_summary)
violinPlot4 <- ggplot(iris, aes(Species, PetalWidthCm, fill = Species)) + geom_violin(trim = FALSE) +
  stat_summary(fun.data = data_summary)
grid.arrange(violinPlot1, violinPlot2, violinPlot3, violinPlot4, ncol = 2)

# Experimenting with marginal density plots
sp <- ggscatter(iris, x = "SepalLengthCm", y = "SepalWidthCm",
                color = "Species", palette = "jco",
                size = 3, alpha = 0.6)+
  border()                                         
xplot <- ggdensity(iris, "SepalLengthCm", fill = "Species",
                   palette = "jco")
yplot <- ggdensity(iris, "SepalWidthCm", fill = "Species", 
                   palette = "jco")+
  rotate()
yplot <- yplot + clean_theme() 
xplot <- xplot + clean_theme()
ggarrange(xplot, NULL, sp, yplot, 
          ncol = 2, nrow = 2,  align = "hv", 
          widths = c(2, 1), heights = c(1, 2),
          common.legend = TRUE)

# Using heat map to relate all the variables
flowerCorr <- iris %>% select(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
n <- cor(flowerCorr)
corrplot(n)
# This summarises the correlation between all variables. We can see that there is a strong
# positive correlation betweeen petal length and sepal length, and petal width and sepal length
# There is also a strong positive correlation between petal length and petal width

# MACHINE LEARNING:
# Will perform following and write exactly what the methods entail:
# Models: decision tree classifier, logistic regression, k-Nearest Neighbors, Random Forests, SVM
# Explore some more classification techniques

# Splitting dataset into test and train
set.seed(1)
index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
iris <- iris %>% select(-Id)
irisTrain <- iris[index,]
irisTest <- iris[-index,]

# Decision tree classifier https://www.r-bloggers.com/using-decision-trees-to-predict-infant-birth-weights/
# https://stackoverflow.com/questions/29572906/data-prediction-using-decision-tree-of-rpart
irisTree <- train(Species ~ ., irisTrain, method = "rpart")
decisionPredictions <- predict(irisTree, irisTest)

# Plotting decision tree
plot(irisTree$finalModel)
text(irisTree$finalModel)

# Finding accuracy
confusionMatrix(data = decisionPredictions, reference = irisTest$Species)  # Accuracy of 95.6%
# Do this based on selected features????

# Logistic regression:
multinomialFit <- train(Species ~ ., irisTrain, method = "multinom")
logitPrediction <- predict(multinomialFit, irisTest)
confusionMatrix(data = logitPrediction, reference = irisTest$Species) # Exactly the same as decision tree

# K-nearest neighbors
knnFit <- train(Species ~ ., irisTrain, method = "knn")
knnPrediction <- predict(knnFit, irisTest)
confusionMatrix(data = knnPrediction, reference = irisTest$Species) # WOW, accuracy went up to 98%

# Random forest
rfFit <- train(Species ~ ., irisTrain, method = "rf")
rfPrediction <- predict(rfFit, irisTest)
confusionMatrix(data = rfPrediction, reference = irisTest$Species) # Accuracy: 95.6%

# Support vector machine
svmFit <- train(Species~ ., irisTrain, method = "svmLinear")
svmPrediction <- predict(svmFit, irisTest)
confusionMatrix(data = svmPrediction, reference = irisTest$Species) # Accuracy: 98%

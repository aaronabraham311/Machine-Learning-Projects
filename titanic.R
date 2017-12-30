library(tidyverse)
library(mice)
library(randomForest)

train <- read.csv("train.csv")
test <- read.csv("test.csv")

full <- bind_rows(train, test)
str(full)

# Cleaning data

## Names: Status could tell us something
full$Title <- trimws(gsub('(.*, )|(\\..*)', '', full$Name)) # Extracts title from Name columns
table(full$Title, full$Sex)

rare_title <- c("Capt", "Col", "Don", "Dona", "Dr","Jonkheer","Lady", "Major", "Rev", "Sir", "the Countess") 

full$Title[full$Title == "Ms"] <- "Miss"
full$Title[full$Title == "Mlle"] <- "Miss"
full$Title[full$Title == "Mme"] <- "Mrs"
full$Title[full$Title %in% rare_title] <- "Rare"

table(full$Title, full$Sex)

full$Surname <- sapply(full$Name,  function(x) strsplit(x, split = '[,.]')[[1]][1])

## Families
full$FamilySize <- full$SibSp + full$Parch + 1 
full$Family <- paste(full$Surname, full$FamilySize, sep = "_" )

ggplot(full[1:891,], aes(x = FamilySize, fill = factor(Survived))) + 
  geom_bar(stat = "count", position = "dodge") +
  ggtitle("How did families fare on the Titanic?") +
  labs(x = "Family Size", y = "Number of Individuals") # Clearly, single people did not fare well

full$DiscretizedFamily[full$FamilySize == 1] <- "single"
full$DiscretizedFamily[full$FamilySize >= 2 || full$FamilySize < 5] <- "normal"
full$DiscretizedFamily[full$FamilySize >= 5] <- "large"

# Missingness

## Sensible value imputation
embark_fare <- full %>% filter(PassengerId != 62 & PassengerId != 830) # Passengers 62 and 830 do not have embarked value
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept = 80 ), color = "red", linetype = "dashed", lwd = 2)  # Fare is most likely from Charbourg

full$Embarked[c(62, 830)] <- "C"

### Passenger 1044 has no fare value. Embarked from Southampton and was in class 3
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)), colour='red', linetype='dashed', lwd=1) # Fill value of passenger with median
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare)

## Predictive imputation
vars <- c('PassengerId','Pclass','Sex','Embarked',
          'Title','Surname','Family','DiscretizedFamily')
full[vars] <- lapply(full[vars], function(x) as.factor(x))

mice_model <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method = "rf")
mice_output <- complete(mice_model)

### Plotting age distributions
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04)) # Distributions are very similar

full$Age <- mice_output$Age

# More feature engineering
full$Demographic[full$Age < 18] <- "Child"
full$Demographic[full$Age >= 18 & full$Age < 35] <- "Young Adult"
full$Demographic[full$Age >= 35 & full$Age < 60] <- "Adult"
full$Demographic[full$Age >= 60] <- "Senior"

table(full$Demographic, full$Survived) # Demographic variable seems to be important

full$Parch <- as.numeric(full$Parch)

full$Parent <- "No Parent"
full$Parent[full$Sex == 'male' & full$Parch > 0 & full$Age > 18] <- "Father"
full$Parent[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- "Mother"

# INSERT VISUALIZATION FOR DEMOGRAPHIC AND PARENT VARIABLE
full$Parent <- as.factor(full$Parent)
full$Demographic <- as.factor(full$Demographic)

# PREDICTION:
train <- full[1:891,]
test <- full[892:1309,]

set.seed(1)

rf_model <- randomForest(data = train, factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + DiscretizedFamily + Demographic + Parent)
plot(rf_model, ylim = c(0.0, 0.3))

importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

# Predict using the test set
prediction <- predict(rf_model, test)
id <- test$PassengerId

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = id, Survived = prediction, row.names = NULL)

# Write the solution to file
write.csv(solution, file = 'solution.csv', row.names = F)

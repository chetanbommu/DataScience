## Logistic Regression

## Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

## Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
train_data = subset(dataset, split == TRUE)
test_data = subset(dataset, split == FALSE)
  
## Feature Scaling
train_data[, 1:2] = scale(train_data[, 1:2])
test_data[, 1:2] = scale(test_data[, 1:2])

## Fitting Logistic Regression to the training set
classifier = glm(formula = Purchased~., data = train_data, family = binomial)

## Predict the test set results
probability_prediction = predict(classifier, newdata = test_data, type = 'response')
y_pred = ifelse(probability_prediction > 0.5, 1, 0)

## confusion matrix
confusion_matrix = table(test_data$Purchased, y_pred, dnn = c('actual','predicted'))

## Visualise the training set results
install.packages('ElemStatLearn')

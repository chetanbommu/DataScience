## loading data
library(MASS)
data("Boston")
?Boston

boston_data = Boston
## Check summary
summary(boston_data)
## Verify all columns are numeric
str(boston_data)

## check correlation
View(cor(boston_data))

## Plot correlation and remove highly correlated columns.
library(corrplot)
corrplot(cor(boston_data), method = "number")

## Scaling => Min-Max
min_max_scaling_fn = function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

## Apply Scaling
for(i in 1:ncol(boston_data)){
  boston_data[,i] = min_max_scaling_fn(boston_data[,i])
}

## Verify Data Scaled
summary(boston_data)

## Identify Best number of clusters
withinByBetween = c()
for(i in 2:15){
  clust = kmeans(x=boston_data,centers = i)
  withinByBetween = c(withinByBetween, mean(clust$withinss)/clust$betweenss)
}
plot(2:15,withinByBetween,type = 'l')

## K-Means Clustering => Always choose minimum number of clusters
k_means_cluster = kmeans(x=boston_data, centers = 9)
k_means_cluster$centers

## Assign cluster number to each row 
boston_data$cluster = k_means_cluster$cluster


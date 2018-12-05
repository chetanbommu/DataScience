"""
Created on Sat Nov 24 19:45:24 2018

@author: chetanbommu
"""
## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

## Splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

## Feature Scaling => Some algorithms doesn't need feature scaling, since algorithm itself does feature scaling for us
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

## Fitting Simple Linear Regresion to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() ## No Compulsory Parameters
regressor.fit(X_train, y_train)

## Predicting the test set results
y_pred = regressor.predict(X_test)

## Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Train set)")
plt.xlabel("Experience in years")
plt.ylabel("Salary in $")
plt.show()

## Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Experience in years")
plt.ylabel("Salary in $")
plt.show()
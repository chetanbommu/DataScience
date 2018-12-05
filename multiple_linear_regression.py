"""
@author: chetanbommu
"""
## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


## Encoding Categorical data
## Label Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
## One Hot Encoding
onehotencoder = OneHotEncoder(categorical_features=[3]) 
        ## O indicates first column
        ## OneHotEncoder takes array of indices for categorical_features
X = onehotencoder.fit_transform(X).toarray()


## Avoid Dummy Variable Trap
X = X[:,1:] ## We need not do this, library takes care of it.


## Splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

## Fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Predicting the test set results
y_pred = regressor.predict(X_test)


## Building the optimal model using Backward Elimination to remove non-statistical important columns
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
    ## result is arr + values
    ## columns of 1's(intercept) are added so that model takes as x0=1
    ## np.ones creates an array, so convert to int
    ## axis = 0 => row
    ## axis = 1 => columns

## Step 1: Select a significance level to stay in the model SL= 0.05
## Step 2: Fit the Full model with all possible predictors
X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()

## Step 3: Consider the predictor with highest p-value. if P>SL => step - 4, otherwise go to FIN
regressor_ols.summary() ## x2 has highest p-value of 0.990

## Step 4: Remove the predictor
X_opt = X[:,[0,1,3,4,5]]

## continue with step-2
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary() ## x1 => 0.940

X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary() ## x2 => 0.602

X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary() ## x2 => 0.060

X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary() ## Every predictor has p-value < 0.05

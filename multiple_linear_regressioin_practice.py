#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:28:03 2019

@author: mattNest
"""

# Simple Linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # X is a matrix (independent variable)
y = dataset.iloc[:, 4].values # y is a vector (dependent variable)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Encoding X categorical data + HotEncoding
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float) # X is now float64, not object type anymore


# Avoid the Dummy Variable Trap
X = X[:,1:] # remove one column


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting the Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # create an regressor object
regressor.fit(X_train,y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

for i in range(10):
    diff = abs(y_pred[i]-y_test[i])
    print(diff)
    

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# Step 1: sl = 0.05
# Step 2: Fit the full model with all possible predictors
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Step 3: Consider the predictor with the highest p-value, if p > sl, go to step4
# Step 4: Remove the predictor
# Step 5: Fit the model without this variable
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# NOTES1: Backward elimination with p-values only
# =============================================================================
# import statsmodels.formula.api as sm
# def backwardElimination(x, sl):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar > sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = np.delete(x, j, 1)
#     regressor_OLS.summary()
#     return x
#  
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)
# 
# =============================================================================


# Backward Elimination with p-values and Adjusted R Squared:
# =============================================================================
# import statsmodels.formula.api as sm
# def backwardElimination(x, SL):
#     numVars = len(x[0])
#     temp = np.zeros((50,6)).astype(int)
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         adjR_before = regressor_OLS.rsquared_adj.astype(float)
#         if maxVar > SL:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     temp[:,j] = x[:, j]
#                     x = np.delete(x, j, 1)
#                     tmp_regressor = sm.OLS(y, x).fit()
#                     adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                     if (adjR_before >= adjR_after):
#                         x_rollback = np.hstack((x, temp[:,[0,j]]))
#                         x_rollback = np.delete(x_rollback, j, 1)
#                         print (regressor_OLS.summary())
#                         return x_rollback
#                     else:
#                         continue
#     regressor_OLS.summary()
#     return x
#  
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)
# =============================================================================








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:03:31 2019

@author: mattNest
"""

# Random Forest Regression
# A team of Decision Trees


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # X is a matrix (independent variable) # write 1:2 to turn X as a matrix
y = dataset.iloc[:, 2].values # y is a vector (dependent variable)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting Random Forest Regression to the dataset
# Create your Regressor here
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0) # n_estimators = # of trees
regressor.fit(X, y)


# Predicting the a new result
y_pred = regressor.predict(np.array(6.5).reshape(1,-1))
print(abs(y_pred - 160000))


# Visualizing the Decision Tree Regression results
X_grid = np.arange(min(X),max(X),0.01) # for higher resolution and smoother curve
X_grid = X_grid.reshape((len(X_grid)),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Foreset Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




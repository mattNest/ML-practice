#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:50:02 2019

@author: mattNest
"""

# Polynomial Regression
# Data Preprocessing Template

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


# Fitting Lineaer Regressioin to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # create lin_reg object
lin_reg.fit(X,y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X,y) # produce the polynomial terms


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y) # fit to the regressioin



# Visualizing the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
lin_output = lin_reg.predict(np.array(6.5).reshape(1,-1))

# Predicting a new result with Polynomial Regression
poly_output = lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1)))








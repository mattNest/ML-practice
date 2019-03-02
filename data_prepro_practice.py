#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 22:06:45 2019

@author: mattNest
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :-1].values # independent variable
Y = dataset.iloc[:, 3].values # dependent variable -- Purchased


# Taking care the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =============================================================================
# Bad way for encoding countries, cuz countries don't have priority
# labelencoder_X = LabelEncoder() 
# X[:, 0] = labelencoder_X.fit_transform(X[:,0])
# =============================================================================

# =============================================================================
# Dummy Variable Encoding (Old method)
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray() 
# =============================================================================

# Encoding X categorical data + HotEncoding
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding Y data
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into Training set and Test set
# =============================================================================
# Make sure that the model doesn't learn by heart
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Feature Scaling
# =============================================================================
# 
# Standardisation xstand= (x-mean(x)) / std(x)
# Normalisation xnorm = (x - min(x)) / max(x) - min(x)
# 
# =============================================================================

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)




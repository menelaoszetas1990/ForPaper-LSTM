# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.neural_network import MLPRegressor
import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import scipy.io
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

WHICH_CAPE = 2

# Preprocessing

# Importing the dataset
dataset = pd.read_csv(
    'D:\\Users\\user\\Desktop\\Maritime\\Hydrus_data\\ME_POWER_PREDICTION\\cape' + str(WHICH_CAPE) + '.csv')
X = dataset.iloc[:, [0, 1, 4, 5, 8]].values
y = dataset.iloc[:, 9].values
y = y.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Feature Scaling
sc1 = MinMaxScaler()
X_train_scaled = sc1.fit_transform(X_train[:, :])
X_test_scaled = sc1.transform(X_test[:, :])
sc2 = MinMaxScaler()
y_train_scaled = sc2.fit_transform(y_train[:, :])
y_test_scaled = sc2.transform(y_test[:, :])

# Training the model on the Training set

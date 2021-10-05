import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#Set n value as to what row onwards you want to predict
n1 = 3360
n = 1680

#Just change dataset path
dataset = pd.read_excel(r"Final Dataset Bajra.xlsx")[:n1]

#Divide into input and output data
X = dataset.drop(['CROP', 'PRODUCTION', 'TEMP'], axis=1)
y = dataset['PRODUCTION']

#Encode string to numerical value because model requires so
encoder = LabelEncoder().fit(X['STATE'])
X['STATE'] = encoder.fit_transform(X['STATE'])

#Divide into train and test data
X_train = X[:n]
X_test = X[n:]
y_train = y[:n]
y_test = y[n:]

model = LinearRegression()

#Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

#Reverse numerical data to string data
X['STATE'] = encoder.inverse_transform(X['STATE'])
X_test = X[n:]

y_pred = pd.DataFrame(y_pred, columns=["PREDICTION"])
X_test = X_test.reset_index(drop=True)
dataset = pd.merge(X_test, y_pred, left_index=True, right_index=True)


dataset.to_excel("Predict Bajra.xlsx")

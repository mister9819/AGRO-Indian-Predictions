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
n1 = 1680
n = 1440

#Just change dataset path
dataset = pd.read_excel(r"Final Dataset Jowar.xlsx")[:n1]
print(dataset.describe(), "\n")

#Divide into input and output data
X = dataset.drop(['CROP', 'PRODUCTION', 'TEMP', 'RAINFALL', 'CULTIVATION'], axis=1)
y = dataset['PRODUCTION']

#Encode string to numerical value because model requires so
encoder = LabelEncoder().fit(X['STATE'])
X['STATE'] = encoder.fit_transform(X['STATE'])

#Divide into train and test data
X_train = X[:n]
X_test = X[n:]
y_train = y[:n]
y_test = y[n:]

model = KNeighborsRegressor(5 ,weights='uniform')

df, score, mae, mse, rmse, y_pred = 0, 0, 0, 0, 0, 0

#Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

#Score model
score = model.score(X_test, y_test)

#Miscellaneous data
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#Reverse numerical data to string data
X['STATE'] = encoder.inverse_transform(X['STATE'])
X_test = X[n:]

#Printing stuff
df = pd.DataFrame({'State':X_test.iloc[:, 1], 'Actual':y_test, 'Predicted':y_pred})
print(df, "\n")
print('Score:', score)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

#Making graph
plt.scatter(X_test.iloc[:, 1][:20], y_test[:20], color='red', label='Actual observation points')
plt.plot(X_test.iloc[:, 1][:20], y_pred[:20], label='KNN5')
plt.title('Jowar January 2015 actual vs predictions')
plt.xlabel('State')
plt.ylabel('Production')
plt.xticks(rotation=90)
plt.tight_layout()

#Display graph
plt.legend()
plt.show()

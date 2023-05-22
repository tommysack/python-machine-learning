import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_diabetes

#Load data
diabets = load_diabetes(as_frame=True)

#General info
print(diabets.DESCR)

'''
The data are points in an hyperspace of 11 dimensions.
The objective is to find the best hyperplane, then the "best" linear function f, in hyperspace.
In this case we use LinearRegression that use a Closed-Form solution trying to minimize the sum of squared residuals OLS.
'''

#Separates data in Dataframe/Series columns data/target 
X = diabets.data 
Y = diabets.target 

#Check no null 
print(X.isnull().sum())
print(Y.isnull().sum())
#Check only numbers
np.isnan(X).any() 
#Check columns correlated
X.corr() # Only s1 and s2 > 0.85

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#X before scaling
print("\nX min before scaling")
print("Min train", np.amin(X_train))
print("Min test", np.amin(X_test))
print("\nX max before scaling")
print("Max train", np.amax(X_train))
print("Max train", np.amax(X_test))

#Transform features by scaling each feature to a given range (0,1)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

#X after scaling
print("\nX min after scaling")
print("Min train", np.amin(X_train))
print("Min test", np.amin(X_test))
print("\nX max after scaling")
print("Max train", np.amax(X_train))
print("Max test", np.amax(X_test))

linear_regression = LinearRegression() #LinearRegression uses Closed-Form/OLS
linear_regression.fit(X_train, Y_train) #Training the model
Y_test_predicted = linear_regression.predict(X_test) #Predict Y_test from X_test

#Model evaluation
print("\nModel evaluation")
print("MAE: ", mean_absolute_error(Y_test, Y_test_predicted))
print("MSE: ", mean_squared_error(Y_test, Y_test_predicted))
print("R2 SCORE: ", r2_score(Y_test, Y_test_predicted))







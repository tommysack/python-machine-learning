import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_diabetes

#Load data
diabets = load_diabetes(as_frame=True)

#General info
print(diabets.DESCR)

'''
The data are points in an hyperspace H of 11 dimensions.
The goal is to predict the value of the target column Y from the columns X as well as possible. 
Technically you need to find the "best" hyperplane, then the linear function f (weights and biases), in H.
In this case we use LinearRegression that use a Closed-Form solution trying to minimize the sum of squared residuals OLS.
'''

#Separates data in Dataframe/Series columns data/target 
X = diabets.data 
Y = diabets.target 

#Check columns correlated (corr function works only for numbers)
np.isnan(X).any() #Check only numbers
X.corr() # Only s1 and s2 > 0.85

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Check if X needs to scaling (makes it easy for a model to learn and understand the problem)
print("\nBEFORE scaling")
print("X train min", np.amin(X_train))
print("X test min", np.amin(X_test))
print("X train max", np.amax(X_train))
print("X test max", np.amax(X_test))

#Standardize features (preferred vs. MinMaxScaler, the features upper/lower boundaries aren't known)
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

#X after scaling
print("\nAFTER scaling")
print("X train min", np.amin(X_train))
print("X test min", np.amin(X_test))
print("X train max", np.amax(X_train))
print("X test max", np.amax(X_test))

linear_regression = LinearRegression() #LinearRegression uses Closed-Form/OLS
linear_regression.fit(X_train, Y_train) #Building the model

Y_test_predicted = linear_regression.predict(X_test) #Predict Y_test from X_test

#Model evaluation (distances from real data, and model performance)
print("\nModel evaluation")
print("MAE: ", mean_absolute_error(Y_test, Y_test_predicted))
print("MSE: ", mean_squared_error(Y_test, Y_test_predicted))
print("R2 SCORE: ", r2_score(Y_test, Y_test_predicted)) #R2=ESS/TSS, best possible score is 1.0







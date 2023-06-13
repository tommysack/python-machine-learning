import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_diabetes

#Load data
diabetes = load_diabetes()

#General info
print(diabetes.DESCR) #10 patients variables X and 1 quantitative Y measure of disease progression one year after
diabetes_df = pd.DataFrame(diabetes.data, columns=["age","sex","bmi","bp","tc","ldl","hdl","tch","ltg","glu"])
diabetes_df['progression'] = diabetes.target
diabetes_df.head()
diabetes_df.describe() 
diabetes_df.shape #11 columns, 442 rows
diabetes_df.isnull().sum() 
np.isnan(diabetes_df).any() #Many algorithms do work only with numerical data

#Correlation between features and target (we assume moderate correlation from 0.5)
diabetes_df.corr()['progression'].sort_values() #Moderate correlation with ltg and bmi 

#Try to draw the correlation and the linear regression model fit
plt.figure(figsize=(6, 6))
sns.regplot(data=diabetes_df, x='ltg', y='progression', color='yellow', line_kws={"color": "red"})
plt.title('Correlation and Linear Regression between ltg and progression')
plt.xlabel('Ltg')
plt.ylabel('Progression')

plt.figure(figsize=(6, 6))
sns.regplot(data=diabetes_df, x='bmi', y='progression', color='yellow', line_kws={"color": "red"})
plt.title('Correlation and Linear Regression between bmi and progression')
plt.xlabel('Bmi')
plt.ylabel('Progression')

'''
The data points are too far from regression lines. 
Anyway I would try with Linear Regression and all features.
'''

'''
The data are points in an hyperspace H of 11 dimensions.
The goal is to predict the value of the target column Y from the columns X as well as possible. 
Technically you need to find the "best" hyperplane of 10 dimensions, then the linear function f (weights and biases), in H.
The "best": in this case we use LinearRegression that use a Closed-Form solution trying to minimize the sum of squared residuals OLS.
'''
#Separates data in Dataframe/Series columns data/target 
X = diabetes.data 
Y = diabetes.target

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

Y_train_predicted = linear_regression.predict(X_train) 

#Model overfitting evaluation
print("\nModel overfitting evaluation")
print("MAE: ", mean_absolute_error(Y_train, Y_train_predicted))
print("MSE: ", mean_squared_error(Y_train, Y_train_predicted))
print("R2 SCORE: ", r2_score(Y_train, Y_train_predicted)) #R2=ESS/TSS, best possible score is 1.0

Y_test_predicted = linear_regression.predict(X_test) 

#Model evaluation 
print("\nModel evaluation")
print("MAE: ", mean_absolute_error(Y_test, Y_test_predicted))
print("MSE: ", mean_squared_error(Y_test, Y_test_predicted))
print("R2 SCORE: ", r2_score(Y_test, Y_test_predicted)) #R2=ESS/TSS, best possible score is 1.0

'''
R2 score in training is much higher than test.
The model would appear to be inappropriate for this problem.
'''


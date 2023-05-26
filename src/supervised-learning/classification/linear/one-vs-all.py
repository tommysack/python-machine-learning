import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

#Load data
digits = load_digits()

#General info
print(digits.DESCR) #Images 8x8 64 columns of pixels (every pixel has a value 0..16), and 1 value integer rapresented
digits_df = pd.DataFrame(digits.data)
digits_df['target'] = digits.target
digits_df.head()
digits_df.describe() 
digits_df.shape #65 columns, 1797 rows

'''
The data are points in an hyperspace H of 65 dimensions.
The goal is to assign a class label Y (classification with values 0..9) to input X.
Technically you need to find the "best" hyperplane, then the linear function f (weights and biases), in H.
In this case we use LinearRegression that use a Closed-Form solution trying to minimize the sum of squared residuals OLS.
'''

#Separates data in Dataframe/Series columns data/target 
X = digits.data 
Y = digits.target 

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Check if X needs to scaling (makes it easy for a model to learn and understand the problem)
print("\nBEFORE scaling")
print("X train min", np.amin(X_train))
print("X test min", np.amin(X_test))
print("X train max", np.amax(X_train))
print("X test max", np.amax(X_test))

#Normalize features (preferred vs. StandardScaler, the features upper/lower boundaries are known)
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

#X after scaling
print("\nAFTER scaling")
print("X train min", np.amin(X_train))
print("X test min", np.amin(X_test))
print("X train max", np.amax(X_train))
print("X test max", np.amax(X_test))

logistic_regression = LogisticRegression() #solver='lbfgs'
logistic_regression.fit(X_train, Y_train) #Building the model
Y_train_predicted = logistic_regression.predict(X_train) #To calculate model's overfitting

Y_test_predicted = logistic_regression.predict(X_test) #Predict Y_test from X_test
Y_test_predicted_proba = logistic_regression.predict_proba(X_test) #To calculate LOG LOSS

#Model evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) #Best possible score is 1.0
print("ACCURACY \"overfitting\" SCORE: ", accuracy_score(Y_train, Y_train_predicted)) #Best possible score is 1.0
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) #Best possible score is 0



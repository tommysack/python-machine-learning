import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_diabetes

#Load data
diabets = load_diabetes(as_frame=True) #10 patients variables X and 1 quantitative Y measure of disease progression one year after

#General info
print(diabets.DESCR) #10 patients variables X and 1 quantitative Y measure of disease progression one year after
diabets_df = pd.DataFrame(diabets.data)
diabets_df['target'] = diabets.target
diabets_df.head()
diabets_df.describe() 
diabets_df.shape #11 columns, 442 rows

#Correlation between data/target (corr function works only for numbers)
diabets_df.isnull().sum() 
np.isnan(diabets_df).any() 
diabets_df.corr()['target'].sort_values() #No special correlation with target

'''
The data are points in an hyperspace H of 11 dimensions.
The goal is to predict the value of the target column Y from the columns X as well as possible. 
Technically you need to find the "best" curved surface, then the best polynomial function f (weights and biases), in the H.
In this case we use LinearRegression that use a Closed-Form solution trying to minimize the sum of squared residuals OLS.
'''

#Separates data in Dataframe/Series columns data/target 
X = diabets.data 
Y = diabets.target 

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Check if X needs to scaling
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

degree_best = 0
r2_score_best = 0

for i in range (1, 11):
  
  print(f"\nDEGREE: {i}")

  linear_regression = LinearRegression() #LinearRegression uses Closed-Form/OLS
  pf = PolynomialFeatures(degree=i)

  X_train_poly = pf.fit_transform(X_train)
  linear_regression.fit(X_train_poly, Y_train) #Building the model using X_train_poly
  Y_train_predicted = linear_regression.predict(X_train_poly) #To calculate model's overfitting

  X_test_poly = pf.transform(X_test)
  Y_test_predicted = linear_regression.predict(X_test_poly) #Predict usign X_test_poly

  #Model evaluation (distances from real data, and model performance)
  print("\nModel evaluation")
  print("MAE: ", mean_absolute_error(Y_test, Y_test_predicted))
  print("MSE: ", mean_squared_error(Y_test, Y_test_predicted))
  r2_score_current = r2_score(Y_test, Y_test_predicted)
  print("R2 SCORE: ", r2_score_current) #R2=ESS/TSS, best possible score is 1.0
  print("R2 \"overfitting\" SCORE: ", r2_score(Y_train, Y_train_predicted)) #R2=ESS/TSS, best possible score is 1.0

  if r2_score_current > r2_score_best:
    degree_best = i
    r2_score_best = r2_score_current   
    
'''
With polynomial function of degree >= 2 the model is worse than with linear function
'''






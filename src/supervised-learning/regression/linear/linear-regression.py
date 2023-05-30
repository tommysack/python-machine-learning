import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_diabetes

#Load data
diabets = load_diabetes()

#General info
print(diabets.DESCR) #10 patients variables X and 1 quantitative Y measure of disease progression one year after
diabets_df = pd.DataFrame(diabets.data, columns=["age","sex","bmi","bp","tc","ldl","hdl","tch","ltg","glu"])
diabets_df['progression'] = diabets.target
diabets_df.head()
diabets_df.describe() 
diabets_df.shape #11 columns, 442 rows

#Correlation between data/target (corr function works only for numbers)

diabets_df.isnull().sum() 
np.isnan(diabets_df).any() 
diabets_df.corr()['progression'].sort_values() #Moderate correlation with ltg and bmi, try to draw linear regression model fit

plt.figure(figsize=(6, 6))
sns.regplot(data=diabets_df, x='ltg', y='progression', color='yellow', line_kws={"color": "red"})
plt.title('Correlation and Linear Regression between ltg and progression')
plt.xlabel('Ltg')
plt.ylabel('Progression')

plt.figure(figsize=(6, 6))
sns.regplot(data=diabets_df, x='bmi', y='progression', color='yellow', line_kws={"color": "red"})
plt.title('Correlation and Linear Regression between bmi and progression')
plt.xlabel('Bmi')
plt.ylabel('Progression')

'''
The data points are too far from regression lines. 
Anyway I would try with Linear Regression model, since there is a weak correlation with progression.

The data are points in an hyperspace H of 11 dimensions.
The goal is to predict the value of the target column Y from the columns X as well as possible. 
Technically you need to find the "best" hyperplane of 10 dimensions, then the linear function f (weights and biases), in H.
In this case we use LinearRegression that use a Closed-Form solution trying to minimize the sum of squared residuals OLS.
'''
#Separates data in Dataframe/Series columns data/target 
X = diabets.data 
Y = diabets.target

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

Y_train_predicted = linear_regression.predict(X_train) #To calculate model's overfitting

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
R2 score in training is higher than test, than it's probably a case of overfitting.
Try to use k-fold cross validation.
'''

print("\nk-fold cross validation..")

linear_regression_kfold = LinearRegression()

kfold = KFold(n_splits=10) #Splits X_train in n_splits folders, and every of them is used to process training/test
scores_kfold = []

for fold_number, (train, test) in enumerate(kfold.split(X_train)):

  linear_regression_kfold.fit(X_train[train], Y_train[train]) #Building the model
  score_kfold = linear_regression_kfold.score(X_train[test], Y_train[test])
  scores_kfold.append(score_kfold)

  print("\nFOLD =", fold_number)
  print("LINEAR REGRESSION SCORE: ", score_kfold)

final_score = np.array(scores_kfold).mean()
print("\nFINAL SCORE: ", final_score)

Y_train_predicted = linear_regression_kfold.predict(X_train)

#Model overfitting evaluation
print("\nModel overfitting evaluation")
print("MAE: ", mean_absolute_error(Y_train, Y_train_predicted))
print("MSE: ", mean_squared_error(Y_train, Y_train_predicted))
print("R2 SCORE: ", r2_score(Y_train, Y_train_predicted)) #R2=ESS/TSS, best possible score is 1.0

Y_test_predicted = linear_regression_kfold.predict(X_test)

#Model evaluation (distances from real data, and model performance)
print("\nModel evaluation")
print("MAE: ", mean_absolute_error(Y_test, Y_test_predicted))
print("MSE: ", mean_squared_error(Y_test, Y_test_predicted))
print("R2 SCORE: ", r2_score(Y_test, Y_test_predicted)) #R2=ESS/TSS, best possible score is 1.0

'''
Also with k-fold the situation has not improved:
R2 score in test is too bad, and in training is higher than test.
Linear Regression isn't appropriate for this case. 
'''

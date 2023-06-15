import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

#Load data
breast_cancer = load_breast_cancer()

#General info
print(breast_cancer.DESCR) #Cancers 31 columns of informations, e 1 column with the diagnosis
breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['diagnosis'] = breast_cancer.target 
breast_cancer_df['diagnosis'].unique() #array([0, 1]) => binary classification
breast_cancer_df.head()
breast_cancer_df.describe() 
breast_cancer_df.shape #31 columns, 569 rows
sns.countplot(data=breast_cancer_df, x='diagnosis') #Ok, the classes are quite distributed
breast_cancer_df.isnull().sum() 
np.isnan(breast_cancer_df.drop('diagnosis',axis=1)).any() #Many algorithms do work only with numerical data

#Correlation between data/target (we assume moderate correlation from 0.5)
breast_cancer_df.corr()['diagnosis'].sort_values() #There are many features mildly correlated with target

'''
I would try with Logistic Regression and all features.
'''

'''
The data are points in an hyperspace H of 32 dimensions.
The goal is to assign a class label Y (binary classification with values "M" or "B") to input X.
Technically you need to find the "best" hyperplane of 31 dimensions which best separates the points classified in H.
The "best": in this case we use LogisticRegression that use "solver" to maximize Likelihood.
It make a linear binary (but is possible multi-class one vs all) probabilistic (returns the probability between 0 and 1 that a point belongs to a class using sigmoid function) 
classifier. 
'''

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer.data
Y = breast_cancer.target

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

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

logistic_regression = LogisticRegression(penalty='l2', C=10.0, solver='lbfgs') #l2 regularisation to avoid overfitting, C inverse of regularization strength
logistic_regression.fit(X_train, Y_train) 

Y_train_predicted = logistic_regression.predict(X_train) 
Y_train_predicted_proba = logistic_regression.predict_proba(X_train) 

#Model overfitting evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
print("\nModel overfitting evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted)) #Best possible score is 1.0
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) #Best possible score is 0

Y_test_predicted = logistic_regression.predict(X_test) 
Y_test_predicted_proba = logistic_regression.predict_proba(X_test)

#Model evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) #Best possible score is 1.0
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) #Best possible score is 0

'''
The model would appear to be appropriate for this problem.
'''


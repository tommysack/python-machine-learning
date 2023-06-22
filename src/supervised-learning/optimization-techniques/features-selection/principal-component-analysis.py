import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
PCA is unsupervised learning technique to use when your data has linear relationships between features.

PROS: it converts high dimensional data to low dimensional data, then it removes correlated features, prevents overfitting, 
and improves the performance.
CONS: the original features are transformed into principal components that are linear combinations of the features. 
Than it may become more difficult to understand what they are the most significant features after PCA.
'''

#Load data
wines = load_wine()

#General info
print(wines.DESCR) #Wines 13 columns of informations, and 1 value that rapresent a class 
wines_df = pd.DataFrame(wines.data, columns=wines.feature_names)
wines_df['target'] = wines.target
wines_df['target'].unique() #array([0, 1, 2]) => multi-class classification
wines_df.head()
wines_df.describe() 
wines_df.shape #14 columns, 178 rows
wines_df.isnull().sum() 
sns.countplot(data=wines_df, x='target') #Ok, the classes are quite distributed
np.isnan(wines_df).any() #Many algorithms do work only with numerical data

#Separates data in numpy.ndarray columns data/target 
X = wines.data
Y = wines.target

#Check if X needs to scaling (makes it easy for a model to learn and understand the problem)
print("\nBEFORE scaling")
print("X min", np.amin(X))
print("X max", np.amax(X))

#Standardize features (preferred vs. MinMaxScaler, the features upper/lower boundaries aren't known)
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

print("\nAFTER scaling")
print("X min", np.amin(X))
print("X max", np.amax(X))

pca = PCA(n_components=2) #Fixed numbers of principal components
X_pca = pca.fit_transform(X) 

print("\nX number of culumns:", X.shape[1])
print("X_pca number of culumns:", X_pca.shape[1])

#Let's build the model WITH PCA

#Separates data in rows train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=0)

logistic_regression = LogisticRegression(
  penalty='l2', #L2 regularization to avoid overfitting 
  C=0.1, #inverse of regularization strength (C lower => Higher regularization)
  solver='lbfgs', #algorithm to use
  verbose=True
) 
logistic_regression.fit(X_train, Y_train) 

Y_train_predicted = logistic_regression.predict(X_train) 
Y_train_predicted_proba = logistic_regression.predict_proba(X_train) 

#Model overfitting evaluation 
print("\n----------------WITH PCA----------------")
print("\nModel overfitting evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted)) 
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) 

Y_test_predicted = logistic_regression.predict(X_test) 
Y_test_predicted_proba = logistic_regression.predict_proba(X_test) 

#Model evaluation
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) 
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) 
print("\n----------------------------------------")

#Let's build the model WITHOUT PCA

#Separates data in rows train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

logistic_regression = LogisticRegression(
  penalty='l2', #L2 regularization to avoid overfitting 
  C=0.1, #inverse of regularization strength (C lower => Higher regularization)
  solver='lbfgs', #algorithm to use
  verbose=True
) 
logistic_regression.fit(X_train, Y_train)

Y_train_predicted = logistic_regression.predict(X_train) 
Y_train_predicted_proba = logistic_regression.predict_proba(X_train) 

#Model overfitting evaluation
print("\n----------------WITHOUT PCA----------------")
print("\nModel overfitting evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted)) 
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) 

Y_test_predicted = logistic_regression.predict(X_test) 
Y_test_predicted_proba = logistic_regression.predict_proba(X_test) 

#Model evaluation
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) 
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) 
print("\n-------------------------------------------")

'''
The model with PCA appear to be more appropriate for this problem rather than without PCA.
'''

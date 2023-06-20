import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'''
Kernel PCA is an extension of PCA used for nonlinear dimensionality reduction.
'''

#Creating data: a large circle containing a smaller circle with 1000 points
X, Y = make_circles(n_samples=1000, noise=0.1, factor=0.2)

#General info
print("Classes in Y:", np.unique(Y, return_counts=True)) #500 classes => multi-class classification

#Check if X needs to scaling (makes it easy for a model to learn and understand the problem)
print("\nBEFORE scaling")
print("X min", np.amin(X))
print("X max", np.amax(X))

#No scaling is needs

kernel_pca = KernelPCA(kernel="rbf", gamma=5) #We use "rbf" when the dataset has a circular shape (non-linear)
X_kpca = kernel_pca.fit_transform(X)

print("\nX number of culumns:", X.shape[1])
print("X_pca number of culumns:", X_kpca.shape[1])

#Let's build the model WITH KPCA

#Separates data in rows train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X_kpca, Y, test_size=0.3, random_state=0)

logistic_regression = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs') #l2 regularisation to avoid overfitting, C inverse of regularization strength
logistic_regression.fit(X_train, Y_train) 

Y_train_predicted = logistic_regression.predict(X_train) 
Y_train_predicted_proba = logistic_regression.predict_proba(X_train) 

#Model overfitting evaluation 
print("\n----------------WITH KPCA----------------")
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
  C=10.0, #inverse of regularization strength (C lower => Higher regularization)
  solver='lbfgs', #algorithm to use
  verbose=True
) 
logistic_regression.fit(X_train, Y_train)

Y_train_predicted = logistic_regression.predict(X_train) 
Y_train_predicted_proba = logistic_regression.predict_proba(X_train) 

#Model overfitting evaluation 
print("\n----------------WITHOUT KPCA----------------")
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
The model with KPCA appear to be more appropriate for this problem rather than without KPCA.
'''






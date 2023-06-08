import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

#Load data
breast_cancer = load_breast_cancer()

#General info
print(breast_cancer.DESCR)
breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['diagnosis'] = breast_cancer.target 
breast_cancer_df.head()
breast_cancer_df.describe() 
breast_cancer_df.shape #32 columns, 569 rows
breast_cancer_df['diagnosis'].unique() #array([0, 1]) => binary classification
sns.countplot(data=breast_cancer_df, x='diagnosis') #Ok, the classes are quite distributed
breast_cancer_df.isnull().sum() 
np.isnan(breast_cancer_df.drop('diagnosis',axis=1)).any() #Many algorithms do work only with numerical data

#Correlation between data/target (corr function works only for numbers)
breast_cancer_df.corr()['diagnosis'].sort_values() #Only mean fractal dimension and smoothness error
#Correlation between "concave points_worst" and "perimeter_worst" to exclude duplicate feature 
breast_cancer_df.corr()['mean fractal dimension'].sort_values() 
breast_cancer_df.corr()['smoothness error'].sort_values() 

plt.figure(figsize=(6, 6))
sns.regplot(data=breast_cancer_df, x='mean fractal dimension', y='diagnosis', logistic=True)
plt.title('Correlation between mean fractal dimension and diagnosis')
plt.xlabel('mean fractal dimension')
plt.ylabel('diagnosis')
plt.show()

plt.figure(figsize=(6, 6))
sns.regplot(data=breast_cancer_df, x='smoothness error', y='diagnosis', logistic=True)
plt.title('Correlation between smoothness error and diagnosis')
plt.xlabel('smoothness error')
plt.ylabel('diagnosis')
plt.show()

'''
The data are points in an hyperspace H of 32 dimensions.
The goal is to assign a class label Y (binary classification with values "M" or "B") to input X.
Technically you need to find the "best" hypercurve of 31 dimensions which best separates the points classified in H.
In this case we use K-NN classification that assign a class to a point based on the classes of k neighboring points
(very low k => overfitting). In training phase K-NN does not learn any model ("lazy") but it only stores the points, 
and the predictions are made just-in-time by calculating the similarity. 
It make a non-probabilistic binary non-linear classifier. 
'''

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer_df.drop("diagnosis", axis=1).values 
Y = breast_cancer_df["diagnosis"].values 

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

num_neighbors = [4,5,7,10,12,15,18,20, 25] 

for K in num_neighbors:

  print("K=", str(K))

  kn_classifier = KNeighborsClassifier(n_neighbors=K, algorithm='auto', metric='minkowski')  
  kn_classifier.fit(X_train,Y_train)

  Y_train_predicted = kn_classifier.predict(X_train)
  Y_train_predicted_proba = kn_classifier.predict_proba(X_train)  

  #Model overfitting evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
  print("\nModel overfitting evaluation")
  print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted)) #Best possible score is 1.0
  print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) #Best possible score is 0
  
  Y_test_predicted = kn_classifier.predict(X_test)
  Y_test_predicted_proba = kn_classifier.predict_proba(X_test)
  
  #Model evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
  print("\nModel evaluation")
  print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) #Best possible score is 1.0
  print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) #Best possible score is 0

'''
The model would appear to be appropriate for this problem.
'''

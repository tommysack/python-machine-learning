import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
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
digits_df.isnull().sum() 
sns.countplot(data=digits_df, x='target') #Ok, the classes are quite distributed
np.isnan(digits_df).any() #Many algorithms do work only with numerical data

'''
The data are points in an hyperspace H of 65 dimensions.
The goal is to assign a class label Y (classification with values 0..9) to input X.
In this case we use K-NN classification that assign a class to a point based on the classes of k neighboring points
(very low k => overfitting). In training phase K-NN does not learn any model ("lazy"), 
and the predictions are made just-in-time by calculating the similarity. 
It make a non-probabilistic multiclass non-linear classifier. 
'''

#Separates data in Dataframe/Series columns data/target 
X = digits.data 
Y = digits.target 

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

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

num_neighbors = [4,5,7,10,12,15,18,20,25] 

for K in num_neighbors:

  print("\nK=", str(K))

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


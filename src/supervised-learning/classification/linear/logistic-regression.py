import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

#Load data
breast_cancer_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
  names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
)

#General info
breast_cancer_df.head()
breast_cancer_df.describe() 
breast_cancer_df.shape #32 columns, 569 rows
breast_cancer_df['diagnosis'].unique() #array(['M', 'B'], dtype=object) => binary classification
sns.countplot(data=breast_cancer_df, x='diagnosis') #Ok, the classes are quite distributed
breast_cancer_df.isnull().sum() #Check no null 

#Correlation between data/target (corr function works only for numbers)
breast_cancer_df.isnull().sum() 
np.isnan(breast_cancer_df.drop('diagnosis',axis=1)).any()
diagnosis_mapping = {'B': 0, 'M': 1}
breast_cancer_df['diagnosis_numbers'] = breast_cancer_df['diagnosis'].map(diagnosis_mapping)
breast_cancer_df.drop('diagnosis', axis=1).corr()['diagnosis_numbers'].sort_values() #Only concave points_worst and perimeter_worst
#Correlation between "concave points_worst" and "perimeter_worst" to exclude duplicate feature 
breast_cancer_df.drop('diagnosis', axis=1).corr()['concave points_worst'].sort_values() 
breast_cancer_df.drop('diagnosis', axis=1).corr()['perimeter_worst'].sort_values() 

plt.figure(figsize=(6, 6))
sns.regplot(data=breast_cancer_df, x='concave points_worst', y='diagnosis_numbers', logistic=True)
plt.title('Correlation between concave points_worst and diagnosis numbers')
plt.xlabel('concave points_worst')
plt.ylabel('diagnosis numbers')
plt.show()

plt.figure(figsize=(6, 6))
sns.regplot(data=breast_cancer_df, x='perimeter_worst', y='diagnosis_numbers', logistic=True)
plt.title('Correlation between perimeter_worst and diagnosis numbers')
plt.xlabel('perimeter_worst')
plt.ylabel('diagnosis numbers')
plt.show()

'''
The data are points in an hyperspace H of 32 dimensions.
The goal is to assign a class label Y (binary classification with values "M" or "B") to input X.
Technically you need to find the "best" hyperplane of 31 dimensions which best separates the points classified in H.
In this case we use LogisticRegression that use LBFGS method (Gradient Ascent to maximize Likelihood) and returns
the probability between 0 and 1 that a point belongs to a class (using sigmoid function).
'''

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer_df[["concave points_worst","perimeter_worst"]].values 
Y = breast_cancer_df['diagnosis_numbers'].values 

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

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
logistic_regression.fit(X_train, Y_train) #Building the model

Y_train_predicted = logistic_regression.predict(X_train) #To calculate model's overfitting
Y_train_predicted_proba = logistic_regression.predict_proba(X_train) #To calculate model's overfitting

#Model overfitting evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
print("\nModel overfitting evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted)) #Best possible score is 1.0
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) #Best possible score is 0

Y_test_predicted = logistic_regression.predict(X_test) 
Y_test_predicted_proba = logistic_regression.predict_proba(X_test) #To calculate LOG LOSS

#Model evaluation (the percentage of samples that were correctly classified, and the negative likelihood)
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) #Best possible score is 1.0
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) #Best possible score is 0

'''
Both metrics suggest that the Logistic Regression model is correct.
Could be improved the moderate overfitting.
'''

#Try to predict a new case
x=[[-0.85095647, -0.48784158]]
y_predicted = logistic_regression.predict(x)
y_predicted_proba = logistic_regression.predict_proba(x)
print("\nDiagnosis: ", y_predicted[0])
print("Probability class 0: ", y_predicted_proba[0][0])
print("Probability class 1: ", y_predicted_proba[0][1])



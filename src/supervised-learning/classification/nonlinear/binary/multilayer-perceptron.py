import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

#Load data
passengers_df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")

#General info
passengers_df.head()
passengers_df.describe() 
passengers_df.shape #8 columns, 887 rows
passengers_df['Survived'].unique() #array([0, 1]) => binary classification (Not Survived, Survived)
sns.countplot(data=passengers_df, x='Survived') #The 1 class are many more than 0 class
passengers_df.isnull().sum() 

#Correlation between features and target (we assume moderate correlation from 0.5)
passengers_df= passengers_df.drop("Name",axis=1) #Drop column Name that is clearly irrelevant
mapping_sex = {'male': 0, 'female': 1}
passengers_df['Sex'] = passengers_df['Sex'].map(mapping_sex)
passengers_df.corr()['Survived'].sort_values() #There may be a correlation with Age, Pclass and Sex
sns.barplot(data=passengers_df, x='Sex', y='Survived') #The female(1) have survival average more than male(0)
sns.barplot(data=passengers_df, x='Pclass', y='Survived') #The Pclass(3) flu the survival average 
def map_age(age): 
  return round(age / 10)
passengers_df['Age_block'] = passengers_df['Age'].apply(lambda age: map_age(age))
sns.barplot(data=passengers_df, x='Age_block', y='Survived')
sns.barplot(data=passengers_df, x='Age_block', y='Survived', hue='Sex') #The Age flu the survival average

'''
I would try with Multi-layer perceptron classifier and the features Age, Pclass and Sex.
'''

'''
The data are points in an hyperspace H of 8 dimensions.
The goal is to assign a class label Y (binary classification with values 0 or 1) to input X.
Technically you need to find the "best" hypercurve of 7 dimensions which best separates the points classified in H.
The "best": in this case we use  MLPClassifier that uses "solver" to to minimize Log Loss.
It make a non-linear binary (but is possible multi-class) probabilistic classifier. 
'''

#Separates data in numpy.ndarray columns data/target 
#X = passengers_df[["Pclass", "Age", "Sex"]].values
X = passengers_df.drop('Survived', axis=1).values #
Y = passengers_df["Survived"].values

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

mlp_classifier = MLPClassifier(
  hidden_layer_sizes=(200, 200, 200), #each tupla's element represents the numbers of neurons in its specific layer
  activation='relu', #activation function 
  solver='adam', #to use for weight optimization
  alpha=0.01, #constant that multiplies the L2 regularization
  batch_size='auto', #size of minibatches 
  learning_rate='constant', #the rate to update the weights
  #max_iter=200, #max number of iterations 
  early_stopping=True, #to stop training iterations when the score is not improving
  verbose=True
)
mlp_classifier.fit(X_train, Y_train)

Y_train_predicted = mlp_classifier.predict(X_train) 
Y_train_predicted_proba = mlp_classifier.predict_proba(X_train) 

#Model overfitting evaluation 
print("\nModel overfitting evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted)) 
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) 

Y_test_predicted = mlp_classifier.predict(X_test) 
Y_test_predicted_proba = mlp_classifier.predict_proba(X_test)

#Model evaluation 
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted)) 
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) 

'''
The model would appear to be appropriate for this problem.
'''






import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer

#Load data
iris = load_iris()

#General info
print(iris.DESCR) #Flowers 4 columns of values, and 1 column with the class
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target'].unique() #array([0, 1, 2]) => multi-class classification
iris_df.head()
iris_df.describe() 
iris_df.shape 
sns.countplot(data=iris_df, x='target') #Ok, the classes are quite distributed
iris_df.isnull().sum() 
np.isnan(iris_df).any() 

#Correlation between features (we assume moderate correlation from 0.5)
iris_df_corr = iris_df.corr()
plt.figure(figsize=(5, 5))
sns.heatmap(iris_df_corr, cmap="viridis", annot=True, linewidths=0.5) 

'''
All the columns are moderatly correlated between them, however there are only 5 columns so I choose to keep them all
'''

#Separates data in numpy.ndarray columns data/target 
X = iris.data 
Y = iris.target 

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

'''
We want to adopt:
- Logistic Regression classifier LogisticRegression
- SGDClassifier to perform SGD in Logistic Regression classification
- KFold to training with cross validation
- GASearchCV to tuning hyperparameters of Logistic Regression classifier
'''

sgd_classifier = SGDClassifier(
  loss="log_loss", #log gives a Logistic Regression
  eta0=1,
  early_stopping=True   
) 
param_grid = { 
  "penalty": Categorical(['l2', 'l1', 'elasticnet']),
  "alpha": Categorical([0.0001, 0.001, 0.01, 0.1, 1, 10]), 
  "fit_intercept": Categorical([True, False]), 
  "learning_rate": Categorical(["constant", "optimal", "invscaling", "adaptive"])
}

kfold = KFold(n_splits=10)

ga_search_cv = GASearchCV(
  estimator=sgd_classifier, 
  cv=kfold, 
  param_grid=param_grid, 
  population_size=10,
  generations=5, 
  verbose=True, 
  n_jobs=-1
)

ga_search_cv.fit(X_train, Y_train)

#print(randomized_search_cv.best_params_)
#OUTPUT:
#{'penalty': 'l2', 'learning_rate': 'optimal', 'fit_intercept': False, 'alpha': 0.1}
#print(randomized_search_cv.best_score_)

Y_train_predicted = ga_search_cv.predict(X_train)

#Model overfitting evaluation
print("\nModel overfitting evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_train, Y_train_predicted))

Y_test_predicted = ga_search_cv.predict(X_test)

#Model evaluation
print("\nModel evaluation")
print("ACCURACY SCORE: ", accuracy_score(Y_test, Y_test_predicted))






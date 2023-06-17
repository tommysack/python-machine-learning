import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'''
It is an embedded feature selection because the selection occurs during model fitting.
Lasso that can shrink some of the coefficients to zero, than that feature can be dropped.
'''

#Load data
breast_cancer = load_breast_cancer()

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer.data
Y = breast_cancer.target

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

logistic_regression = LogisticRegression(penalty='l2', C=10.0, solver='lbfgs')
logistic_regression.fit(X_train, Y_train)
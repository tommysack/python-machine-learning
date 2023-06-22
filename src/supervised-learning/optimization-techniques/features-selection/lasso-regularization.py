import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

'''
It is an embedded feature selection because the selection occurs during model fitting.
Lasso that can shrink some of the weights to zero, then some feature can be dropped.
'''

#Load data
breast_cancer = load_breast_cancer()

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer.data
Y = breast_cancer.target

logistic_regression = LogisticRegression(
  penalty='l2', #L2 regularization to avoid overfitting 
  C=10.0, #inverse of regularization strength (C lower => Higher regularization)
  solver='lbfgs', #algorithm to use
  verbose=True
) 
logistic_regression.fit(X, Y)

#SelectFromModel dropes the features relying to importance weights (default=median)
select_from_model = SelectFromModel(logistic_regression, prefit=True)
X_selected = select_from_model.transform(X)


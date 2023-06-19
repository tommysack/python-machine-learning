import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

'''
This approach is based on decision trees which causes that the nodes with higher "impurity-decrese"
are at the top of the tree, and the node with lower "impurity-decrese" at the bottom.
Than we can find the most important features in the first part of the tree.
'''

#Load data
digits = load_digits()

#Separates data in Dataframe/Series columns data/target 
X = digits.data 
Y = digits.target 

random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=10, random_state=False)
random_forest_classifier.fit(X, Y)

importances_df = pd.DataFrame({"features": digits.feature_names, "importances": random_forest_classifier.feature_importances_}).sort_values('importances')
importances_df.plot.bar()




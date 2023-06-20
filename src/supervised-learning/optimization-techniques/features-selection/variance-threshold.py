import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import VarianceThreshold

'''
It returns the features whose variance does not exceed a certain threshold (default threshold=0). 
It supposes that a feature with similar values in all rows is of little influence.
'''

#Load data
breast_cancer = load_breast_cancer()

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer.data
Y = breast_cancer.target

variance_threshold = VarianceThreshold(
  threshold=0.001 #features with variance lower than this will be removed
) 
variance_threshold.fit(X)

variance_threshold.get_support()


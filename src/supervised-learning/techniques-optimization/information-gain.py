import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif

'''
It is commonly used to select the variable that maximizes the information gain, 
that therefore minimizes the entropy (or "surprise").
'''

#Load data
breast_cancer = load_breast_cancer()

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

#Separates data in numpy.ndarray columns data/target 
X = breast_cancer.data
Y = breast_cancer.target

info_classif = mutual_info_classif(X, Y)

breast_cancer_df_len = len(breast_cancer_df.columns)
breast_cancer_df_importances = pd.Series(info_classif, breast_cancer_df.columns[0:breast_cancer_df_len])
breast_cancer_df_importances.plot(kind='barh', color='#BB0000', figsize=(5, 5))
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

'''
It misures the linear relationship between the features.
If two features are correlated, than we can predict one from the other.
Generally we drop the feature which has less correlation with the target.
'''

#Load data
breast_cancer = load_breast_cancer()

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

breast_cancer_df_corr = breast_cancer_df.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(breast_cancer_df_corr, cmap="viridis", annot=True, linewidths=0.5)


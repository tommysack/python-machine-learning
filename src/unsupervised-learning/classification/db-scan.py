import numpy as np
from sklearn.datasets._samples_generator import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

#It creates 200 points to distribute and nois 0.05 (two interleaving half circles)
X, Y = make_moons(n_samples=200, noise=0.1) 

#General info
np.unique(np.array(Y.tolist())) #array([0, 1]) => binary classification
X.shape #200 rows, 2 columns
Y.shape #200 row, 1 column

'''
The data are points in an hyperspace H of 3 dimensions.
The goal is to assign a class label Y to input X.
In this case we use the DB Scan.
It make an unsupervised non-linear binary classifier.  
'''

'''
DB scan steps:

1 - Select the parameters:
  ε maximum distance between 2 observations of the same cluster 
  MinPts minimum number of observations required to form a cluster
2 - Select a point p that has not been visited
3 - Draw a "circle" with center p and radius ε, and count how many points n are inside the circle
4 - If n >= MinPts 
    then 
      p is a core_point of a cluster c and go to the next point (step 2)
    else 
      if there is a core_point in the circle 
        then 
          p is a border_point and is assigned to cluster c and go to the next point (step 2) 
        else 
          p is a noise_point and goes to the next point (step 2) 

PROS: it doesn't require to predefine K parameter, and it can handle clusters that are not necessarily spherical or of uniform size
CONS: it requires to predefine eps and MinPts parameters
'''

dbscan = DBSCAN(
  eps=0.25,  #maximum distance between two observations of the same cluster
  min_samples=5 #min_samples minimum number of observations required to form a cluster
)

Y_predicted = dbscan.fit_predict(X)

#Draw correlation between the first two numerical features and class Y_predict
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y_predicted, palette='viridis')
plt.title("DB Scan result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

ch_score = calinski_harabasz_score(X, Y_predicted)
print("Calinski/Harabasz score:", ch_score)

db_score = davies_bouldin_score(X, Y_predicted)
print("Davies Bouldin score:", db_score)




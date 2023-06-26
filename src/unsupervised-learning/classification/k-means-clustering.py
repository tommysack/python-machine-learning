import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

#It creates 250 points to distribute in clusters, with 3 features, 5 centroids and 0.8 cluster std deviation
X, Y = make_blobs(n_samples=250, n_features=3, centers=5, cluster_std=0.8)

#General info
np.unique(np.array(Y.tolist())) #array([0, 1, 2, 3, 4]) => multi-class classification
X.shape #250 rows, 3 columns
Y.shape #250 row, 1 column

'''
The data are points in an hyperspace H of 4 dimensions.
The goal is to assign a class label Y to input X.
In this case we use K-means clustering.
It make an unsupervised non-linear multi-class classifier.  
'''
'''
K-mean steps:
1 - Choose the K value with Elbow method (when the SSE value doesn't go down any faster)
2 - Randomly select K centroids
3 - Calculate the distance between each of the observations and the K centroids
4 - Assign each observation to the cluster represented by the centroid closest to that observation
5 - Recalculation of the centroid as the midpoint of the observation of each cluster
6 - We repeat the process until no more observations change clusters (at each iteration there will be fewer and fewer observations that change clusters)

PROS: it works very well with big data 
CONS: it requires to define K (however prerequisite solvable with the Elbow method) and it does not allow for noisy data
'''

for k in range(2, 11):

  kmeans = KMeans(
    n_clusters=k, #num of clusters to form (and centroids to generate)
    init='k-means++', 
    n_init= 10, #the algorithm will run n_init times with different centroid (because it is sensitive to the initial placement of centroids)
    max_iter=2000 #every time of n_init it iterates a maximum of 2000 times to converge
  )    
  kmeans.fit(X)

  Y_predicted = kmeans.predict(X)

  #Draw correlation between the first two numerical features and class Y_predict
  sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y_predicted, palette='viridis')
  plt.title("K-means Clustering result")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.show()  

  print("K=", k)
  print("SSE: ", kmeans.inertia_) 
  ch_score = calinski_harabasz_score(X, Y_predicted)
  print("Calinski/Harabasz score:", ch_score)
  db_score = davies_bouldin_score(X, Y_predicted)
  print("Davies Bouldin score:", db_score)
  print("Final position of centers:  ", kmeans.cluster_centers_) 
  print("Number of iterations to converge: ", kmeans.n_iter_) 
  print("Label classification of points: ", Y_predicted)
  print("----------------------------")

'''
From k>5 SSE it doesn't drop as fast as before, then we can use 5 clusters.
'''



import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs

#It creates 250 points to distribute in clusters, with 3 features, 5 centroids and 0.8 cluster std deviation
X, Y = make_blobs(n_samples=250, n_features=3, centers=5, cluster_std=0.8)

#General info
np.unique(np.array(Y.tolist())) #array([0, 1, 2, 3, 4]) => multi-class classification
X.shape #250 rows, 3 columns
Y.shape #250 row, 1 column

'''
The data are points in an hyperspace H of 4 dimensions.
The goal is to assign a class label Y to input X.
In this case we use the Agglomerative Hierarchical Clustering.
It make an unsupervised non-linear multi-class classifier.  
'''
'''
Agglomerative Hierarchical Clustering steps:

1 - Start with each point as a separate cluster.
2 - Calculate the "distance" (or similarity) between each pair of clusters. 
3 - Merge the two closest clusters into a single cluster.
4 - Recalculate the distance between the new cluster and the remaining clusters.
5 - Repeat steps 3 and 4 until all the data points are part of a single cluster.
The clusters are represented by a dendrogram, that visualize the merging process as different levels.
Than you should cut the dendrogram at a given height (which identifies the maximum distance between the clusters) to partition your data into clusters.

PROS: it doesn't require to predefine K parameter
CONS: it's not better solution for large dataset
'''

agglomerative_clustering = AgglomerativeClustering() 

Y_predicted = agglomerative_clustering.fit_predict(X)


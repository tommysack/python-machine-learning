# Documentation

## How to start project

If you use VS Code, than you don't need start container. VS Code open folder in container for you.
Otherwise to start "ml-py" container and exec bash into container:

```
docker run --name=ml-py -it -d --restart always -v /path/to/ML-py:/home/ML-py tommasosacramone/ml-py
docker exec -w /home/ML-py -it ml-py bash 
```

To restart "ml-py" container:

```
docker start ml-py
docker exec -w /home/ML-py -it ml-py bash 
```

## Performance metrics 

**Regression**

- MAE (Mean Absolute Error)
  - MAE = (1 / n) * Σ (Y_predicted - Y)

- MSE (Mean Squared Error)
  - MSE = (1 / n) * Σ (Y_predicted - Y)^2

- R2 score (Coefficient of determination)
  - SSR = Σ (Y - Y_predicted)^2
  - SST = Σ (Y - mean(Y))^2
  - R2 = 1 - (SSR / SST) 
  - Best score is 1.0

**Supervised classification**

- Accuracy score (The percentage of samples that were correctly classified)
  - Accuracy = Num of correct predictions / Total num of samples 
  - Best score is 1.0

- F1 score (Harmonic Precision-Recall Mean)
  - Precision = True Positives / (True Positives + False Positives)
  - Recall = True Positives / (True Positives + False Negatives)
  - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
  - Best score is 1.0

- Log loss (Negative Likelihood)
  - Log Loss = - (1 / num of samples) * Σ [Y * log(Y_predicted) + (1 - Y) * log(1 - Y_predicted)]
  - Best score is 0

**Unsupervised classification**

- SSE (Inertia or Distorsion)
  - Best score is 0

- Calinski Harabasz score (Variance Ratio Criterion)
  - BSS is Between-Cluster Sum of Squares
  - WSS is Within-Cluster Sum of Squares
  - Cal-inski-Harabasz = (BSS / WSS) * ((n_samples - k) / (k - 1))
  - Higher score indicates better clustering 

- Davies Bouldin score (Average similarity measure between clusters)
  - Best score is 0

## Data visualization

- seaborn.regplot
  - to draw two continue features and the relative linear regression model on two-dimension plane
- seaborn.heatmap
  - to display relationships between all features in a matrix, the values of matrix are represented as colors
- seaborn.countplot
  - to draw bars representing counts of a discrete feature
- seaborn.barplot
  - to draw bars representing values of continue feature y for a discrete feature x (but if y is discrete/categorical feature then representing a mean, or an other estimator, of central tendency for y)
- seaborn.scatterplot
  - to draw two continue features on two-dimension plane, with hue to draw one more discrete/categorical feature with the colors

## Source folder

**Supervised Learning**

1. Regression

  - Linear
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Gradient Descent Stochastic Linear Regression
    
  - Non linear
    - Polynomial Regression

2. Classification

  - Linear

    - Binary      
      - Logistic Regression
      - Support Vector Machines

    - Multiclass      
      - Logistic Regression (one vs all)
      - Support Vector Machines (one vs all)      
      - Linear Discriminant Analysis

  - Non linear

    - Binary
      - K-nearest Neighbors
      - Decision Tree
      - Random Decision Forest
      - Multi-layer Perceptron 
      - Bernoulli Naive Bayes      

    - Multiclass
      - K-nearest Neighbors
      - Decision Tree
      - Random Decision Forest
      - Multi-layer Perceptron 
      - Multinomial Naive Bayes       

3. Optimization techniques

  - Algorithm optimization
    - Gradient Descent Stochastic

  - Data training
    - K-fold Cross-Validation  

  - Hyperparameter model optimization
    - Randomize Search
    - Genetic Algorithm Search 
  
  - Features extraction techniques
    - Principal Component Analysis
    - Kernel Principal Component Analysis
    - Linear Discriminant Analysis

  - Features filter methods
    - Correlation Cofficient
    - Variance Threshold
    - Information Gain    
    - Fisher's Score
  
  - Features selection with embedded methods
    - Random Forest Importance
    - Lasso Regularization and Select From Model  
  
  - Samples of optimization techniques integrations
  
**Unsupervised Learning**    

1. Classification

  - K-means Clustering
  - Agglomerative Hierarchical Clustering
  - DB Scan
  





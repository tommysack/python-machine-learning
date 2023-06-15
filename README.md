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

## Performance metrics used

**Regression**

- MAE (Mean Absolute Error)

- MSE (Mean Squared Error)

- R2 score (Coefficient of determination)
R2 = ESS / TSS (best score is 1.0)

**Classification**

- Accuracy score (the percentage of samples that were correctly classified)
Num of correct predictions / Total num of predictions (best score is 1.0)

- F1 score (Harmonic Precision-Recall Mean)
Best score is 1.0

- Log loss (Negative Likelihood)
-1 * log-likelihood (best score is 0)

## The source folder

**Supervised Learning**

1. Regression

  - Linear
    - Linear regression
  - Non linear
    - Polynomial regression

2. Classification

  - Linear

    - Binary      
      - Logistic regression
      - Support Vector Machines
      - Bernoulli Naive Bayes

    - Multiclass      
      - Logistic regression (one vs all)
      - Support Vector Machines (one vs all)
      - Linear Discriminant Analysis
      - Multinomial Naive Bayes      

  - Non linear

    - Binary
      - K-nearest Neighbors
      - Decision Tree
      - Random Decision Forest

    - Multiclass
      - K-nearest Neighbors
      - Decision Tree
      - Random Decision Forest
      - Multi-layer Perceptron 

3. Optimization techniques

  - K-fold Cross-Validation
  - Gradient Descent Stochastic
  - Randomize Search
  - Principal Component Analysis
  - Kernel Principal Component Analysis
  - Linear Discriminant Analysis





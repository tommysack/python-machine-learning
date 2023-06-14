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
      - One vs all
        - Logistic regression 
        - Support Vector Machines
      - Linear Discriminant Analysis
      - Multinomial Naive Bayes      

  - Non linear

    - Binary
      - K Nearest Neighbors
      - Decision Tree
      - Random Decision Forest

    - Multiclass
      - K Nearest Neighbors

3. Techniques

  - k-fold Cross-Validation
  - Gradient Descent Stochastic
  - Randomize Search
  - Principal Component Analysis
  - Kernel Principal Component Analysis
  - Linear Discriminant Analysis





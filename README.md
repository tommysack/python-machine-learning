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

## The contained use cases

**Supervised Learning**

1. Regression

  - Linear
    - linear-regression.py
  - Non linear
    - polynomial-regression.py 

2. Classification

  - Linear

    - Binary      
      - logistic-regression.py
      - support-vector-machines.py
      - bernoulli-nb.py

    - Multiclass
      - One vs all
        - logistic-regression.py
        - support-vector-machines.py
      - multinomial-nb.py

  - Non linear

    - Binary
      - k-nearest-neighbors.py
      - decision-tree.py
      - random-decision-forest.py
      
    - Multiclass
      - k-nearest-neighbors.py




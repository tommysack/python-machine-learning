import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#Load data
reviews_df = pd.read_csv("../../../../../data/movie_review.csv")

#General info
reviews_df.head()
reviews_df.describe() 
reviews_df.shape #6 columns, 64720 rows
reviews_df['tag'].unique() #array(['pos', 'neg'], dtype=object) => binary classification
sns.countplot(data=reviews_df, x='tag') #Ok, the classes are quite distributed
reviews_df.isnull().sum() 

#Correlation between data/target
#The column "text" contains phrases in natural language, it's not possible check correlation with corr() that  works only for numbers

'''
The data are phrases in natural language.
The goal is to assign a class label Y (binary classification with values "pos" or "neg") to input X.
In this case we use BernoulliNB that use  Naive Bayes algorithm.
It makes a binary classifier. 
'''

#Separates data in numpy.ndarray columns data/target 
X = reviews_df["text"].values 
Y = reviews_df["tag"].values

#BernoulliNB is designed for binary/boolean features, than encoding of X with CountVectorizer
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(X)
#print("Features names after vectorizer: ", vectorizer.get_feature_names_out())

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

bernoulli = BernoulliNB()
bernoulli.fit(X_train, Y_train) #Building model

#Model overfitting evaluation (the percentage of samples that were correctly classified)
print("\nModel overfitting evaluation")
print("SCORE: ", bernoulli.score(X_train, Y_train)) #Best possible score is 1.0

#Model evaluation (the percentage of samples that were correctly classified)
print("\nModel evaluation")
print("SCORE: ", bernoulli.score(X_test, Y_test)) #Best possible score is 1.0

'''
The metric makes me think that there is moderate overfitting.
'''

#Try to predict a new case

x = ["I liked soundtrack, photography and the cast but the film was really long and it lacked of a good plot."]
x = vectorizer.transform(x)

y = bernoulli.predict(x)
print("\nSentiment analysis of review: ", y[0])




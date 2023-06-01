import pandas as pd
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#Load data
news = fetch_20newsgroups(random_state=42)

#General info
print(news.DESCR) #20 newsgroups dataset (18000 newsgroups)
news.target_names #All categories labels
news.data[10] #Content of article with index 10
news.target[10] #Category of article with index 10

'''
The data are articles in natural language.
The goal is to assign a class label Y (multi-class classification with values news_train.target_names) to input X.
In this case we use MultinomialNB that use Naive Bayes algorithm.
It makes a multi-class classifier. 
'''

#Separates data in rows train/test
news_train = fetch_20newsgroups(subset="train")
X_train = news_train.data 
Y_train = news_train.target
news_test = fetch_20newsgroups(subset="test")
X_test = news_test.data 
Y_test = news_test.target

multinomial = MultinomialNB(alpha=1)

#MultinomialNB is designed for integer feature counts, than we need encoding. 
#We can encode X with "bag of words" CountVectorizer.

#To transforms the matrix to tf-idf representation we can encode with TfidfTransformer. 
#It rankes a word in a doc, based on the frequency that the word appears in all docs.
#The words that appears once have higher score than those that appears most frequently.

#TfidfTransformer = CountVectorizer + TfidfTransformer
tfidf_vectorizer = TfidfVectorizer()

X_train_vector = tfidf_vectorizer.fit_transform(X_train) 

multinomial.fit(X_train_vector, Y_train)

Y_train_predicted = multinomial.predict(X_train_vector)

#Model overfitting evaluation (the Harmonic Precision-Recall Mean)
print("\nModel overfitting evaluation")
print("F1 SCORE: ", metrics.f1_score(Y_train, Y_train_predicted, average='macro')) #Best possible score is 1.0

X_test_vector = tfidf_vectorizer.transform(X_test)

Y_test_predicted = multinomial.predict(X_test_vector)

#Model evaluation (the Harmonic Precision-Recall Mean)
print("\nModel evaluation")
print("F1 SCORE: ", metrics.f1_score(Y_test, Y_test_predicted, average='macro')) #Best possible score is 1.0

'''
The metric makes me think that there is moderate overfitting.
'''

#Try to predict a new case
x = ["The Italy will have to vote for the political elections"]
x_vector = tfidf_vectorizer.transform(x)
y = multinomial.predict(x_vector)
print("\nCategory of doc: ", news.target_names[y[0]])
    


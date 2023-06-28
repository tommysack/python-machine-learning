import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score,KFold

#Load data
reviews_df = pd.read_csv("data/movie_review_imdb.csv")

#General info
reviews_df.head()
reviews_df.describe() 
reviews_df.shape #2 columns, 50000 rows
reviews_df['sentiment'].unique() #array(['positive', 'negative'], dtype=object) => binary classification
sns.countplot(data=reviews_df, x='sentiment') #Ok, the classes are quite distributed
reviews_df.isnull().sum() 

#Correlation between data/target (we assume moderate correlation from 0.5)
#The column "text" contains phrases in natural language, it's not possible check correlation with corr() that  works only for numbers

'''
The data are phrases in natural language.
The goal is to assign a class label Y (binary classification with values "pos" or "neg") to input X.
In this case we use BernoulliNB that uses Naive Bayes algorithm and it works on Bernoulli distribution.
It makes a non-linear binary probabilistic classifier. 
'''

#Separates data in numpy.ndarray columns data/target 
X = reviews_df["review"].values 
Y = reviews_df["sentiment"].values

#Separates data in rows train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

bernoulli = BernoulliNB(
  #alpha=2
)

#BernoulliNB is designed for binary/boolean features, then we need one-hot encoding. 
#We can encode X with "bag of words" CountVectorizer using binary=True parameter (One-hot).
#It returns scipy.sparse._csr.csr_matrix that is allowed for train_test_split 
count_vectorizer = CountVectorizer(
  lowercase=True, 
  stop_words='english', #terms to be ignored  
  binary=True #all non zero values are set to 1 (for binary classification)  
)

X_train_vector = count_vectorizer.fit_transform(X_train)
X_test_vector = count_vectorizer.transform(X_test)

bernoulli.fit(X_train_vector, Y_train) #Building model

Y_train_predicted = bernoulli.predict(X_train_vector)
Y_train_predicted_proba = bernoulli.predict_proba(X_train_vector)

#Model overfitting evaluation 
print("\nModel overfitting evaluation")
print("ACCURACY: ", accuracy_score(Y_train, Y_train_predicted)) 
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) 

Y_test_predicted = bernoulli.predict(X_test_vector)
Y_test_predicted_proba = bernoulli.predict_proba(X_test_vector)

#Model evaluation 
print("\nModel evaluation")
print("ACCURACY: ", accuracy_score(Y_test, Y_test_predicted)) 
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) 

'''
The model would appear moderately overfitted for this problem.
'''

#Try to predict a new case

x1 = ["I liked soundtrack, photography and the cast but the film was really long and it lacked of a good plot."]
x1 = count_vectorizer.transform(x1)

y1 = bernoulli.predict(x1)
print("\nSentiment analysis of review: ", y1[0])

x2 = ["I liked the actors and direction, but the soundtrack wasn't the best."]
x2 = count_vectorizer.transform(x2)

y2 = bernoulli.predict(x2)
print("\nSentiment analysis of review: ", y2[0])

x3 = ["I loved this movie, beautiful cast and soundtrack."]
x3 = count_vectorizer.transform(x3)

y3 = bernoulli.predict(x3)
print("\nSentiment analysis of review: ", y3[0])



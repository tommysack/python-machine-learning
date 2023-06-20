import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, log_loss
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#import nltk
#import gensim

#Load data
news = fetch_20newsgroups(random_state=42)

#General info
print(news.DESCR) #News in 2 columns: 1 with text, and 1 with category
news_df = pd.DataFrame(news.data)
news_df['category'] = news.target
news_df['category'].unique() #array([ 7,  4,  1, 14, 16, 13,  3,  2,  8, 19,  6,  0, 12,  5, 10,  9, 15, 17, 18, 11]) => multi-class classification
news_df.head()
news_df.describe() 
news_df.shape #2 columns, 11314 rows
sns.countplot(data=news_df, x='category') #Ok, the classes are quite distributed
news_df.isnull().sum()

'''
The data are articles in natural language.
The goal is to assign a class label Y (multi-class classification with values news_train.target_names) to input X.
In this case we use MultinomialNB that uses Naive Bayes algorithm and it works on Multinomial distribution.
It makes a linear multi-class probabilistic classifier. 
'''

#Separates data in rows train/test
news_train = fetch_20newsgroups(subset="train")
X_train = news_train.data 
Y_train = news_train.target
news_test = fetch_20newsgroups(subset="test")
X_test = news_test.data 
Y_test = news_test.target

multinomial = MultinomialNB()

#MultinomialNB is designed for integer feature counts, than we need encoding. 
#We can encode X with "bag of words" CountVectorizer.

#To transforms the matrix to tf-idf representation we can encode with TfidfTransformer. 
#It rankes a word in a doc, based on the frequency that the word appears in all docs.
#The words that appears once have higher score than those that appears most frequently.

#I decide to use TfidfVectorizer = CountVectorizer + TfidfTransformer

#I also decided to process and tokenized the news with Stemmer and Lemmatizer

# nltk.download('wordnet') #Import dictionary
# lemmatizer = nltk.stem.WordNetLemmatizer() #words in third person, verbs in past/future, ...
# stemmer = nltk.stem.SnowballStemmer("english") #to  map different forms of the same word to a stem

# def tokenizer(text):
#   tokens = gensim.utils.simple_preprocess(text)
#   tokens_processed = []
#   for token in tokens :
#     token_lemma = lemmatizer.lemmatize(token, pos='v')
#     token_lemma_stemma = stemmer.stem(token_lemma)
#     tokens.append(token_lemma_stemma)            
#   return tokens_processed

#tfidf_vectorizer = TfidfVectorizer(lowercase=True, tokenizer=tokenizer, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(
  lowercase=True, 
  stop_words='english' #terms to be ignored 
)

X_train_vector = tfidf_vectorizer.fit_transform(X_train) 

multinomial.fit(X_train_vector, Y_train)

Y_train_predicted = multinomial.predict(X_train_vector)
Y_train_predicted_proba = multinomial.predict_proba(X_train_vector)

#Model overfitting evaluation 
print("\nModel overfitting evaluation")
print("F1 SCORE: ", f1_score(Y_train, Y_train_predicted, average='macro')) 
print("LOG LOSS: ", log_loss(Y_train, Y_train_predicted_proba)) 

X_test_vector = tfidf_vectorizer.transform(X_test)

Y_test_predicted = multinomial.predict(X_test_vector)
Y_test_predicted_proba = multinomial.predict_proba(X_test_vector)

#Model evaluation 
print("\nModel evaluation")
print("F1 SCORE: ", f1_score(Y_test, Y_test_predicted, average='macro')) 
print("LOG LOSS: ", log_loss(Y_test, Y_test_predicted_proba)) 

'''
The model would appear moderately overfitted for this problem.
'''

#Try to predict a new case
x = ["The Italy will have to vote for the political elections"]
x_vector = tfidf_vectorizer.transform(x)
y = multinomial.predict(x_vector)
print("\nCategory of doc: ", news.target_names[y[0]])
    


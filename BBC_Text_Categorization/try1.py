import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix,cohen_kappa_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def string_form(value):
	return str(value)

def clean_text(text):
   
	text = BeautifulSoup(text, "lxml").text
	text = text.lower()
	text = REPLACE_BY_SPACE_RE.sub(' ', text)
	text = BAD_SYMBOLS_RE.sub('', text)
	text = ' '.join(word for word in text.split() if word not in STOPWORDS)
	return text

df=pd.read_csv('bbc-text.csv')
df['text'] = df['text'].apply(string_form)
df['text'] = df['text'].apply(clean_text)
category=['tech' ,'business' ,'sport' ,'entertainment' ,'politics']
# print(df['category'].unique())
# print(df.head(5))
def nb_classifier(X_train, X_test, y_train, y_test):

	nb = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
	nb.fit(X_train, y_train)
	y_pred = nb.predict(X_test)
	print('accuracy %s' % accuracy_score(y_pred, y_test))
	kappa = cohen_kappa_score(y_test,y_pred)
	print('Kappa= ',kappa)
  # print(classification_report(y_test, y_pred,target_names=category))

def linear_svm(X_train, X_test, y_train, y_test):
  
  
	sgd = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))])
	sgd.fit(X_train, y_train)
	y_pred = sgd.predict(X_test)
	print('accuracy %s' % accuracy_score(y_pred, y_test))
	kappa = cohen_kappa_score(y_test,y_pred)
	print('Kappa= ',kappa)
  # print(classification_report(y_test, y_pred,target_names=category))



def mlpclassifier(X_train, X_test, y_train, y_test):
    
	mlp = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MLPClassifier(hidden_layer_sizes=(128,64,32,16)))])
	mlp.fit(X_train, y_train)
	y_pred = mlp.predict(X_test)
	print('accuracy %s' % accuracy_score(y_pred, y_test))
	kappa = cohen_kappa_score(y_test,y_pred)
	print('Kappa= ',kappa)
	


def train_test(X,y):
 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
	print("Results of Naive Bayes Classifier")
	nb_classifier(X_train, X_test, y_train, y_test)
	print("Results of Linear Support Vector Machine")
	linear_svm(X_train, X_test, y_train, y_test)
	print("Results of MLP Classifier")
	mlpclassifier(X_train, X_test, y_train, y_test)




cat = df.category
V = df.text
train_test(V,cat)


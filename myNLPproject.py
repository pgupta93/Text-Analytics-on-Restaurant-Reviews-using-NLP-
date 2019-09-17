# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:38:39 2019

@author: prabh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#imporing dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t' , quoting= 3 )

#Cleaning first review and then will apply for loop for all the reviews
#has great tools to clean text
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  
#corpus is a collection of text of same time
corpus = []
for i in range(0,1000):
#don't want to remove a-z and A-Z for the first review and replace the removed numbers or punctuations by space
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
#removing all capital letters 
    review = review.lower()
#remove non-significant words like the, that, and,in etc. for this will import nltk library

#forloop for all the reviews but we have to split review to different words so review will be list of d/t words
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#Stemming (taking the root of the word like love from loved) import a class and use its object doing it in last step
#Now joining back the components of the review but separate by space
    review =' '.join(review)
#for loop for all the reviews will replace Review[0] to Review[i] for i in Review
    corpus.append(review)
    
    # TF-IDF




messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

#Creating Bag of Word Model to remove words that are not used a lot through tokenization(countvectorization class)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
#Train our ML model we need dependent variable that is the result taking all the rows of reviews and index 1
y = dataset.iloc[:,1].values

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(X)

#Applying Naive bayes

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.20, random_state = 0)

#Fitting Naive Bayes  to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)
 
#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:15:51 2020

@author: parth
"""


import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

#nltk.download('wordnet') # lemmatizer
#nltk.download('stopwords') # to remove stop words and the of an etc
data = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

# replacing everthing except english alphabets with a space
review = re.sub('[^a-zA-Z]',' ' ,data['text'][0]) 
review = review.lower()
review = review.split() # convert str to list of words

# removing stopwords
lem = WordNetLemmatizer()
#port = PorterStemmer()
review = [(lem.lemmatize(word)) for word in review if not word in set(stopwords.words('english'))]
#exrev = [snow.stem(w) for w in ['loves', 'hates' ,'liked']] love hate like

#list to string
review = ' '.join(review)
corpus = []

for i in range(len(data)):
    textrev = data['text'][i]
    review = re.sub('[^a-zA-Z]',' ' ,textrev) 
    review = review.lower()
    review = review.split() # convert str to list of words
    review = [ (lem.lemmatize(word)) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 70) # change to 200
bag = cv.fit_transform(corpus).toarray()    

y = data.iloc[:,4]
# naive bayes

from sklearn.naive_bayes import GaussianNB
nbclassifier = GaussianNB()
nbclassifier.fit(bag,y)


#testing
tcorpus = []

for i in range(len(testdata)):
    textrev = testdata['text'][i]
    review = re.sub('[^a-zA-Z]',' ' ,textrev) 
    review = review.lower()
    review = review.split() # convert str to list of words
    review = [ (lem.lemmatize(word)) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    tcorpus.append(review)
tbag = cv.fit_transform(tcorpus).toarray()    

res = nbclassifier.predict(tbag)
df1 = pd.DataFrame({'id':testdata.iloc[:,0],'target':res})
df1.to_csv('predict.csv',index=False)

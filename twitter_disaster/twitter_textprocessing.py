# -*- coding: utf-8 -*-
"""
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet') # lemmatizer
nltk.download('stopwords') # to remove stop words and the of an etc
data = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

# replacing everthing except english alphabets with a space
review = re.sub('[^a-zA-Z]',' ' ,data['text'][0]) 
review = review.lower()
review = review.split() # convert str to list of words

# removing stopwords

set () makes ot faster to search a word in stopwords 
using snowball stemmer stem()
snowball removes e from the en of words like forgive. using lemmatizer to prevent it

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
    review = [ port.stem(lem.lemmatize(word)) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 50) # change to 200
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
    review = [ port.stem(lem.lemmatize(word)) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    tcorpus.append(review)
tbag = cv.fit_transform(tcorpus).toarray()    

res = nbclassifier.predict(tbag)
df1 = pd.DataFrame({'id':testdata.iloc[:,0],'target':res})
df1.to_csv('predict.csv',index=False)
"""
import pandas as pd
import numpy as np
import re
#import spacy
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stopwords.add('im')
stopwords.add('u')
train = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#nlp = spacy.load('en_core_web_sm')
f = open('glove.6B.300d.txt','r')
embeddings_dict = {}
for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    embeddings_dict[word] = vector


train['keyword'].fillna('',inplace=True)
train = train.iloc[:,[1,3,4]]
i=0
glove_vects= []
for ind ,row in train.iterrows():
    #print(i)
    i+=1
    keyword = row['keyword']
    review = row['text']
    rev = [word for word in review.split() if not word in stopwords]
    review = ' '.join(rev)
    filtered_review = ''
    for word in review.split():
        word = word.lower()
        word = re.sub('[^a-zA-Z]','',word)
        if not word in stopwords and word.find('http')==-1:
            if word in embeddings_dict:
                filtered_review+=word.lstrip()+" "
    review = filtered_review
    r =''
    row['keyword'] = keyword.replace('%20',' ')
    row['text'] = review +" "+row['keyword']
    bag = [word for word in row['text'].split()]
    vec =[0]*300
    vec = np.asarray(vec)
    for i in range(len(bag)):
        vec = np.add(vec,embeddings_dict[bag[i]])
        vec = np.true_divide(vec,len(bag))
    glove_vects.append(vec)
    
glove_vects = np.asarray(glove_vects)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(glove_vects[:,:50],train.iloc[:,2])


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=17)
lr.fit(glove_vects[:,:50],train.iloc[:,2])

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(glove_vects[:,:50],train.iloc[:,2])

    
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=49)
clf.fit(glove_vects[:,:50],train.iloc[:,2])
testdata = pd.read_csv('test.csv')

testdata['keyword'].fillna('',inplace=True)
ids = testdata.iloc[:,0]

#data = data[31:]
testdata = testdata.iloc[:,[1,3]]
i=0
glove_vects_test= []
for ind ,row in testdata.iterrows():
    #print(i)
    i+=1
    keyword = row['keyword']
    #print(keyword.replace('%20',' '))
    review = row['text']
    rev = [word for word in review.split() if not word in stopwords]
    review = ' '.join(rev)
    filtered_review = ''
    for word in review.split():
        word = word.lower()
        word = re.sub('[^a-zA-Z]','',word)
       # print(word)
        #word = re.sub('\s+','',word)
        if not word in stopwords and word.find('http')==-1:
            if word in embeddings_dict:
                filtered_review+=word.lstrip()+" "
    review = filtered_review
    #rev = [re.sub('[^a-zA-Z]','',word) for word in review.split()]
    r =''
    #print(review)
    row['keyword'] = keyword.replace('%20',' ')
    row['text'] = review +" "+ row['keyword']
    
    bag = [word for word in row['text'].split()]
    vec =[0]*300
    vec = np.asarray(vec)
    
    for i in range(len(bag)):
        vec = np.add(vec,embeddings_dict[bag[i]])
        vec = np.true_divide(vec,len(bag))
        #vec+=embeddings_dict[bag[i]]
        #vec =vec/len(bag)
        #print(vec)
    glove_vects_test.append(vec)
glove_vects_test = np.asarray(glove_vects_test)
pred = clf.predict(glove_vects_test[:,:50])

pred = gnb.predict(glove_vects_test[:,:50])

pred = lr.predict(glove_vects_test[:,:50])

pred = lda.predict(glove_vects_test[:,:50])
'''
from sklearn.metrics import accuracy_score
yactual = np.asarray(testdata.iloc[:,1])
yactual = [int(i) for i in testdata.iloc[:,1]]
score = accuracy_score(testdata.iloc[:,2].values,pred)*100
'''
df1 = pd.DataFrame({'id':testdata.iloc[:,0],'target':pred})
df1.to_csv('predict.csv',index=False)
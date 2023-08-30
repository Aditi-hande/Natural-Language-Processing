#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore") 

import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.metrics import f1_score,recall_score,precision_score
from bs4 import BeautifulSoup
import contractions


# In[2]:


get_ipython().system(' pip install bs4')

from platform import python_version
print(python_version())

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


data=pd.read_csv("amazon_reviews_us_Beauty_v1_00.tsv",sep='\t',on_bad_lines='skip')


# ## Keep Reviews and Ratings

# In[4]:


df=data.loc[:,["review_body","star_rating"]]


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[5]:


def classfunc(star_):
    if star_=='5'or star_=='4':
        return 3
    elif star_=='3':
        return 2
    elif star_ =='2'or star_ =='1':
        return 1
    else:
        return 0

df['star_rating'] = df['star_rating'].astype(str)
df['class'] = df['star_rating'].apply(lambda x: classfunc(x[0]))
df.drop(df[(df['class'] == 0)].index, inplace=True) #dropping all entries with incorrect data

df_balanced = pd.DataFrame()
df_balanced = df.groupby(["class"]).apply(lambda grp: grp.sample(n=20000)) #selecting 2k entries from all three classes



# # Data Cleaning
# 
# 

# # Pre-processing

# In[6]:


before_datacleaning_len = df_balanced['review_body'].str.len().mean()

df_balanced['review_body'] = df_balanced['review_body'].str.lower()
df_balanced['review_body'] = df_balanced['review_body'].str.replace('http\S+|www.\S+', '', case=False)
df_balanced['review_body'] = df_balanced['review_body'].str.replace('[^a-zA-Z ]', '')
X=df_balanced.review_body
df_balanced['review_body'] = [BeautifulSoup(X).get_text() for X in df_balanced['review_body'].astype(str) ]
df_balanced['review_body'] = df_balanced['review_body'].str.strip()

df_balanced['review_body'] = df_balanced['review_body'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df_balanced['review_body'] = [' '.join(map(str, word)) for word in df_balanced['review_body']]

after_datacleaning_len = df_balanced['review_body'].str.len().mean()


print("Avg length of review character before and after data-cleaning: ", before_datacleaning_len,",",after_datacleaning_len)


# ## remove the stop words 

# In[7]:


before_preprocessing_len = df_balanced['review_body'].str.len().mean()

from nltk.corpus import stopwords

#stop words removal
english_stopwords = stopwords.words('english')
#df_balanced['review_body'] = [t for t in tokens if t not in english_stopwords]
df_balanced['review_body']= df_balanced['review_body'].apply(lambda x: [item for item in x.split() if item not in english_stopwords])
df_balanced['review_body'] = [' '.join(map(str, word)) for word in df_balanced['review_body']]

 


# ## perform lemmatization  

# In[8]:


#lemmatization

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_pos(word):
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

  return tag_dict.get(tag, wordnet.NOUN)

df_balanced['review_body'] = df_balanced['review_body'].apply(lambda x: [lemmatizer.lemmatize(w, get_pos(w)) for w in x.split()])
df_balanced['review_body'] = [' '.join(map(str, word)) for word in df_balanced['review_body']]

after_preprocessing_len = df_balanced['review_body'].str.len().mean()



print("Avg length of review character before and after preprocessing: ",before_preprocessing_len, ",", after_preprocessing_len)


# # TF-IDF Feature Extraction

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


tfidf = TfidfVectorizer(sublinear_tf=True,ngram_range=(1, 3),binary=True)


x = df_balanced['review_body']
y = df_balanced['class']

vectorized_x = tfidf.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(vectorized_x, y,test_size=0.2, stratify=y, random_state = 44)


# # Perceptron

# In[10]:


from sklearn.utils import multiclass
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report


model_perceptron = Perceptron(random_state=42)
model_perceptron.fit(X_train, y_train)

y_hat = model_perceptron.predict(X_test)

precision_perceptron = precision_score(y_test,y_hat,average=None)
recall_perceptron = recall_score(y_test,y_hat,average=None)
f1_perceptron = f1_score(y_test,y_hat,average=None)

print("Results from Perceptron model")

print("precision, recall, f1score for class 1: ", precision_perceptron[0], ", ", recall_perceptron[0], ", ", f1_perceptron[0])
print("precision, recall, f1score for class 2: ", precision_perceptron[1], ", ", recall_perceptron[1], ", ", f1_perceptron[1])
print("precision, recall, f1score for class 3: ", precision_perceptron[2], ", ", recall_perceptron[2], ", ", f1_perceptron[2])
print("precision, recall, f1score   average  : ", precision_score(y_test,y_hat,average='weighted'), ", ", recall_score(y_test,y_hat,average='weighted'), ", ", f1_score(y_test,y_hat,average='weighted'))



#print(classification_report(model_perceptron.predict(X_test), y_test))



# # SVM

# In[11]:


from sklearn.svm import LinearSVC

model_svc = LinearSVC(random_state=42)
model_svc.fit(X_train, y_train)

y_hat = model_svc.predict(X_test)

precision_svc = precision_score(y_test,y_hat,average=None)
recall_svc = recall_score(y_test,y_hat,average=None)
f1_svc = f1_score(y_test,y_hat,average=None)

print("Results from SVM model")


print("precision, recall, f1score for class 1: ", precision_svc[0], ", ", recall_svc[0], ", ", f1_svc[0])
print("precision, recall, f1score for class 2: ", precision_svc[1], ", ", recall_svc[1], ", ", f1_svc[1])
print("precision, recall, f1score for class 3: ", precision_svc[2], ", ", recall_svc[2], ", ", f1_svc[2])
print("precision, recall, f1score   average  : ", precision_score(y_test,y_hat,average='weighted'), ", ", recall_score(y_test,y_hat,average='weighted'), ", ", f1_score(y_test,y_hat,average='weighted'))


#print(classification_report(model_svc.predict(X_test), y_test))


# # Logistic Regression

# In[12]:


from sklearn.linear_model import LogisticRegression

model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_train, y_train)

y_hat = model_logistic.predict(X_test)

precision_logistic = precision_score(y_test,y_hat,average=None)
recall_logistic = recall_score(y_test,y_hat,average=None)
f1_logistic = f1_score(y_test,y_hat,average=None)

print("Results from Logistic Regression model")


print("precision, recall, f1score for class 1: ", precision_logistic[0], ", ", recall_logistic[0], ", ", f1_logistic[0])
print("precision, recall, f1score for class 2: ", precision_logistic[1], ", ", recall_logistic[1], ", ", f1_logistic[1])
print("precision, recall, f1score for class 3: ", precision_logistic[2], ", ", recall_logistic[2], ", ", f1_logistic[2])
print("precision, recall, f1score   average  : ", precision_score(y_test,y_hat,average='weighted'), ", ", recall_score(y_test,y_hat,average='weighted'), ", ", f1_score(y_test,y_hat,average='weighted'))


#print(classification_report(model_logistic.predict(X_test), y_test))


# # Naive Bayes

# In[13]:


from sklearn.naive_bayes import MultinomialNB

model_mnb = MultinomialNB()
model_mnb.fit(X_train,y_train)

y_hat = model_mnb.predict(X_test)

precision_mnb = precision_score(y_test,y_hat,average=None)
recall_mnb = recall_score(y_test,y_hat,average=None)
f1_mnb = f1_score(y_test,y_hat,average=None)

print("Results from Multinomial Naive Bayes model")


print("precision, recall, f1score for class 1: ", precision_mnb[0], ", ", recall_mnb[0], ", ", f1_mnb[0])
print("precision, recall, f1score for class 2: ", precision_mnb[1], ", ", recall_mnb[1], ", ", f1_mnb[1])
print("precision, recall, f1score for class 3: ", precision_mnb[2], ", ", recall_mnb[2], ", ", f1_mnb[2])
print("precision, recall, f1score   average  : ", precision_score(y_test,y_hat,average='weighted'), ", ", recall_score(y_test,y_hat,average='weighted'), ", ", f1_score(y_test,y_hat,average='weighted'))


#print(classification_report(model_mnb.predict(X_test), y_test))


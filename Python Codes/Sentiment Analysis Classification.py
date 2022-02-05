#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Importing necessary packages

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[10]:


# Setting Work Directory

os.chdir(r'C:\Users\rifay\OneDrive\Documents\MSIS 5193\Project\Final Datasets')


# In[11]:


# Reading data from csv file, this dataset comes from our R sentiment Analysis

positive_negative_data = pd.read_csv('positive_negative.csv', sep=",", encoding='cp1252')


# In[38]:


# Creating feautres and labels

features = positive_negative_data['word']

stop = stopwords.words('english')

vectorizer = TfidfVectorizer(max_features=5600, min_df=1, max_df=0.001, stop_words=stop)

processed_features = vectorizer.fit_transform(features).toarray()

labels = positive_negative_data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# In[39]:


# Creating Predictive Model

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)


# In[41]:


# Printing Model Results

print(classification_report(y_test,predictions))


# In[ ]:





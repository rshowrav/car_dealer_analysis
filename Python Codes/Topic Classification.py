# Importing necessary packages

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Setting Work Directory

os.chdir(r'C:\Users\rifay\OneDrive\Documents\MSIS 5193\Project\Final Datasets')


# Reading data from csv file and reading column names

dealer_review_data = pd.read_csv('dealer_review_data.csv', sep=",", encoding='cp1252')
dealer_review_data.columns

## Removing Numerical Values and puntuation

patterndigits = '\\b[0-9]+\\b'
dealer_review_data['review'] = dealer_review_data['review'].str.replace(patterndigits,'')

patternpunc = '[^\w\s]'
dealer_review_data['review'] = dealer_review_data['review'].str.replace(patternpunc,'')


# Using Latent Dirichlet allocation (LDA) create 10 topics

vectorizer = CountVectorizer(max_df=0.8, min_df=4, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(dealer_review_data['review'].values.astype('U'))
doc_term_matrix.shape
LDA = LatentDirichletAllocation(n_components=10, random_state=35)
LDA.fit(doc_term_matrix)

for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i+1}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# Adding LDA topic column to dataset 

topic_values = LDA.transform(doc_term_matrix)

topic_values.shape

dealer_review_data['topic'] = topic_values.argmax(axis=1) + 1


# Getting a count of LDA topic frequencies

values = dealer_review_data['topic'].value_counts().sort_index().to_frame()

print(values)


# In[ ]:





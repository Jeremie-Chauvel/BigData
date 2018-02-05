#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:59:56 2018

@author: Marine
"""



#import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

# Returns a DataFrame of testdata
df_test = pd.read_excel('test_dataset .xlsx', sheet_name='Sheet1', header=None, names=['Comments'])

# Returns a DataFrame of training data
df = pd.read_csv('training_dataset.txt', sep="	", header=None, names=["target", "data"])

df.target_names = ['negative','positive']



#from nltk.classify.scikitlearn import SklearnClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#classif = SklearnClassifier(GaussianNB())

#Tokenizing

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df.data)
X_train_counts.shape


#From occurances to frequencies
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


clf = MultinomialNB().fit(X_train_tfidf, df.target)


#docs_new_initial = ['I loved it !', 'breakfast was amazing', 'I hated it', 'I dont know what to think of it']

docs_new = list(df_test.Comments)
#docs_new = [x.encode('UTF8') for x in docs_new]
docs_new = [x.encode('ascii','ignore') for x in docs_new]

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)


import datetime
time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
file_name_csv = 'out_' + time + '.csv'
file_name_xlsx = 'out_' + time + '.xlsx'

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, df.target_names[category]))
    #Print to csv file
    with open(file_name_csv, 'a') as f:
        print >> f, doc,";",df.target_names[category]

df_out = DataFrame({'Comments': docs_new, 'Predicted': predicted})
df_out.to_excel(file_name_xlsx, sheet_name='sheet1', index=False)

#classif.train(df)
#print("GaussianNB accuracy percent:",nltk.classify.accuracy(GaussianNB, testing_set))


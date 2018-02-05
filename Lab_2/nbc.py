#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:59:56 2018

@author: Marine
"""

import pandas as pd
from pandas import DataFrame

# Returns a DataFrame of testdata
df_test = pd.read_excel('test_dataset .xlsx', sheet_name='Sheet1', header=None, names=['Comments'])

# Returns a DataFrame of training data
df = pd.read_csv('training_dataset.txt', sep="	", header=None, names=["target", "data"])

df.target_names = ['negative','positive']



from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])

text_clf.fit(df.data, df.target)


docs_test = list(df_test.Comments)
docs_test = [x.encode('ascii','ignore') for x in docs_test]

predicted = text_clf.predict(docs_test)


import datetime
time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
file_name_csv = 'out_' + time + '.csv'
file_name_xlsx = 'out_' + time + '.xlsx'

for doc, category in zip(docs_test, predicted):
    print('%r => %s' % (doc, df.target_names[category]))
    #Print to csv file
    with open(file_name_csv, 'a') as f:
        print >> f, doc,";",df.target_names[category]

df_out = DataFrame({'Comments': docs_test, 'Predicted': predicted})
df_out.to_excel(file_name_xlsx, sheet_name='sheet1', index=False)


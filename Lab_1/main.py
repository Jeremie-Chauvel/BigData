import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


trainData = pd.read_csv("lab_train.txt")
testData = pd.read_csv("lab_test.txt")
#preprocessData
def create_label(df):
    df['score_str'] = "positive"
    df.loc[df['score'] <= 2.0, 'score_str'] = "negative"
    df.loc[df['score'] == 3.0, 'score_str'] = "neutral"

create_label(trainData)
create_label(testData)
#building pipelne
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                     ])

# Train the model using the training sets
text_clf.fit(trainData.review, trainData.score_str)

#predict

predicted = text_clf.predict(testData.review)

print("accuracy : "+str(np.mean(predicted == testData.score_str)))
#
# df = pd.DataFrame(predicted, index=range(len(predicted)),columns=["predicted"])
# result = pd.concat([testData,df], axis=1, join_axes=[testData.index])
# for k in range(len(predicted)):
#     if predicted[k:k + 1][0] != testData[k:k+1]["score"].values[0]:
#         print("Predicted value : {}  || actual value : {} -dif".format(predicted[k:k+1][0],testData[k:k+1]["score"].values[0]))
#     else:
#         print("Predicted value : {}  || actual value : {} -equ".format(predicted[k:k + 1][0],testData[k:k + 1]["score"].values[0]))
#
#
#
#

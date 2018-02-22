import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


data = pd.read_csv("training_dataset.txt",sep='\t',names=['score','review'])
# careful ! the original "test_dataset .xlsx" have a space in the file name !
try :
    predictData = pd.read_excel("test_dataset .xlsx",names=["review"])
except (FileNotFoundError):
    predictData = pd.read_excel("test_dataset.xlsx", names=["review"])
# preprocessData
def create_label(df):
    df['score_str'] = "positive"
    df.loc[df['score'] == 0, 'score_str'] = "negative"

create_label(data)
# split the labeled data between training set and test set
trainData = data.sample(frac=0.8) # recup 80% for training
testData = data.drop(trainData.index) # 20% of the rest for testing

# building pipelne
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC()),
                     ])

# Train the model using the training sets
text_clf.fit(trainData.review, trainData.score_str)

# predict for test set

predicted = text_clf.predict(testData.review)
# compute accuracy
print('#######')
print("\naccuracy : "+str(np.mean(predicted == testData.score_str)))

# prediction for unlabeled data :
print('#######')
print('predictions for unlabel data')
predict = pd.DataFrame(text_clf.predict(predictData.review),columns=["predicted"])
result = pd.concat([predictData,predict], axis=1, join_axes=[predictData.index])
print(result.head())

# saving new data to csv file :
result.to_csv("test_set_with_prediction.csv", sep=';', encoding='utf-8')

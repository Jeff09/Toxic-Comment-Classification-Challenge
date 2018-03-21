# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from scipy.sparse import hstack
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('data/train.csv').fillna(' ')
test = pd.read_csv('data/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']

# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('data/glove.840B.300d.txt', encoding='utf-8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    assert len(values[-300:]) == 300
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set
train_glove = [sent2vec(x) for x in tqdm(train_text)]
test_glove = [sent2vec(x) for x in tqdm(test_text)]

train_glove = np.array(train_glove)
test_glove = np.array(test_glove)

losses = []
predictions = {'id': test['id']}
for class_name in class_names:
    train_target = train[class_name]
    classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)

    cv_loss = np.mean(cross_val_score(classifier, train_glove, train_target, cv=5, scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_glove, train_target)
    predictions[class_name] = classifier.predict_proba(test_glove)[:, 1]

print('Total CV score is {}'.format(np.mean(losses)))

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('xgb_wordvector.csv', index=False)

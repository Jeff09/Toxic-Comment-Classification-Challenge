# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('data/train.csv').fillna(' ')
test = pd.read_csv('data/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=20000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

losses = []
predictions = {'id': test['id']}
for class_name in class_names:
    train_target = train[class_name]
    #classifier = LogisticRegression(solver='saga')
    classifier = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=2018, warm_start =True )

    cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc', n_jobs =-1))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(losses)))

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('RF_words_ngram_cv3.csv', index=False)
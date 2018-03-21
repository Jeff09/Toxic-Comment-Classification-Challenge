# -*- coding: utf-8 -*-

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, BatchNormalization, SpatialDropout1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import gc
import gensim.models.keyedvectors as word2vec
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 50000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 400
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

word_index = tokenizer.word_index


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)

losses = []
predictions = {'id': test['id']}
for class_name in list_classes:
    cvscores = []
    y = train[class_name]
    for train, valid in kfold.split(X_t, y):        
        # GRU with glove embeddings and two dense layers
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                             300,
                             weights=None,
                             input_length=maxlen,
                             trainable=True))
        model.add(SpatialDropout1D(0.3))
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))
        
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.8))
        
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.8))
        
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Fit the model with early stopping callback
        earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
        History = model.fit(X_t[train], y[train], batch_size=320, epochs=10, verbose=1, callbacks=[earlystop])
        scores = model.evaluate(X_t[valid], y[valid], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))      

    model.fit(X_t, y, batch_size=320, epochs=100, verbose=1, callbacks=[earlystop])
    for class_name in list_classes:
        print('fit', class_name)
        predictions[class_name] = model.predict(X_te)[:, 1]

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('two_gru.csv', index=False)



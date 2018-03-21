# -*- coding: utf-8 -*-

"""
Fork of https://www.kaggle.com/antmarakis/bi-lstm-conv-layer?scriptVersionId=2789290
Just replaced the data with Preprocessed data
Public LB score 0.9833 => 0.9840

"""
import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

EMBEDDING_FILE = 'data/glove.840B.300d.txt'
train_x = pd.read_csv('data/cleaned-toxic-comments/train_preprocessed.csv').fillna(" ")
test_x = pd.read_csv('data/cleaned-toxic-comments/test_preprocessed.csv').fillna(" ")

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))

max_features=100000
maxlen=150
embed_size=300

train_x['comment_text'].fillna(' ')
test_x['comment_text'].fillna(' ')
train_y = train_x[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
train_x = train_x['comment_text'].str.lower()

test_x = test_x['comment_text'].str.lower()


# Vectorize text + Prepare GloVe Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Build Model
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.35)(x)

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])

out = Dense(6, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prediction
batch_size = 32
epochs = 1

X_tra, X_val, y_tra, y_val = train_test_split(train_x, train_y, test_size=0.1)



RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)    
    
# prepare callbacks
model_path = 'biLSTM_CONV.h5'
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        #mode='max',
        verbose=1),
    ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        #mode='max',
        verbose=1),
    RocAuc         
]

history = model.fit(X_tra, y_tra, batch_size=128, epochs=10, validation_data=(X_val, y_val),
                 callbacks=callbacks, verbose=2, shuffle=True)
    
# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    model = load_model(model_path)

predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

submission = pd.read_csv('data/sample_submission.csv')
submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = predictions
submission.to_csv('submit/bilstm_conv.csv', index=False)
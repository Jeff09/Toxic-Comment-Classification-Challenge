# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate, SpatialDropout1D
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from sklearn.metrics import roc_auc_score

# define network parameters
max_features = 64
maxlen = 200
embed_size = 300

train = pd.read_csv('data/cleaned-toxic-comments/train_preprocessed.csv')
test = pd.read_csv('data/cleaned-toxic-comments/test_preprocessed.csv')
submission = pd.read_csv('data/sample_submission.csv')

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

train = standardize_text(train, "comment_text")
list_sentences_train = train["comment_text"].fillna("unknown").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values


tokenizer = keras_text.Tokenizer(char_level = True)
tokenizer.fit_on_texts(list(list_sentences_train))
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = keras_seq.pad_sequences(list_tokenized_train, maxlen=maxlen)

# test data
test = standardize_text(test, "comment_text")
list_sentences_test = test["comment_text"].fillna("unknown").values
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = keras_seq.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

base_path_input = 'data/'
EMBEDDING_FILE = base_path_input + 'fasttext-crawl-300d-2m/crawl-300d-2M.vec' 

# BUILD EMBEDDING MATRIX    
print('Preparing embedding matrix...')
# Read the FastText word vectors (space delimited strings) into a dictionary from word->vector
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))
print("embeddings_index size: ", len(embeddings_index))

# https://github.com/uclmr/emoji2vec
# e2v = gsm.Word2Vec.load_word2vec_format('emoji2vec.bin', binary=True)

word_index = tokenizer.word_index
print("word_index size: ", len(word_index))   
words_not_found = []
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():        
    if i >= max_features: 
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def build_model(conv_layers = 4, 
                dilation_rates = [0, 2, 4, 8, 16], 
                embed_size = 300):
    inp = Input(shape=(None, ))
    x = Embedding(input_dim = len(tokenizer.word_counts), weights=[embedding_matrix], 
                  output_dim = embed_size)(inp)
    prefilt_x = SpatialDropout1D(0.2)(x)
    #prefilt_x = Dropout(0.2)(x)
    out_conv = []
    # dilation rate lets us use ngrams and skip grams to process 
    for dilation_rate in dilation_rates:
        x = prefilt_x
        for i in range(conv_layers):
            if dilation_rate>0:
                x = Conv1D(16*2**(i), 
                           kernel_size = 7, 
                           dilation_rate = dilation_rate,
                          activation = 'relu',
                          name = 'ngram_{}_cnn_{}'.format(dilation_rate, i)
                          )(x)
            else:
                x = Conv1D(16*2**(i), 
                           kernel_size = 1,
                          activation = 'relu',
                          name = 'word_fcl_{}'.format(i))(x)
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = build_model()
model.summary()

from sklearn.model_selection import train_test_split
any_category_positive = np.sum(y,1)
print('Distribution of Total Positive Labels (important for validation)')
print(pd.value_counts(any_category_positive))
X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y, 
                                                        test_size = 0.1, 
                                                        stratify = any_category_positive,
                                                       random_state = 2017)
print('Training:', X_t_train.shape)
print('Testing:', X_t_test.shape)

batch_size = 128 # large enough that some other labels come in
epochs = 50

file_path="best_weights.h5"
RocAuc = RocAucEvaluation(validation_data=(X_t_test, y_test), interval=1)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        #mode='max',
        verbose=1),
    ModelCheckpoint(
        file_path,
        monitor='val_loss',
        save_best_only=True,
        #mode='max',
        verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2 , min_lr=0.00001),
    RocAuc         
]

model.fit(X_t_train, y_train, 
          validation_data=(X_t_test, y_test),
          batch_size=batch_size, 
          epochs=epochs, 
          shuffle = True,
          callbacks=callbacks)

model.load_weights(file_path)
y_test = model.predict(X_te)
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("characterCNN.csv", index=False)
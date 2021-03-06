# -*- coding: utf-8 -*-

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import os
from numpy import mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, BatchNormalization, GlobalMaxPooling1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Activation, SpatialDropout1D 
from keras.optimizers import Adam
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import nltk as nlp
import gensim.models as gsm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

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

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def get_model(maxlen, max_features, embedding_dim, embedding_matrix, filter_sizes = [2, 3, 4, 5, 6], num_filters = 64,  lr=1e-3, lr_d=0, units=128, dr=0.2):
        inp = Input(shape=(maxlen,)) 
        x = Embedding(max_features, embedding_dim, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(dr)(x)
        
        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(filters=num_filters, kernel_size=sz, padding='same')(x)
            conv = BatchNormalization()(conv)
            conv = Activation("relu")(conv)
            conv = Dropout(dr)(conv)
            #
            conv = Conv1D(filters=num_filters, kernel_size=sz, padding='same')(x)
            conv = BatchNormalization()(conv)
            conv = Activation("relu")(conv)
            conv = Dropout(dr)(conv)
            
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        conc = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        
        outp = Dense(6, activation="sigmoid")(conc)
        
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr, decay=lr_d),
                      metrics=['accuracy']) 
    
        print(model.summary())
        
        return model
    
if __name__ == "__main__":    
    
    # nlp.download()    
    bad_words = ['shit', 'fuck', 'damn', 'bitch', 'crap', 'piss', 'dick', 'darn', 'cock', 'pussy', 'ass', 'asshole', 'fag', 'bastard', 'slut', 'douche', 'bastard', 'darn', 'bloody', 'bugger', 'bollocks', 'arsehole', 'nigger', 'nigga', 'moron', 'gay', 'antisemitism', 'anti', 'nazi', 'poop']
       
    base_path_input = 'data/'
    base_path_output = ''
    # define path to save model
    model_path = base_path_output + 'textCNN.h5'
    
    EMBEDDING_FILE = base_path_input + 'fasttext-crawl-300d-2m/crawl-300d-2M.vec'      
    train = pd.read_csv(base_path_input + 'cleaned-toxic-comments/train_preprocessed.csv')
    test = pd.read_csv(base_path_input + 'cleaned-toxic-comments/test_preprocessed.csv') 
    

    print("num train: ", train.shape[0])
    print("num test: ", test.shape[0])     
    
    embed_size = 300  # how big is each word vector
    max_features = 100000  # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 150  # max number of words in a comment to use
    
    stop_words = set(stopwords.words('english'))     
     
    #tokenizer = RegexpTokenizer(r'[a-zA-Z]+')   
    #tokenizer = RegexpTokenizer(r'[\s|\n|\t]+')  
    
    ####################################################
    # DATA PREPARATION
    ####################################################  
    
    X_train = train["comment_text"].fillna("fillna").values
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    X_test = test["comment_text"].fillna("fillna").values
    
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    '''
    # TRAIN
    train["comment_text"].fillna('_NA_')
    #train = standardize_text(train, "comment_text")
    train["tokens"] = train["comment_text"].apply(tokenizer.tokenize)
    # delete Stop Words
    train["tokens"] = train["tokens"].apply(lambda vec: [word for word in vec if word not in stop_words])
    # Normalize Bad Words    
    #train["tokens"] = train["tokens"].apply(lambda vec: normalize_bad_word(vec, bad_words))
    #train.to_csv(base_path_output + 'train_normalized.csv', index=False)
    
    all_training_words = [word for tokens in train["tokens"] for word in tokens]
    training_sentence_lengths = [len(tokens) for tokens in train["tokens"]]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
    print("Max sentence length is %s" % max(training_sentence_lengths))
    print("Min sentence length is %s" % min(training_sentence_lengths))
    print("Mean sentence length is %s" % mean(training_sentence_lengths))
    
    train["tokens"] = train["tokens"].apply(lambda vec :' '.join(vec))
    print("num train: ", train.shape[0])
    print(train.head())
    
    # TEST    
    test["comment_text"].fillna('_NA_')
    test = standardize_text(test, "comment_text")
    test["tokens"] = test["comment_text"].apply(tokenizer.tokenize)
    # delete Stop Words
    test["tokens"] = test["tokens"].apply(lambda vec: [word for word in vec if word not in stop_words])
    # Normalize Bad Words
    # test["tokens"] = test["tokens"].apply(lambda vec: normalize_bad_word(vec, bad_words))    
    # test.to_csv(base_path_output + 'test_normalized.csv', index=False)
    
    all_test_words = [word for tokens in test["tokens"] for word in tokens]
    test_sentence_lengths = [len(tokens) for tokens in test["tokens"]]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
    print("Max sentence length is %s" % max(test_sentence_lengths))
    print("Min sentence length is %s" % min(test_sentence_lengths))
    print("Mean sentence length is %s" % mean(test_sentence_lengths))
    
    test["tokens"] = test["tokens"].apply(lambda vec :' '.join(vec))
    print("num test: ", test.shape[0])
    print(test.head())
    
    # Turn each comment into a list of word indexes of equal length (with truncation or padding as needed)
    list_sentences_train = train["tokens"].values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["tokens"].values
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)'''
    
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
    print("sample words not found: ", np.random.choice(words_not_found, 10))
    df = pd.DataFrame(words_not_found)
    #df.to_csv(base_path_output + "word_not_found.csv", header=None, index=False)
    
    #pd.DataFrame(embedding_matrix).to_csv(base_path_output + "embedding_matrix.csv", header=None, index=False)
    
    ####################################################
    # MODEL
    ####################################################  
       
    model = get_model(maxlen, max_features, embed_size, embedding_matrix, lr=1e-3, lr_d=0, units=128, dr=0.5)
    
    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, test_size=0.1)
    
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)    
    
    # prepare callbacks
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
        ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2 , min_lr=0.00001),
        RocAuc         
    ]
    
    history = model.fit(X_tra, y_tra, batch_size=128, epochs=20, validation_data=(X_val, y_val),
                     callbacks=callbacks, verbose=2, shuffle=True)
    '''
    # summarize history for Accuracy
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('Acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    fig_acc.savefig(base_path_output + "model_accuracy.png")
    
    # summarize history for loss
    fig_loss = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    fig_loss.savefig(base_path_output + "model_loss.png")'''
    
    # if best iteration's model was saved then load and use it
    if os.path.isfile(model_path):
        estimator = load_model(model_path)
    y_test = estimator.predict([x_test], batch_size=1024, verbose=2)
    sample_submission = pd.read_csv(base_path_input + "sample_submission.csv")
    sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
    sample_submission.to_csv(base_path_output + 'textCNN.csv', index=False)
# -*- coding: utf-8 -*-

#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
#from wordcloud import WordCloud ,STOPWORDS
#from PIL import Image
#import matplotlib_venn as venn

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
#import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
from tqdm import tqdm


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from spell import *

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

train = pd.read_csv('data/train_nodate.csv')
test = pd.read_csv('data/test_nodate.csv')

with open("data/pronous_list.txt", "r") as f:
    pronous = f.read()

pronouslist = pronous.lower().split(", ")

def get_cleanData(data):
    word_freq_dict = {}
    total_words = 0
    total_clean_words = 0
    
    for sentence in tqdm(data):
        #words = find_words(str(sentence))
        word_sen = str(sentence).strip().lower().split()
        #toxic_words = tokenizer.tokenize(str(sentence).lower())
        total_words += len(word_sen)
        for word in word_sen:
            if word not in eng_stopwords and word not in pronouslist:
                #if re.match("(\w*)(\.|\?|\\|\,|\*|\!|\~|\;|\-|\(|\[|\{|\@|\#|\$|\%|\^|\&|\)|\_|\=|\/|\|\:|\"|\'|\<|\>|\`)(\w*)", word):
                #    word = re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]', '', word)
                #    if word not in WORDS:
                #        word = correction(word)
                if word not in eng_stopwords and word not in pronouslist:
                    total_clean_words += 1
                    #word = re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]', '', word)
                    word = lem.lemmatize(word)
                    if word not in word_freq_dict:
                        word_freq_dict[word] = 0
                    word_freq_dict[word] += 1
    print("total words:", total_words)
    print("total clean words:", total_clean_words)
    print("total distinct clean words:", len(word_freq_dict))
    return word_freq_dict

import os
def store_file(word_freq_dict, filename):
    w_f ="\n".join([str(k)+' '+" " + str(v) for k,v in word_freq_dict.items()]) 
    with open(os.path.join('data',filename), 'w', encoding='utf-8') as f:  # Just use 'w' mode in 3.x
        f.write(w_f)

severe_toxic_data = train[train['severe_toxic'] > 0]['fewer_dates']
severe_toxic_word_freq_dict = get_cleanData(severe_toxic_data)
store_file(severe_toxic_word_freq_dict, "dsevere_toxic_clean_data.csv")

obscene_data = train[train['obscene'] > 0]['fewer_dates']
obscene_word_freq_dict = get_cleanData(obscene_data)
store_file(obscene_word_freq_dict, "dobscene_clean_data.csv")

threat_data = train[train['threat'] > 0]['fewer_dates']
threat_word_freq_dict = get_cleanData(threat_data)
store_file(threat_word_freq_dict, "threat_clean_data.csv")

insult_data = train[train['insult'] > 0]['fewer_dates']
insult_word_freq_dict = get_cleanData(insult_data)
store_file(insult_word_freq_dict, "insult_clean_data.csv")

identity_hate_data = train[train['identity_hate'] > 0]['fewer_dates']
identity_hate_word_freq_dict = get_cleanData(identity_hate_data)
store_file(identity_hate_word_freq_dict, "identity_hate_clean_data.csv")

toxic_data = train[train['toxic'] > 0]['fewer_dates']
toxic_word_freq_dict = get_cleanData(toxic_data)
store_file(toxic_word_freq_dict, "toxic_clean_data.csv")
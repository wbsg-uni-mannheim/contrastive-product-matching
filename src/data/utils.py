import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords

from copy import deepcopy

import re
PATTERN1 = re.compile("\"@\S+\s+")
PATTERN2 = re.compile("\s+")

def clean_string_wdcv2(words):
    if not words:
        return None
    words = words.partition('"')[2]
    words = words.rpartition('"')[0]
    words = re.sub(PATTERN1, ' ', words)
    words = re.sub(PATTERN2, ' ', words)
    words = words.replace('"', '')
    words = words.strip()
    return words

def clean_specTableContent_wdcv2(words):
    if not words:
        return None
    words = re.sub(PATTERN2, ' ', words)
    words = words.strip()
    return words

def tokenize(words, delimiter=None):
    #check for NaN
    if isinstance(words, float):
        if words != words:
            return []
    words = str(words)
    return words.split(sep=delimiter)

def remove_stopwords(words, lower=False):
    #check for NaN
    if isinstance(words, float):
        if words != words:
            return words
    stop_words_list = deepcopy(stopwords.words('english'))
    if lower:
        stop_words_list = list(map(lambda x: x.lower(), stop_words_list))
    word_list = tokenize(words)
    word_list_stopwords_removed = [x for x in word_list if x not in stop_words_list]
    words_processed = ' '.join(word_list_stopwords_removed)
    return words_processed

def stem(words):
    stemmer = PorterStemmer()
    word_list = tokenize(words)
    stemmed_words = [stemmer.stem(x) for x in word_list]
    words_processed = ' '.join(stemmed_words)
    return words_processed
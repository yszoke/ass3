import nltk
# nltk.download("stopwords")
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# nltk.download('punkt')
# nltk.download('sentiwordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.corpus import sentiwordnet as swn

def lower_case_review(review):
    new_review=""
    lower_case_tokens=[]
    words = str.split(review)
    string_check= re.compile("[',@_!#$%.``^&*()<>?/\|}{""~:]")
    for token in words:
        token =token.lower()
        if(string_check.search(token)==None):
            lower_case_tokens.append(token)
        elif token.startswith("@"):
            continue
        else:
            for ch in ['\\',",", '`', '*', '_', '{', '}', '[', ']','&','/','?','~',':','^',
                       '(', ')', '>','<', '#', '+', '-', '.', '!', '$', '\'']:
                if ch in token:
                    token = token.replace(ch,"")
            if(token!="" and token!=" "):
                lower_case_tokens.append(token)
    new_review=" ".join(lower_case_tokens)
    return new_review

def remove_stop_words_review(review):
    stop_words=set(stopwords.words('english'))
    tokens_without_stop_words=[]
    split_review=str.split(review)
    for w in split_review:
        if w not in stop_words:
            tokens_without_stop_words.append(w)
    clean_review=" ".join(tokens_without_stop_words)
    return clean_review


def stemming_review(review):
    ps = PorterStemmer()
    stemming_words=[]
    words_review=str.split(review)
    for token in words_review:
        stemming_token=ps.stem(token)
        stemming_token=stemming_token.replace(".","")
        if(stemming_token=="." or stemming_token=="br" or stemming_token=="'s" or stemming_token==""):
            continue
        stemming_words.append(stemming_token)
    stemmed_review=" ".join(stemming_words)
    return stemmed_review

def getData(tweets):
    new = []
    for value in tweets:
        new.append(stemming_review(remove_stop_words_review(lower_case_review(value))))
    return new

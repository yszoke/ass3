import nltk
# nltk.download("stopwords")
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import sklearn
from sklearn.model_selection import train_test_split
# nltk.download('punkt')
# nltk.download('sentiwordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.corpus import sentiwordnet as swn



def lower_case(tokens):
    lower_case_tokens = []
    string_check = re.compile("[',@_!#$%.``^&*()<>?/\|}{""~:]")
    for token in tokens:
        lower_case_token=token[0].lower()
        if(string_check.search(lower_case_token)==None):
            lower_case_tokens.append((lower_case_token,token[1]))
        elif lower_case_token.startswith("@"):
            continue
        else:
            for ch in ['\\',",", '`', '*', '_', '{', '}', '[', ']','&','/','?','~',':','^',
                       '(', ')', '>','<', '#', '+', '-', '.', '!', '$',"'"]:
                if ch in lower_case_token:
                    lower_case_token = lower_case_token.replace(ch,"")
            if(lower_case_token!="" and lower_case_token!=" "):
                lower_case_tokens.append((lower_case_token, token[1]))
    return lower_case_tokens

def remove_stop_words(tokens):
    stop_words=set(stopwords.words('english'))
    tokens_without_stop_words=[]
    for w in tokens:
        words= str.split(w[0])
        if (len(words)>1):
            if words[0] in stop_words and words[1] in stop_words:
                continue
            else:
                tokens_without_stop_words.append((w[0],w[1]))
#             elif words[0] in stop_words and words[1] not in stop_words:
#                 tokens_without_stop_words.append((words[1],w[1]))
#             elif words[0] not in stop_words and words[1] in stop_words:
#                 tokens_without_stop_words.append((words[0],w[1]))
        else:
            if w[0] not in stop_words:
                tokens_without_stop_words.append((w[0],w[1]))
#     tokens_without_stop_words = [w for w in tokens if not w[0] in stop_words]
    return tokens_without_stop_words

def stemming(tokens):
    ps = PorterStemmer()
    stemming_words=[]
    for token in tokens:
        stemming_token=ps.stem(token[0])
        stemming_token=stemming_token.replace(".","")
        if(stemming_token=="." or stemming_token=="nt" or stemming_token=="'s" or stemming_token==""):
            continue
        stemming_words.append([stemming_token,token[1]])
    return stemming_words


corpus = pd.read_csv("Train.csv",encoding='latin-1')
x_train=corpus.SentimentText
y_train = corpus.Sentiment

sentiment_words_train={}

tokens_in_train_set=[]
for cellX, cellY in zip(x_train,y_train):
    cell_tokens= word_tokenize(cellX)
    label=cellY
    if label=="sentiment":
        continue
    words_in_cell=len(cell_tokens)
    if label in sentiment_words_train:
        sentiment_words_train[label]=sentiment_words_train[label]+words_in_cell
    else:
        sentiment_words_train[label]=words_in_cell
    length= len(cell_tokens)
    for word in cell_tokens[1:length]:
        tokens_in_train_set.append((word,label))
    bigrams_train = list(nltk.bigrams(cellX.split()))
    for bigrm in bigrams_train:
        word = bigrm[0]+" " +bigrm[1]
        tokens_in_train_set.append((word,label))


clean_tokens_test=stemming(remove_stop_words(lower_case(tokens_in_train_set)))
print(clean_tokens_test[20:40])

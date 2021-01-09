import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from nltk.stem.porter import *


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

# function to collect hashtags

def hashtag_extract(x):
    hashtags = []    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

def word_vector(model_w2v, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

def exe():
    pd.set_option("display.max_colwidth", 200)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Let’s read train and test datasets.

    train  = pd.read_csv('Train.csv',encoding='latin-1')
    test = pd.read_csv('Test.csv',encoding='latin-1')

    combi = train.append(test, ignore_index=True, sort=True)

    combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['SentimentText'], "@[\w]*")

    combi.tidy_tweet = combi.tidy_tweet.str.replace("[^a-zA-Z#]", " ")

    combi.tidy_tweet = combi.tidy_tweet.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    tokenized_tweet = combi.tidy_tweet.apply(lambda x: x.split())

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

    # Now let’s stitch these tokens back together. It can easily be done using nltk’s MosesDetokenizer function.
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    combi['tidy_tweet'] = tokenized_tweet

    # extracting hashtags from non racist/sexist tweets
    HT_regular = hashtag_extract(combi['tidy_tweet'][combi['Sentiment'] == 0])

    import gensim

    tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())  # tokenizing

    model_w2v = gensim.models.Word2Vec(
        tokenized_tweet,
        size=150,  # desired no. of features/independent variables
        window=5,  # context window size
        min_count=2,  # Ignores all words with total frequency lower than 2.
        sg=1,  # 1 for skip-gram model
        hs=0,
        negative=10,  # for negative sampling
        workers=32,  # no.of cores
        seed=34
    )

    model_w2v.train(tokenized_tweet, total_examples=len(combi['tidy_tweet']), epochs=20)

    wordvec_arrays = np.zeros((len(tokenized_tweet), 150))
    for i in range(len(tokenized_tweet)):
        wordvec_arrays[i, :] = word_vector(model_w2v, tokenized_tweet[i], 150)
    wordvec_df = pd.DataFrame(wordvec_arrays)
    wordvec_df.shape

    train_w2v = wordvec_df.iloc[:79998, :]
    test_w2v = wordvec_df.iloc[79998:, :]

    return train_w2v,test_w2v

if __name__ == "__main__":
    ghgh()
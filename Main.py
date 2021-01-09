# import nltk
# nltk.download('stopwords')
import scipy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import Perceptron

import Preprocessing as pre
# from word2vec import execute
def test():

    df = pd.read_csv('Train.csv',encoding='latin-1')
################################################################
    df['length'] = check_happy(df['SentimentText'].values)

    df['processedtext'] = np.array(pre.getData(df['SentimentText'].array))
    # word2vec_train, word2vec_test=execute()
    target = df['Sentiment']

################################################################

    X_train, X_test, y_train, y_test = train_test_split(df['processedtext'], target, test_size=0.20, random_state=100)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df['length'], target, test_size=0.20, random_state=100)

    # 3 types of vectorizer
    vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.75, ngram_range=(1, 2),analyzer='word',strip_accents="ascii")
    vectorizer_count = CountVectorizer(analyzer='word')
    vectorizer_hash = HashingVectorizer(ngram_range=(1, 2),strip_accents="ascii")

    # print(type(vectorizer_hash))
    from sklearn.pipeline import FeatureUnion
    # custom_vect = YourCustomVectorizer()
    combined_features = FeatureUnion([("hash", vectorizer_hash),
                                      ("count", vectorizer_count),
                                      ("tfidf",vectorizer_hash)])

    train_tfIdf = vectorizer_hash.fit_transform(X_train.values.astype('U'))
    test_tfIdf = vectorizer_hash.transform(X_test.values.astype('U'))

    # print("features names: ",vectorizer_tfidf.get_feature_names()[:10])

    print("train tf idf shape: ",train_tfIdf.shape)
    print("test tf idf shape: ",test_tfIdf.shape,"\n")

    from scipy.sparse import hstack
    train_tfIdf = hstack((train_tfIdf, np.array(X_train2)[:, None]))
    test_tfIdf = hstack((test_tfIdf, np.array(X_test2)[:, None]))

    print("train tf idf shape: ",train_tfIdf.shape)
    print("test tf idf shape: ",test_tfIdf.shape,"\n")

    ################################################################
    scoring = 'accuracy'
    seed = 7
    # 10 cross validation to check wich model to choose

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('RFC', RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=100)))

    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None)
        cv_results = model_selection.cross_val_score(model, train_tfIdf, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    ######################### KNN ###########################

    # print("KNN:")
    # knn_classifier = KNeighborsClassifier()
    # knn_classifier.fit(train_tfIdf, y_train)
    # pred2 = knn_classifier.predict(test_tfIdf)

    # # Calculate the accuracy score: score
    # accuracy_tfidf = metrics.accuracy_score(y_test, pred2)
    # print(accuracy_tfidf)
    # print(confusion_matrix(y_test, pred2))
    # print(classification_report(y_test, pred2))

    ######################### LR ###########################
    print("LR:")
    lr_classifier = LogisticRegression(solver='liblinear')
    lr_classifier.fit(train_tfIdf, y_train)
    pred3 = lr_classifier.predict(test_tfIdf)

    # Calculate the accuracy score: score
    accuracy_tfidf = metrics.accuracy_score(y_test, pred3)
    print(accuracy_tfidf)
    print(confusion_matrix(y_test, pred3))
    print(classification_report(y_test, pred3))

    ######################### RFC ###########################
    # print("Random Forest Classifier:")
    # classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=100)
    # classifier.fit(train_tfIdf, y_train)
    # predRF = classifier.predict(test_tfIdf)
    #
    # # Calculate the accuracy score
    # accuracy_RF = metrics.accuracy_score(y_test, predRF)
    # print(accuracy_RF)
    #
    # Conf_metrics_RF = metrics.confusion_matrix(y_test, predRF, labels=[1, 0])
    # print(Conf_metrics_RF)
    # print(confusion_matrix(y_test, predRF))
    # print(classification_report(y_test, predRF))


    #################### EXECUTE ###########################

    print("***test***")
    df2 = pd.read_csv('Test.csv',encoding='latin-1')
    df2['length'] = check_happy(df2['SentimentText'].values)
    temp = np.array(pre.getData(df2['SentimentText'].array))
    test_tfIdf2 = vectorizer_hash.transform(temp.astype('U'))

    # df2['length'] = df2['SentimentText'].str.len()
    test_tfIdf2 = hstack((test_tfIdf2, np.array(df2['length'])[:, None]))

    # test_x = temp
    # test_tfIdf2 = vectorizer_tfidf.transform(test_x.values.astype('U'))
    pred_test = lr_classifier.predict(test_tfIdf2)
    df2['SentimentText'] = pred_test
    del df2['length']
    df2.columns = ['ID','Sentiment']
    df2.to_csv("results15.csv",index=False)

    Conf_metrics_tfidf = metrics.confusion_matrix(y_test, pred3, labels=[1, 0])
    print(Conf_metrics_tfidf,"\n")


def check_happy(ndarray):

    lines = ['\\',",", '`', '*', '_', '{', '}', '[', ']','&','/','?','~',':','^',
                       '(', ')', '>','<', '#', '+', '-', '.', '!', '$', '\'']
    res_list=[]
    for tweet in ndarray:
        tweet_list=tweet.split()
        c = sum(el in lines for el in tweet_list)
        res_list.append(c)

    return np.array(res_list)


if __name__ == "__main__":
    test()
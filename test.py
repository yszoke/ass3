# import nltk
# nltk.download('stopwords')
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

import Preprocessing as pre

def test():
    df = pd.read_csv('Train.csv',encoding='latin-1')
    df.groupby('Sentiment').SentimentText.count().plot.bar(ylim=0)
    # plt.show()
################################################################

    df['processedtext'] = np.array(pre.getData(df['SentimentText'].array))
    df['length'] = df['SentimentText'].str.len()

    target = df['Sentiment']
################################################################
    X_train, X_test, y_train, y_test = train_test_split(df['processedtext'], target, test_size=0.20, random_state=100)


    # vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    # vectorizer_tfidf = CountVectorizer(analyzer='word')
    vectorizer_tfidf = HashingVectorizer(analyzer='word',ngram_range=(1, 2))


    train_tfIdf = vectorizer_tfidf.fit_transform(X_train.values.astype('U'))
    test_tfIdf = vectorizer_tfidf.transform(X_test.values.astype('U'))

    # print("features names: ",vectorizer_tfidf.get_feature_names()[:10])

    print("train tf idf shape: ",train_tfIdf.shape)
    print("test tf idf shape: ",test_tfIdf.shape,"\n")

    ################################################################
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

    scoring = 'accuracy'
    seed = 7
    # 10 cross validation to check wich model to choose

    models = []
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    models.append(('LR', LogisticRegression(solver='liblinear')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))

    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None)
        cv_results = model_selection.cross_val_score(model, train_tfIdf, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    ######################### NB ###########################

    # print("NB:")
    # nb_classifier = MultinomialNB()
    #
    # nb_classifier.fit(train_tfIdf, y_train)
    #
    # pred2 = nb_classifier.predict(test_tfIdf)
    # # print(pred2[:10])
    #
    # # Calculate the accuracy score: score
    # accuracy_tfidf = metrics.accuracy_score(y_test, pred2)
    # print(accuracy_tfidf)

    ######################### LR ###########################
    # print("LR:")
    lr_classifier = LogisticRegression(solver='liblinear')
    #
    # lr_classifier.fit(train_tfIdf, y_train)
    #
    #
    # pred3 = lr_classifier.predict(test_tfIdf)
    # # print(pred2[:10])
    #
    # # Calculate the accuracy score: score
    # accuracy_tfidf = metrics.accuracy_score(y_test, pred3)
    # print(accuracy_tfidf)

    #################### EXECUTE ###########################

    # print("***test***")
    # df2 = pd.read_csv('Test.csv',encoding='latin-1')
    # # temp = df2['SentimentText'].apply(
    # #     lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    # temp = np.array(pre.getData(df2['SentimentText'].array))
    # test_tfIdf2 = vectorizer_tfidf.transform(temp.astype('U'))
    # # test_x = temp
    # # test_tfIdf2 = vectorizer_tfidf.transform(test_x.values.astype('U'))
    # # pred_test=nb_classifier.predict(test_tfIdf2)
    # pred_test = lr_classifier.predict(test_tfIdf2)
    # df2['SentimentText'] = pred_test
    # df2.columns = ['ID','Sentiment']
    # df2.to_csv("results8.csv",index=False)



    # Conf_metrics_tfidf = metrics.confusion_matrix(y_test, pred2, labels=[1, 0])
    # print(Conf_metrics_tfidf,"\n")
    ######################### RFC ###########################
    # print("Random Forest Classifier:")
    # classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=100)
    #
    # classifier.fit(train_tfIdf, y_train)
    #
    # predRF = classifier.predict(test_tfIdf)
    # # print(predRF[:10])
    #
    # # Calculate the accuracy score
    # accuracy_RF = metrics.accuracy_score(y_test, predRF)
    # print(accuracy_RF)
    #
    # Conf_metrics_RF = metrics.confusion_matrix(y_test, predRF, labels=[1, 0])
    # print(Conf_metrics_RF)


if __name__ == "__main__":
    test()
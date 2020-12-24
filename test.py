# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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




def test():
    df = pd.read_csv('Train.csv',encoding='latin-1')
    df.groupby('Sentiment').SentimentText.count().plot.bar(ylim=0)
    plt.show()

    stemmer = PorterStemmer()
    words = stopwords.words("english")

    df['processedtext'] = df['SentimentText'].apply(
        lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    # print(df.shape)
    # df.head(10)

    target = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(df['processedtext'], target, test_size=0.30, random_state=100)

    print("shape: ",df.shape);
    print("train shape: ",X_train.shape);
    print("test shape: ",X_test.shape)

    vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    train_tfIdf = vectorizer_tfidf.fit_transform(X_train.values.astype('U'))

    test_tfIdf = vectorizer_tfidf.transform(X_test.values.astype('U'))

    print("features names: ",vectorizer_tfidf.get_feature_names()[:10])
    print("train tf idf shape: ",train_tfIdf.shape)
    print("test tf idf shape: ",test_tfIdf.shape,"\n")

    print("NB:")
    nb_classifier = MultinomialNB()

    nb_classifier.fit(train_tfIdf, y_train)

    pred2 = nb_classifier.predict(test_tfIdf)
    # print(pred2[:10])

    # Calculate the accuracy score: score
    accuracy_tfidf = metrics.accuracy_score(y_test, pred2)
    print(accuracy_tfidf)

    Conf_metrics_tfidf = metrics.confusion_matrix(y_test, pred2, labels=[1, 0])
    print(Conf_metrics_tfidf,"\n")

    print("Random Forest Classifier:")
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=100)

    classifier.fit(train_tfIdf, y_train)

    predRF = classifier.predict(test_tfIdf)
    # print(predRF[:10])

    # Calculate the accuracy score
    accuracy_RF = metrics.accuracy_score(y_test, predRF)
    print(accuracy_RF)

    Conf_metrics_RF = metrics.confusion_matrix(y_test, predRF, labels=[1, 0])
    print(Conf_metrics_RF)


if __name__ == "__main__":
    test()
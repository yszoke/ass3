import pandas
import matplotlib.pyplot as plt
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

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names1 = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
dataset = pandas.read_csv(url, names=names1)
# print(dataset.shape)
# print(dataset.head(20))
# print(dataset.describe())
# dataset.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
# plt.show()


# # histograms
# dataset.hist()
# plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=None)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))




clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_pred_test = clf.predict(X_validation)
y_pred_train = clf.predict(X_train)

# Prints train' accuracy
accuracy = metrics.accuracy_score(y_pred_train, Y_train)
print("Training accuracy : %s" % "{0:.3%}".format(accuracy))

# Model test' accuracy, (how often is the classifier correct)
print("DecisionTreeClassifier accuracy:", metrics.accuracy_score(Y_validation, y_pred_test) * 100, "%")

# Create a series with features' importance:
featimp = pandas.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print(featimp)



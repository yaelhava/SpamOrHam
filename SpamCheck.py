import pandas as pd
from sklearn import feature_extraction, model_selection, metrics, svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


def decisionTree(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier()
    model = dtc.fit(X_train, y_train)
    target_pred = model.predict(X_test)
    res = metrics.accuracy_score(y_test, target_pred)
    print("Decision tree accuracy:", res)


def knn(X_train, X_test, y_train, y_test):
    max = 0
    for i in range(3, 49):
        for j in range(1, 2):
            if i % 2 == 1:
                knc = KNeighborsClassifier(n_neighbors=i, p=j)
                model = knc.fit(X_train, y_train)
                target_pred = model.predict(X_test)
                res = metrics.accuracy_score(y_test, target_pred)
            if res > max:
                max = res
                numOfNeighbors = i
                p = j

    if p == 1:
        metric = "manhattan"
    else:
        metric = "euclidian"

    print("KNN accuracy:", max, " number of neighbors:", numOfNeighbors, "\n\t\t\t  using metric:", metric)

def svm(X_train, X_test, y_train, y_test):
    svcSigmoid = SVC(kernel='sigmoid')
    svcLinear = SVC(kernel='linear')
    svcPoly = SVC(kernel='poly')
    svcRBF = SVC(kernel='rbf')
    svc = [svcSigmoid, svcLinear, svcPoly, svcRBF]
    max = 0
    for type in svc:
        model = type.fit(X_train, y_train)
        target_pred = model.predict(X_test)
        res = metrics.accuracy_score(y_test, target_pred)
        if res > max:
            max = res
            maxModel = type.kernel

    print("SVM accuracy:", max, " in model:", maxModel)


def adaboost(X_train, X_test, y_train, y_test):
    abc = AdaBoostClassifier()
    model = abc.fit(X_train, y_train)
    target_pred = model.predict(X_test)
    print("Adaboost accuracy:", metrics.accuracy_score(y_test, target_pred))


def run(file):
    data = pd.read_csv(file, encoding='latin-1')

    f = feature_extraction.text.CountVectorizer(stop_words = 'english')

    X = f.fit_transform(data["message"])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['label'], test_size=0.33)

    adaboost(X_train, X_test, y_train, y_test)
    svm(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    decisionTree(X_train, X_test, y_train, y_test)


if __name__ == '__main__':

    smsSpamCollection = "SpamOrHam.csv"
    run(smsSpamCollection)



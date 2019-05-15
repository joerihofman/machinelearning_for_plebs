import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from nl.joerihofman.machinelearningforplebs.FileLoader import FileLoader


class Main:

    dataset = FileLoader.get_dataset_from_file("tijdvakMetAanlever.csv")

    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([5, 20, 14, 32, 22, 38])
    seed = 7
    scoring = "accuracy"

    models = [
        ("LR", LogisticRegression(solver="liblinear", multi_class="ovr")),
        ("KNN", KNeighborsClassifier()),
        ("CART", DecisionTreeClassifier()),
        ("NB", GaussianNB()),
        ("SVM", SVC(gamma="auto")),
        ("SGD", SGDClassifier(loss="hinge", penalty="l2", max_iter=3000, tol=0.001, shuffle=True))
    ]

    def __init__(self):
        self.model = LinearRegression().fit(self.x, self.y)

    def do_something_with_dataset(self):
        return self.dataset.iloc[:].values

    def get_score(self):
        return self.model.score(self.x, self.y)

    def get_intercept(self):
        return self.model.intercept_

    def get_slope(self):
        return self.model.coef_

    def get_prediction(self):
        return self.model.predict(self.x)

    def get_shape(self):
        return self.dataset.shape

    def overig(self):
        array = self.dataset.values
        x = array[:, 0:16]
        y = array[:, 1]
        validation_size = 0.20
        # return x_train, x_val, y_train, y_val =test_split(X, Y, test_size=validation_size, random_state=seed)
        return model_selection.train_test_split(x, y, test_size=validation_size, random_state=self.seed)

    def check_most_accurate_model(self):
        x_train, x_val, y_train, y_val = main.overig()

        print("%s: %f (%f)" % ("LNRG", self.get_score(), self.get_intercept()))

        results = []
        names = []

        for name, model in self.models:
            kfold = model_selection.KFold(n_splits=10, random_state=self.seed)
            cv_res = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring)
            results.append(cv_res)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_res.mean(), cv_res.std())
            print(msg)

        self.plot(results, names)

    @staticmethod
    def plot(results, names):
        figure = plt.figure()
        figure.suptitle("algorithm compare")

        ax = figure.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

        sys.exit(2)

    @staticmethod
    def predict():
        x_train, x_val, y_train, y_val = main.overig()
        cart = DecisionTreeClassifier()
        cart.fit(x_train, y_train)
        predictions = cart.predict(x_val)

        print(accuracy_score(y_val, predictions))
        print(confusion_matrix(y_val, predictions))
        print(classification_report(y_val, predictions))


main = Main()
main.check_most_accurate_model()

# print(main.do_something_with_dataset())
# print(x_train)
# print("------------------------------------------------------------------------------")
# print(x_val)


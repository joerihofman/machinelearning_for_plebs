import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
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
    scoring = "explained_variance"

    models = [
        ("LR", LogisticRegression(solver="liblinear", multi_class="ovr")),
        ("KNN", KNeighborsClassifier()),
        ("CART", DecisionTreeClassifier()),
        ("NB", GaussianNB()),
        ("SVM", SVC(gamma="auto")),
        ("Elastic", ElasticNet(alpha=1.0, random_state=0))
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

    def split_data_in_test_en_training(self):
        array = self.dataset.values
        x = array[:, 0:16]
        y = array[:, 1]
        validation_size = 0.20
        return model_selection.train_test_split(x, y, test_size=validation_size, random_state=self.seed)

    def check_most_accurate_model(self):
        x_train, x_val, y_train, y_val = main.split_data_in_test_en_training()

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

    @staticmethod
    def predict():
        x_train, x_val, y_train, y_val = main.split_data_in_test_en_training()
        cart = DecisionTreeClassifier()
        cart.fit(x_train, y_train)
        predictions = cart.predict(x_val)

        print(accuracy_score(y_val, predictions))
        print(confusion_matrix(y_val, predictions))
        print(classification_report(y_val, predictions))

    def probeer_iets(self):
        lasso = Lasso(alpha=0.1)
        x_train, x_val, y_train, y_val = main.split_data_in_test_en_training()
        elnet = ElasticNet(alpha=0.1, l1_ratio=0.7)
        y_pred_elnet = elnet.fit(x_train, y_train).predict(x_val)
        r2_score_elnet = r2_score(y_val, y_pred_elnet)
        print(elnet)
        print("r^2 on test data : %f" % r2_score_elnet)

        y_pred_lasso = lasso.fit(x_train, y_train).predict(x_val)
        r2_score_lasso = r2_score(y_val, y_pred_lasso)

        m, s, _ = plt.stem(np.where(elnet.coef_)[0], elnet.coef_[elnet.coef_ != 0],
                           markerfmt='x', label='Elastic net coefficients')
        plt.setp([m, s], color="#2ca02c")
        m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0],
                           markerfmt='x', label='Lasso coefficients')
        plt.setp([m, s], color='#ff7f0e')

        plt.legend(loc='best')
        plt.title("Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f"
                  % (r2_score_lasso, r2_score_elnet))
        plt.show()

main = Main()
main.check_most_accurate_model()


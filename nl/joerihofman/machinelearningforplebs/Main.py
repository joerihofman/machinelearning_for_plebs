import numpy as np
from sklearn.linear_model import LinearRegression

from nl.joerihofman.machinelearningforplebs.FileLoader import FileLoader


class Main:

    dataset = FileLoader.get_dataset_from_file("")

    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([5, 20, 14, 32, 22, 38])

    def __init__(self):
        self.model = LinearRegression().fit(self.x, self.y)

    def do_something_with_dataset(self):
        return self.dataset.iloc[:, :-1].values

    def get_score(self):
        return self.model.score(self.x, self.y)

    def get_intercept(self):
        return self.model.intercept_

    def get_slope(self):
        return self.model.coef_

    def get_prediction(self):
        return self.model.predict(self.x)


main = Main()

print(main.do_something_with_dataset())

print("INTERCEPT : " + main.get_intercept())
print("SLOPE : " + main.get_slope())
print("PREDICTION : " + main.get_prediction())

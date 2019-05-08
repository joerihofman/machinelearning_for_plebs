import numpy as np
from sklearn.linear_model import LinearRegression


class Main:

    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([5, 20, 14, 32, 22, 38])

    def __init__(self):
        self.model = LinearRegression().fit(self.x, self.y)

    def getScore(self):
        return self.model.score(self.x, self.y)

    def getIntercept(self):
        return self.model.intercept_

    def getSlope(self):
        return self.model.coef_

    def getPrediction(self):
        return self.model.predict(self.x)


main = Main()

print("INTERCEPT : " + main.getIntercept())
print("SLOPE : " + main.getSlope())
print("PREDICTION : " + main.getPrediction())

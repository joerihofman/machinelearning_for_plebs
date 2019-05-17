import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

from nl.joerihofman.machinelearningforplebs.FileLoader import FileLoader

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

seed = 7
dataset = FileLoader.get_dataset_from_file("tijdvakMetAanlever.csv")
array = dataset.values
x = array[:, 0:16]
y = array[:, 1]
validation_size = 0.20
x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_val, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_val, y_pred))

# Plot outputs
plt.scatter(x_val, y_val,  color='black')
plt.plot(x_val, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
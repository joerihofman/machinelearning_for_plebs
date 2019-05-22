import datetime

import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LinearRegression

from nl.joerihofman.machinelearningforplebs.FileLoader import FileLoader


def date_time_to_days_since_epoch(date):
    try:
        parsed_date = datetime.datetime.strptime(date, "%d-%m-%y %H:%M").date()
        return (parsed_date - datetime.date(1970, 1, 1)).days
    except ValueError:
        return date_to_days_since_epoch(date)


def date_to_days_since_epoch(date):
    try:
        parsed_date = datetime.datetime.strptime(date, "%d-%m-%y").date()
        return (parsed_date - datetime.date(1970, 1, 1)).days
    except TypeError:
        raise


def bereken_die_zooi(aanlever, tijdvak):
    try:
        aanl = date_time_to_days_since_epoch(aanlever)
        tijdv = date_to_days_since_epoch(tijdvak)
        return aanl - tijdv
    except TypeError:
        return 1000000000


# date_to_days_since_epoch(row[8]) - date_time_to_days_since_epoch(row[3])

class LinReg:
    seed = 7

    lm = LinearRegression()

    @staticmethod
    def maak_dataframe():
        dataset = FileLoader.get_dataset_from_file("data_export_zonder_header.csv")
        data_frame = pandas.DataFrame(dataset)
        data_frame["verschil"] = data_frame.apply(lambda row: bereken_die_zooi(row[3], row[8]), axis=1)

        data_frame = data_frame[data_frame.verschil != 1000000000]

        data_frame["tijdvakbegin_epoch"] = data_frame.apply(lambda row: date_to_days_since_epoch(row[8]), axis=1)
        data_frame["ontvangst_epoch"] = data_frame.apply(lambda row: date_time_to_days_since_epoch(row[3]), axis=1)
        return data_frame

    @staticmethod
    def print_dit(data_frame):
        for it in data_frame["verschil"]:
            print(it)

    def schatting_samenhang(self, data_frame):
        Y = data_frame[["tijdvakbegin_epoch", "ontvangst_epoch", "psn_id_fonds"]].copy()  # .drop("verschil", axis=1).drop("id_extern", axis=1)
        Y = Y.fillna(data_frame.mean())
        X = data_frame[["verschil"]].copy().fillna(data_frame["verschil"].mean())
        self.lm.fit(X, Y)
        return pandas.DataFrame(zip(Y.columns, self.lm.coef_), columns=["features", "estimatedcoef"])

    def print_scatterplot(self, data_frame):
        plt.scatter(data_frame.tijdvakbegin_epoch, data_frame.verschil)
        plt.ylabel("verschil in dagen")
        plt.xlabel("tijdvak begin")
        plt.title("titel")
        plt.show()


lr = LinReg()
dataframe = lr.maak_dataframe()
print(dataframe.shape)
print(lr.schatting_samenhang(dataframe))
lr.print_scatterplot(dataframe)











"""
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
"""
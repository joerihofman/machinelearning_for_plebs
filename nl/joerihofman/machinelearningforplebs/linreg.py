import datetime

import matplotlib.pyplot as plt
import pandas
import sklearn
from sklearn.linear_model import LinearRegression

from nl.joerihofman.machinelearningforplebs.FileLoader import FileLoader

START_EPOCH = datetime.date(1970, 1, 1)
SEED = 7
LM = LinearRegression()


def date_time_to_days_since_epoch(date):
    try:
        parsed_date = datetime.datetime.strptime(date, "%d-%m-%y %H:%M").date()
        return (parsed_date - START_EPOCH).days
    except ValueError:
        return date_to_days_since_epoch(date)


def date_to_days_since_epoch(date):
    try:
        parsed_date = datetime.datetime.strptime(date, "%d-%m-%y").date()
        return (parsed_date - START_EPOCH).days
    except TypeError:
        raise


def calculate_difference_in_days(aanlever, tijdvak):
    try:
        aanl_aantal_dagen = date_time_to_days_since_epoch(aanlever)
        tijdv_aantal_dagen = date_to_days_since_epoch(tijdvak)
        return aanl_aantal_dagen - tijdv_aantal_dagen
    except TypeError:
        return 100000000


def create_dataframe():
    dataset = FileLoader.get_dataset_from_file("data_export_zonder_header.csv")
    data_frame = pandas.DataFrame(dataset)
    data_frame["verschil"] = data_frame.apply(lambda row: calculate_difference_in_days(row[3], row[8]), axis=1)

    data_frame = data_frame[data_frame.verschil != 100000000]

    data_frame["tijdvakbegin_epoch"] = data_frame.apply(lambda row: date_to_days_since_epoch(row[8]), axis=1)
    data_frame["ontvangst_epoch"] = data_frame.apply(lambda row: date_time_to_days_since_epoch(row[3]), axis=1)
    return data_frame


def estimate_coefficient(data_frame):
    x = data_frame[["verschil"]].copy().fillna(data_frame["verschil"].mean())
    y = data_frame[["tijdvakbegin_epoch", "ontvangst_epoch", "psn_id_fonds"]].copy()
    y = y.fillna(data_frame.mean())
    make_relationship_plot(x, y, data_frame)
    # make_residual_plot(x, y, data_frame)
    return pandas.DataFrame(zip(y.columns, LM.coef_), columns=["features", "estimatedcoef"])


def make_relationship_plot(x, y, data_frame):
    LM.fit(x, y)
    print(x.shape)
    print(data_frame.tijdvakbegin_epoch.shape)
    plt.scatter(data_frame.tijdvakbegin_epoch, LM.predict(x)[:,0])
    plt.ylabel("Ontvangstdatum (dagen sinds epoch)")
    plt.xlabel("Begin tijdvak (dagen sinds epoch)")
    plt.title("Relatie tussen begin van het tijdvak en de ontvangstdatum")
    plt.show()

def make_residual_plot(x, y, data_frame):
    x_train, x_test, y_train, y_test = make_testdata(x, data_frame)
    LM.fit(x_train, y_train)
    plt.scatter(LM.predict(x_train), LM.predict(x_train) - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(LM.predict(x_test), LM.predict(x_test) - y_test, c='g', s=40)
    # plt.hlines(y=0, xmin=0, xmax=1400)
    plt.ylabel("Residuals")
    plt.xlabel("Begin tijdvak (dagen sinds epoch)")
    plt.title("Residual plot met training (blauw) en test (groen) data")
    plt.show()

def make_testdata(x, data_frame):
    return sklearn.model_selection.train_test_split(
        x, data_frame.tijdvakbegin_epoch, test_size=0.33, random_state=5
    )


def print_scatterplot(data_frame):
    plt.scatter(data_frame.tijdvakbegin_epoch, data_frame.verschil)
    plt.ylabel("Verschil aanlevermoment en tijdvak (dagen)")
    plt.xlabel("Begin tijdvak (dagen sinds epoch)")
    plt.title("Scatterplot onderzoek")
    plt.show()


dataframe = create_dataframe()
estimate_coefficient(dataframe)
# print(estimate_coefficient(dataframe))
# print_scatterplot(dataframe)











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
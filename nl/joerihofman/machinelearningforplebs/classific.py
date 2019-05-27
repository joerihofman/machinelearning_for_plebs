import pandas
import sklearn

from nl.joerihofman.machinelearningforplebs.FileLoader import FileLoader


def create_dataframe():
    dataset = FileLoader.get_dataset_from_fout_file("IO_Status_berichten_zonder_header.csv")
    return pandas.DataFrame(dataset)


def make_testdata(data_frame):
    return sklearn.model_selection.train_test_split(
        data_frame.OLD_STATUS, data_frame, test_size=0.33, random_state=5
    )


dataframe = create_dataframe()
x_train, x_test, y_train, y_test = make_testdata(dataframe)
print(y_train)
print(x_train)

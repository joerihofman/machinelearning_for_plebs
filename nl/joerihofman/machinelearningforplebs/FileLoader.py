import pandas


class FileLoader:

    @staticmethod
    def get_dataset_from_file(file_name):
        path = "resources/" + file_name
        return pandas.read_csv(path)

import pandas


class FileLoader:

    @staticmethod
    def get_dataset_from_file(file_name):
        return pandas.read_csv(file_name)

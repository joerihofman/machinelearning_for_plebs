import pandas


class FileLoader:

    @staticmethod
    def get_dataset_from_file(file_name):
        names = ["dagen sinds epoch", "dagen tussen moment van aanleveren en begin tijdvak"]
        return pandas.read_csv(file_name, names=names)

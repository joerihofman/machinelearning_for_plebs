import pandas


class FileLoader:

    @staticmethod
    def get_dataset_from_file(file_name):
        names = ["id", "id_extern", "aanmaakdatum", "ontvangstdatum", "gebruiker", "ps_id_werkgever", "lb_nummer",
                 "verloningsperiode", "tijdvak_begin", "tijdvak_eind", "status", "reden", "herkomst", "softwarepakket",
                 "reden_code", "bron_id", "psn_id_fonds", "relatie_nummer"]
        return pandas.read_csv(file_name, names=names)

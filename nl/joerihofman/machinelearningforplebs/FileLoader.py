import pandas


def getFile(file_name, names):
    return pandas.read_csv(file_name, names=names)


class FileLoader:

    @staticmethod
    def get_dataset_from_aanlever_file(file_name):
        names = ["id", "id_extern", "aanmaakdatum", "ontvangstdatum", "gebruiker", "ps_id_werkgever", "lb_nummer",
                 "verloningsperiode", "tijdvak_begin", "tijdvak_eind", "status", "reden", "herkomst", "softwarepakket",
                 "reden_code", "bron_id", "psn_id_fonds", "relatie_nummer"]
        return getFile(file_name, names)

    @staticmethod
    def get_dataset_from_fout_file(file_name):
        names = ["JN_USER", "JN_DATE_TIME", "JN_OPERATION", "OLD_ID", "OLD_ID_EXTERN", "OLD_AANMAAKDATUM",
                 "OLD_ONTVANGSTDATUM", "OLD_GEBRUIKER", "OLD_PSN_ID_WERKGEVER", "OLD_LB_NUMMER",
                 "OLD_VERLONINGSPERIODE", "OLD_TIJDVAK_BEGINDATUM", "OLD_TIJDVAK_EINDDATUM", "OLD_STATUS", "OLD_REDEN",
                 "OLD_HERKOMST", "OLD_SOFTWAREPAKKET", "NEW_ID", "NEW_ID_EXTERN", "NEW_AANMAAKDATUM",
                 "NEW_ONTVANGSTDATUM", "NEW_GEBRUIKER", "NEW_PSN_ID_WERKGEVER", "NEW_LB_NUMMER",
                 "NEW_VERLONINGSPERIODE", "NEW_TIJDVAK_BEGINDATUM", "NEW_TIJDVAK_EINDDATUM", "NEW_STATUS", "NEW_REDEN",
                 "NEW_SOFTWAREPAKKET", "NEW_HERKOMST", "OLD_REDEN_CODE", "NEW_REDEN_CODE", "OLD_BRON_ID", "NEW_BRON_ID",
                 "OLD_PSN_ID_FONDS", "NEW_PSN_ID_FONDS", "OLD_RELATIENUMMER", "NEW_RELATIENUMMER"]

        return getFile(file_name, names)



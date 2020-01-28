import os


# CHANGE to reflect your paths
UMLS_META_PATH = 'data/umls/2017AA/META/'
S2_CONTEXT_PATH = 'data/kb_contexts/'

class StandardFilePath(object):
    def __init__(self):
        return

    @property
    def ontoemma_root_dir(self):
        return os.getcwd()

    @property
    def ontoemma_data_dir(self):
        return 'data'

    @property
    def ontoemma_kb_dir(self):
        return os.path.join(
            self.ontoemma_data_dir, 'kbs'
        )

    @property
    def ontoemma_syn_dir(self):
        return os.path.join(
            self.ontoemma_data_dir, 'synonyms'
        )

    @property
    def ontoemma_training_dir(self):
        return os.path.join(
            self.ontoemma_data_dir, 'training'
        )

    @property
    def ontoemma_umls_subset_dir(self):
        return UMLS_META_PATH

    @property
    def ontoemma_umls_output_dir(self):
        return os.path.join(
            self.ontoemma_data_dir, 'umls_output'
        )

    @property
    def ontoemma_kb_context_dir(self):
        return S2_CONTEXT_PATH

    @property
    def ontoemma_model_dir(self):
        return 'models'

    @property
    def ontoemma_output_dir(self):
        return 'output'

    @property
    def ontoemma_missed_file(self):
        return os.path.join(
            self.ontoemma_output_dir, 'missed.tsv'
        )



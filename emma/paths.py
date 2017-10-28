import os

class StandardFilePath(object):
    ontoemma_root = 'ontoemma'
    ontoemma_kb_dir = 'kbs'
    ontoemma_missed_file = 'missed.tsv'
    ontoemma_umls_subset_root = '2017AA_OntoEmma/2017AA/META'
    ontoemma_umls_output_root = 'umls_output'
    ontoemma_training_root = 'training'
    ontoemma_model_root = 'models'
    ontoemma_output_root = 'output'
    ontoemma_missed_fname = 'missed.tsv'

    def __init__(self, base_dir):
        """Set self.base_dir.
        """
        self.base_dir = base_dir

    @property
    def ontoemma_root_dir(self):
        return os.path.join(self.base_dir, self.ontoemma_root)

    @property
    def ontoemma_umls_subset_dir(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_umls_subset_root
        )

    @property
    def ontoemma_umls_output_dir(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_umls_output_root
        )

    @property
    def ontoemma_kb_dir(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_umls_output_root, 'kbs'
        )

    @property
    def ontoemma_training_dir(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_training_root
        )

    @property
    def ontoemma_model_dir(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_model_root
        )

    @property
    def ontoemma_output_dir(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_output_root
        )

    @property
    def ontoemma_missed_file(self):
        return os.path.join(
            self.ontoemma_root_dir, self.ontoemma_output_root, self.ontoemma_missed_fname
        )

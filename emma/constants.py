import numpy as np

# Constants for use by OntoEmma

# Accepted model types
IMPLEMENTED_MODEL_TYPES = {"nn": "Neural Network",
                           "lr": "Logistic Regression",
                           "rf": "Random Forest"}

# Score threshold for positive alignment
MIN_SCORE_THRESHOLD = 0.10
MAX_SCORE_THRESHOLD = 0.85

# Minimum size for training data
MIN_TRAINING_SET_SIZE = 10

# K number of top candidates to keep from candidate selection module
KEEP_TOP_K_CANDIDATES = 50
ASSIGN_TOP_K_CANDIDATES = 10

# N-gram size for character n-grams
NGRAM_SIZE = 5

# IDF limit below which tokens are thrown out
IDF_LIMIT = np.log(20)

# Negative samples per positive
NUM_HARD_NEGATIVE_PER_POSITIVE = 2
NUM_EASY_NEGATIVE_PER_POSITIVE = 1

# training KB names
TRAINING_KBS = ['CPT', 'GO', 'HGNC', 'HPO', 'MSH', 'OMIM', 'RXNORM']

# development KB names
DEVELOPMENT_KBS = ['DRUGBANK', 'HL7V3.0', 'ICD10CM', 'LNC']

# training, development, and test split
TRAINING_PART = 0.6
DEVELOPMENT_PART = 0.2
TEST_PART = 0.2

# relation labels from UMLS
UMLS_SYNONYM_REL_LABELS = ['RL', 'RQ', 'RU', 'SY']
UMLS_PARENT_REL_LABELS = ['RB', 'PAR', 'is a', 'part of', 'subClassOf', 'is_a', 'part_of']
UMLS_CHILD_REL_LABELS = ['RN', 'CHD', 'has part', 'subClass', 'has_part', 'component']
UMLS_SIBLING_REL_LABELS = ['SIB', 'RO']

# symmetric relations
SYMMETRIC_RELATIONS = {'PAR': 'CHD',
                       'CHD': 'PAR',
                       'RN': 'RB',
                       'RB': 'RN',
                       'subClassOf': 'subClass',
                       'subClass': 'subClassOf',
                       'part_of': 'has_part',
                       'has_part': 'part_of'}

# number of steps for generating regional graph
NUM_STEPS_FOR_KB_REGION = 2

# global similarity iterations
GLOBAL_SIMILARITY_ITERATIONS = 0
import numpy as np

# Constants for use by OntoEmma

# Accepted model types
IMPLEMENTED_MODEL_TYPES = {"nn": "Neural Network",
                           "lr": "Logistic Regression",
                           "rf": "Random Forest"}

# Score threshold for positive alignment
MIN_SCORE_THRESHOLD = 0.20
MAX_SCORE_THRESHOLD = 0.95

# Minimum size for training data
MIN_TRAINING_SET_SIZE = 10

# K number of top candidates to keep from candidate selection module
KEEP_TOP_K_CANDIDATES = 100

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
UMLS_PARENT_REL_LABELS = ['RB', 'PAR', 'Is a', 'Part of', 'subClassOf', 'is_a', 'part_of']
UMLS_CHILD_REL_LABELS = ['RN', 'CHD', 'Has part', 'subClass', 'has_part']
UMLS_SIBLING_REL_LABELS = ['SIB', 'RO']

# number of steps for generating regional graph
NUM_STEPS_FOR_KB_REGION = 5

# global similarity iterations
GLOBAL_SIMILARITY_ITERATIONS = 0
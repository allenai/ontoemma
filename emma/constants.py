import numpy as np

# Constants for use by OntoEmma

# Accepted model types
IMPLEMENTED_MODEL_TYPES = {"nn": "Neural Network",
                           "lr": "Logistic Regression"}

# Score threshold for positive alignment
SCORE_THRESHOLD = 0.50

# Minimum size for training data
MIN_TRAINING_SET_SIZE = 10

# K number of top candidates to keep from candidate selection module
KEEP_TOP_K_CANDIDATES = 100

# N-gram size for character n-grams
NGRAM_SIZE = 5

# IDF limit below which tokens are thrown out
IDF_LIMIT = np.log(20)

# training KB names
TRAINING_KBS = ['CPT', 'GO', 'HGNC', 'HPO', 'MSH', 'OMIM', 'RXNORM']
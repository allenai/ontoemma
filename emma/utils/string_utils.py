import re
import numpy as np
from sklearn.metrics import pairwise_distances


CLEANER_RE = re.compile(r'[^a-zA-Z0-9 ]+')


def clean(kp):
    return CLEANER_RE.sub('', kp)


def canonicalize(kp):
    return clean(kp).lower()


def get_idf(c_size, freq):
    """
    Computes IDF score
    :param c_size: corpus size
    :param freq: frequency of occurrence
    :return:
    """
    return np.log((c_size / freq) + 1)


def get_character_n_grams(s, n):
    """
    Return padded character n-grams from string
    :param s: input string
    :param n: length of n-grams
    :return:
    """
    s_padded = '\0' * (n - 1) + s + '\0' * (n - 1)
    return zip(*[s_padded[i:] for i in range(n)])


def get_tfidf_distance(ind1, ind2, scores):
    """
    # Compute vector distance between two tfidf score vectors
    :param ind1:
    :param ind2:
    :param scores:
    :return:
    """
    if (ind1 == -1 and ind2 == -1):
        return 99.
    else:
        return pairwise_distances(scores[ind1], scores[ind2])[0][0]


def get_jaccard_similarity(token_set1, token_set2):
    """
    Return jaccard index between two token sets; if token sets are invalid, return -1
    :param token_set1:
    :param token_set2:
    :return:
    """
    if token_set1 and token_set2:
        return len(token_set1.intersection(token_set2)
                  ) / len(token_set1.union(token_set2))
    else:
        return -1.0


def get_longest_common_substring_length(s1, s2):
    """
    Return longest common substring found in s1 and s2
    :param s1:
    :param s2:
    :return:
    """
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest:x_longest]


def normalize_string(s):
    """
    Process string name; strip string, lowercase, and replace some characters
    :param s: string
    :return:
    """
    return s.strip().lower().replace('-', '').replace('_', ' ')


def tokenize_string(s, tokenizer, stop):
    """
    Process name string and return tokenized words minus stop words
    :param s: string
    :param tokenizer:
    :param stop: set of stop words
    :return:
    """
    return tuple([t for t in tokenizer.tokenize(s) if t not in stop])

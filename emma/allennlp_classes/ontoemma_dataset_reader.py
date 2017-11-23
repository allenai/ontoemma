from typing import Dict, List
import logging

from overrides import overrides
import json
import tqdm
import spacy

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.fields import Field, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from emma.allennlp_classes.boolean_field import BooleanField
from emma.allennlp_classes.float_field import FloatField

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.metrics.distance import edit_distance
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import emma.utils.string_utils as string_utils
import emma.constants as constants


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ontology_matcher")
class OntologyMatchingDatasetReader(DatasetReader):
    """
    Reads instances from a jsonlines file where each line is in the following format:
    {"match": X, "source": {kb_entity}, "target: {kb_entity}}
     X in [0, 1]
     kb_entity is a slightly modified KBEntity in json with fields:
        canonical_name
        aliases
        definition
        other_contexts
        relationships
    and converts it into a ``Dataset`` suitable for ontology matching.
    Parameters
    ----------
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self) -> None:
        self.PARENT_REL_LABELS = constants.UMLS_PARENT_REL_LABELS
        self.CHILD_REL_LABELS = constants.UMLS_CHILD_REL_LABELS

        self.STOP = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()

        self.nlp = spacy.load('en')

    @overrides
    def read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        # open data file and read lines
        with open(file_path, 'r') as ontm_file:
            logger.info("Reading ontology matching instances from jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(ontm_file):
                training_pair = json.loads(line)
                s_ent = training_pair['source_ent']
                t_ent = training_pair['target_ent']
                label = training_pair['label']

                # convert entry to instance and append to instances
                instances.append(self.text_to_instance(s_ent, t_ent, label))

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @staticmethod
    def _normalize_ent(ent):
        norm_ent = dict()
        norm_ent['canonical_name'] = string_utils.normalize_string(ent['canonical_name'])
        norm_ent['aliases'] = [string_utils.normalize_string(a) for a in ent['aliases']]
        norm_ent['definition'] = string_utils.normalize_string(ent['definition'])
        norm_ent['par_relations'] = set([string_utils.normalize_string(i) for i in ent['par_relations']])
        norm_ent['chd_relations'] = set([string_utils.normalize_string(i) for i in ent['chd_relations']])
        return norm_ent

    def _compute_tokens(self, ent):
        """
        Compute tokens from given entity
        :param ent:
        :return:
        """
        name_tokens = string_utils.tokenize_string(ent['canonical_name'], self.tokenizer, self.STOP)
        stemmed_tokens = tuple([self.stemmer.stem(w) for w in name_tokens])
        lemmatized_tokens = tuple([self.lemmatizer.lemmatize(w) for w in name_tokens])
        character_tokens = tuple(string_utils.get_character_n_grams(
            ent['canonical_name'], constants.NGRAM_SIZE
        ))

        alias_tokens = [string_utils.tokenize_string(a, self.tokenizer, self.STOP) for a in ent['aliases']]

        def_tokens = string_utils.tokenize_string(ent['definition'], self.tokenizer, self.STOP)

        return [
            name_tokens, stemmed_tokens, lemmatized_tokens, character_tokens, alias_tokens, def_tokens
        ]

    def _dependency_parse(self, name):
        """
        compute dependency parse of name and return root word, and all chunk root words
        :param name: name string
        :return:
        """
        doc = self.nlp(name)
        root_text = [(token.dep_, token.head.text) for token in doc]
        root = [t for d, t in root_text if d == 'ROOT'][0]
        root_words = set([t for d, t in root_text])
        return root, root_words

    def _get_features(self, s_ent, t_ent):
        """
        compute all LR model features
        :param s_ent:
        :param t_ent:
        :return:
        """
        s_name_tokens, s_stem_tokens, s_lemm_tokens, s_char_tokens, s_alias_tokens, s_def_tokens = self._compute_tokens(s_ent)
        t_name_tokens, t_stem_tokens, t_lemm_tokens, t_char_tokens, t_alias_tokens, t_def_tokens = self._compute_tokens(t_ent)

        has_same_canonical_name = (s_name_tokens == t_name_tokens)
        has_same_stemmed_name = (s_stem_tokens == t_stem_tokens)
        has_same_lemmatized_name = (s_lemm_tokens == t_lemm_tokens)
        has_same_char_tokens = (s_char_tokens == t_char_tokens)
        has_alias_in_common = (len(set(s_alias_tokens).intersection(set(t_alias_tokens))) > 0)

        # initialize similarity features
        name_token_jaccard_similarity = 1.0
        inverse_name_token_edit_distance = 1.0
        name_stem_jaccard_similarity = 1.0
        inverse_name_stem_edit_distance = 1.0
        name_lemm_jaccard_similarity = 1.0
        inverse_name_lemm_edit_distance = 1.0
        name_char_jaccard_similarity = 1.0
        inverse_name_char_edit_distance = 1.0

        # jaccard similarity and token edit distance
        max_changes = len(s_name_tokens) + len(t_name_tokens)
        max_char_changes = len(s_char_tokens) + len(t_char_tokens)

        if not has_same_canonical_name:
            name_token_jaccard_similarity = string_utils.get_jaccard_similarity(
                set(s_name_tokens), set(t_name_tokens)
            )
            inverse_name_token_edit_distance = 1.0 - edit_distance(
                s_name_tokens, t_name_tokens
            ) / max_changes


        if not has_same_stemmed_name:
            name_stem_jaccard_similarity = string_utils.get_jaccard_similarity(
                set(s_stem_tokens), set(t_stem_tokens)
            )
            inverse_name_stem_edit_distance = 1.0 - edit_distance(
                s_stem_tokens, t_stem_tokens
            ) / max_changes

        if not has_same_lemmatized_name:
            name_lemm_jaccard_similarity = string_utils.get_jaccard_similarity(
                set(s_lemm_tokens), set(t_lemm_tokens)
            )
            inverse_name_lemm_edit_distance = 1.0 - edit_distance(
                s_lemm_tokens, t_lemm_tokens
            ) / max_changes

        if not has_same_char_tokens:
            name_char_jaccard_similarity = string_utils.get_jaccard_similarity(
                set(s_char_tokens), set(t_char_tokens)
            )
            inverse_name_char_edit_distance = 1 - edit_distance(
                s_char_tokens, t_char_tokens
            ) / max_char_changes

        max_alias_token_jaccard = 0.0
        min_alias_edit_distance = 1.0
        best_s_alias = s_ent['aliases'][0]
        best_t_alias = t_ent['aliases'][0]

        if not has_alias_in_common:
            for s_ind, s_a_tokens in enumerate(s_alias_tokens):
                for t_ind, t_a_tokens in enumerate(t_alias_tokens):
                    if s_a_tokens and t_a_tokens:
                        j_ind = string_utils.get_jaccard_similarity(
                            set(s_a_tokens), set(t_a_tokens)
                        )
                        if j_ind > max_alias_token_jaccard:
                            max_alias_token_jaccard = j_ind
                            best_s_alias = s_ent['aliases'][s_ind]
                            best_t_alias = t_ent['aliases'][t_ind]
                        e_dist = edit_distance(s_a_tokens, t_a_tokens) / (
                            len(s_a_tokens) + len(t_a_tokens)
                        )
                        if e_dist < min_alias_edit_distance:
                            min_alias_edit_distance = e_dist

        # has any relationships
        has_parents = (len(s_ent['par_relations']) > 0 and len(t_ent['par_relations']) > 0)
        has_children = (len(s_ent['chd_relations']) > 0 and len(t_ent['chd_relations']) > 0)

        percent_parents_in_common = 0.0
        percent_children_in_common = 0.0

        # any relationships in common
        if has_parents:
            max_parents_in_common = (len(s_ent['par_relations']) + len(t_ent['par_relations'])) / 2
            percent_parents_in_common = len(
                s_ent['par_relations'].intersection(t_ent['par_relations'])
            ) / max_parents_in_common

        if has_children:
            max_children_in_common = (len(s_ent['chd_relations']) + len(t_ent['chd_relations'])) / 2
            percent_children_in_common = len(
                s_ent['chd_relations'].intersection(t_ent['chd_relations'])
            ) / max_children_in_common

        s_acronyms = [(i[0] for i in a) for a in s_alias_tokens]
        t_acronyms = [(i[0] for i in a) for a in t_alias_tokens]
        has_same_acronym = (len(set(s_acronyms).intersection(set(t_acronyms))) > 0)

        s_name_root, s_name_heads = self._dependency_parse(s_ent['canonical_name'])
        t_name_root, t_name_heads = self._dependency_parse(t_ent['canonical_name'])

        has_same_name_root_word = (s_name_root == t_name_root)
        has_same_name_chunk_heads = (s_name_heads == t_name_heads)
        name_chunk_heads_jaccard_similarity = string_utils.get_jaccard_similarity(
            s_name_heads, t_name_heads
        )

        s_alias_root, s_alias_heads = self._dependency_parse(best_s_alias)
        t_alias_root, t_alias_heads = self._dependency_parse(best_t_alias)

        has_same_alias_root_word = (s_alias_root == t_alias_root)
        has_same_alias_chunk_heads = (s_alias_heads == t_alias_heads)
        alias_chunk_heads_jaccard_similarity = string_utils.get_jaccard_similarity(
            s_alias_heads, t_alias_heads
        )

        def_jaccard_similarity = string_utils.get_jaccard_similarity(
            set(s_def_tokens), set(t_def_tokens)
        )

        # form feature vector
        feature_vec = [FloatField(float(has_same_canonical_name)),
                       FloatField(float(has_same_stemmed_name)),
                       FloatField(float(has_same_lemmatized_name)),
                       FloatField(float(has_same_char_tokens)),
                       FloatField(float(has_alias_in_common)),

                       FloatField(name_token_jaccard_similarity),
                       FloatField(inverse_name_token_edit_distance),
                       FloatField(name_stem_jaccard_similarity),
                       FloatField(inverse_name_stem_edit_distance),
                       FloatField(name_lemm_jaccard_similarity),

                       FloatField(inverse_name_lemm_edit_distance),
                       FloatField(name_char_jaccard_similarity),
                       FloatField(inverse_name_char_edit_distance),
                       FloatField(max_alias_token_jaccard),
                       FloatField(1.0 - min_alias_edit_distance),

                       FloatField(percent_parents_in_common),
                       FloatField(percent_children_in_common),
                       FloatField(float(has_same_acronym)),
                       FloatField(float(has_same_name_root_word)),
                       FloatField(float(has_same_name_chunk_heads)),

                       FloatField(name_chunk_heads_jaccard_similarity),
                       FloatField(float(has_same_alias_root_word)),
                       FloatField(float(has_same_alias_chunk_heads)),
                       FloatField(alias_chunk_heads_jaccard_similarity),
                       FloatField(def_jaccard_similarity)
                       ]

        return feature_vec

    @overrides
    def text_to_instance(self,  # type: ignore
                         s_ent: dict,
                         t_ent: dict,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        fields['features'] = ListField(self._get_features(self._normalize_ent(s_ent), self._normalize_ent(t_ent)))

        # add boolean label (0 = no match, 1 = match)
        fields['label'] = BooleanField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'OntologyMatchingDatasetReader':
        params.assert_empty(cls.__name__)
        return OntologyMatchingDatasetReader()
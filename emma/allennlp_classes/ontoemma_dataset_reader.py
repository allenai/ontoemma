from typing import Dict, List
import logging

from overrides import overrides
import json
import tqdm
import random

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

    def _compute_tokens(self, ent):
        """
        Compute tokens from given entity
        :param ent:
        :return:
        """
        name_string = string_utils.normalize_string(ent['canonical_name'])
        name_tokens = string_utils.tokenize_string(name_string, self.tokenizer, self.STOP)
        stemmed_tokens = tuple([self.stemmer.stem(w) for w in name_tokens])
        lemmatized_tokens = tuple([self.lemmatizer.lemmatize(w) for w in name_tokens])
        character_tokens = tuple(string_utils.get_character_n_grams(
            name_string, constants.NGRAM_SIZE
        ))

        alias_tokens = []

        for a in ent['aliases']:
            alias_tokens.append(string_utils.tokenize_string(
                string_utils.normalize_string(a), self.tokenizer, self.STOP))

        parent_names = ent['par_relations']
        child_names = ent['chd_relations']

        return [
            name_tokens, stemmed_tokens, lemmatized_tokens, character_tokens,
            alias_tokens,
            set(parent_names),
            set(child_names)
        ]

    @overrides
    def text_to_instance(self,  # type: ignore
                         s_ent: dict,
                         t_ent: dict,
                         label: str = None) -> Instance:

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        s_name_tokens, s_stemmed_tokens, s_lemmatized_tokens, s_char_tokens, \
        s_alias_tokens, s_parent_names, s_child_names = self._compute_tokens(s_ent)
        t_name_tokens, t_stemmed_tokens, t_lemmatized_tokens, t_char_tokens, \
        t_alias_tokens, t_parent_names, t_child_names = self._compute_tokens(t_ent)

        has_same_canonical_name = (s_name_tokens == t_name_tokens)
        has_same_stemmed_name = (s_stemmed_tokens == t_stemmed_tokens)
        has_same_lemmatized_name = (s_lemmatized_tokens == t_lemmatized_tokens)
        has_same_char_tokens = (s_char_tokens == t_char_tokens)
        has_alias_in_common = (len(set(s_alias_tokens).intersection(set(t_alias_tokens))) > 0)

        # boolean features
        fields['has_same_canonical_name'] = BooleanField(has_same_canonical_name)
        fields['has_same_stemmed_name'] = BooleanField(has_same_stemmed_name)
        fields['has_same_lemmatized_name'] = BooleanField(has_same_lemmatized_name)
        fields['has_same_char_tokens'] = BooleanField(has_same_char_tokens)
        fields['has_alias_in_common'] = BooleanField(has_alias_in_common)

        # jaccard similarity and token edit distance
        max_changes = len(s_name_tokens) + len(t_name_tokens)
        max_char_changes = len(s_char_tokens) + len(t_char_tokens)

        if fields['has_same_canonical_name']:
            fields['name_token_jaccard'] = FloatField(1.0)
            fields['inverse_name_edit_distance'] = FloatField(1.0)
        else:
            fields['name_token_jaccard'] = FloatField(string_utils.get_jaccard_similarity(
                set(s_name_tokens), set(t_name_tokens)
            ))
            fields['inverse_name_edit_distance'] = FloatField(1.0 - edit_distance(
                s_name_tokens, t_name_tokens
            ) / max_changes)

        if fields['has_same_stemmed_name']:
            fields['stemmed_token_jaccard'] = FloatField(1.0)
            fields['inverse_stemmed_edit_distance'] = FloatField(1.0)
        else:
            fields['stemmed_token_jaccard'] = FloatField(string_utils.get_jaccard_similarity(
                set(s_stemmed_tokens), set(t_stemmed_tokens)
            ))
            fields['inverse_stemmed_edit_distance'] = FloatField(1.0 - edit_distance(
                s_stemmed_tokens, t_stemmed_tokens
            ) / max_changes)

        if fields['has_same_lemmatized_name']:
            fields['lemmatized_token_jaccard'] = FloatField(1.0)
            fields['inverse_lemmatized_edit_distance'] = FloatField(1.0)
        else:
            fields['lemmatized_token_jaccard'] = FloatField(string_utils.get_jaccard_similarity(
                set(s_lemmatized_tokens), set(t_lemmatized_tokens)
            ))
            fields['inverse_lemmatized_edit_distance'] = FloatField(1.0 - edit_distance(
                s_lemmatized_tokens, t_lemmatized_tokens
            ) / max_changes)

        if fields['has_same_char_tokens']:
            fields['char_token_jaccard'] = FloatField(1.0)
            fields['inverse_char_token_edit_distance'] = FloatField(1.0)
        else:
            fields['char_token_jaccard'] = FloatField(string_utils.get_jaccard_similarity(
                set(s_char_tokens), set(t_char_tokens)
            ))
            fields['inverse_char_token_edit_distance'] = FloatField(1 - edit_distance(
                s_char_tokens, t_char_tokens
            ) / max_char_changes)

        max_alias_token_jaccard = 0.0
        min_alias_edit_distance = 1.0

        if not fields['has_alias_in_common']:
            for s_a_tokens in s_alias_tokens:
                for t_a_tokens in t_alias_tokens:
                    if s_a_tokens and t_a_tokens:
                        j_ind = string_utils.get_jaccard_similarity(
                            set(s_a_tokens), set(t_a_tokens)
                        )
                        if j_ind > max_alias_token_jaccard:
                            max_alias_token_jaccard = j_ind
                        e_dist = edit_distance(s_a_tokens, t_a_tokens) / (
                            len(s_a_tokens) + len(t_a_tokens)
                        )
                        if e_dist < min_alias_edit_distance:
                            min_alias_edit_distance = e_dist

        fields['max_alias_token_jaccard'] = FloatField(max_alias_token_jaccard)
        fields['inverse_min_alias_edit_distance'] = FloatField(1.0 - min_alias_edit_distance)

        # has any relationships
        has_parents = (len(s_parent_names) > 0 and len(t_parent_names) > 0)
        has_children = (len(s_child_names) > 0 and len(t_child_names) > 0)

        percent_parents_in_common = 0.0
        percent_children_in_common = 0.0

        # any relationships in common
        if has_parents:
            max_parents_in_common = (len(s_parent_names) + len(t_parent_names)) / 2
            percent_parents_in_common = len(
                s_parent_names.intersection(t_parent_names)
            ) / max_parents_in_common

        if has_children:
            max_children_in_common = (len(s_child_names) + len(t_child_names)) / 2
            percent_children_in_common = len(
                s_child_names.intersection(t_child_names)
            ) / max_children_in_common

        fields['percent_parents_in_common'] = FloatField(percent_parents_in_common)
        fields['percent_children_in_common'] = FloatField(percent_children_in_common)

        # add boolean label (0 = no match, 1 = match)
        fields['label'] = BooleanField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'OntologyMatchingDatasetReader':
        params.assert_empty(cls.__name__)
        return OntologyMatchingDatasetReader()
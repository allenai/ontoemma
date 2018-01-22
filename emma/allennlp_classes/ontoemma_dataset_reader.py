from typing import Dict, List
import logging
import random
import itertools

from overrides import overrides
import json
import tqdm

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

from emma.EngineeredFeatureGenerator import EngineeredFeatureGenerator
import emma.utils.string_utils as string_utils


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
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 name_token_indexer: Dict[str, TokenIndexer] = None,
                 token_only_indexer: Dict[str, TokenIndexer] = None) -> None:
        self._name_token_indexer = name_token_indexer or \
                                   {'tokens': SingleIdTokenIndexer(namespace="tokens"),
                                    'token_characters': TokenCharactersIndexer(namespace="token_characters")}
        self._token_only_indexer = token_only_indexer or \
                                   {'tokens': SingleIdTokenIndexer(namespace="tokens")}
        self._tokenizer = tokenizer or WordTokenizer()

        self._empty_token_text_field = TextField(self._tokenizer.tokenize('00000'), self._token_only_indexer)

        self.feat_gen = EngineeredFeatureGenerator()

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

    def _get_features(self, s_ent: dict, t_ent: dict):
        """
        Calculate features between two entities s_ent and t_ent from source and target KBs respectively
        :param s_ent: entity from source KB
        :param t_ent: entity from target KB
        :return:
        """
        # get feature dictionary from feature generator
        feat_dict = self.feat_gen.calculate_features(s_ent, t_ent)

        # form feature vector
        feature_vec = [FloatField(float(feat_dict['has_same_canonical_name'])),
                       FloatField(float(feat_dict['has_same_canonical_name_tokens'])),
                       FloatField(float(feat_dict['has_same_canonical_name_token_set'])),
                       FloatField(float(feat_dict['has_same_stemmed_name_tokens'])),
                       FloatField(float(feat_dict['has_same_stemmed_name_token_set'])),

                       FloatField(float(feat_dict['has_same_lemmatized_name_tokens'])),
                       FloatField(float(feat_dict['has_same_lemmatized_name_token_set'])),
                       FloatField(feat_dict['name_char_4gram_jaccard']),
                       FloatField(feat_dict['name_char_5gram_jaccard']),
                       FloatField(float(feat_dict['has_alias_in_common'])),

                       FloatField(float(feat_dict['has_alias_tokens_in_common'])),
                       FloatField(float(feat_dict['has_alias_token_set_in_common'])),
                       FloatField(feat_dict['alias_token_jaccard']),
                       FloatField(feat_dict['max_alias_token_jaccard']),
                       FloatField(feat_dict['max_alias_4gram_jaccard']),

                       FloatField(feat_dict['max_alias_5gram_jaccard']),
                       FloatField(float(feat_dict['has_same_acronym'])),
                       FloatField(feat_dict['definition_token_jaccard']),
                       FloatField(float(feat_dict['has_same_wiki_entity'])),
                       FloatField(feat_dict['wiki_entity_jaccard']),

                       FloatField(feat_dict['max_wiki_entity_jaccard']),
                       FloatField(float(feat_dict['has_same_mesh_synonym'])),
                       FloatField(feat_dict['mesh_synonym_jaccard']),
                       FloatField(feat_dict['max_mesh_synonym_jaccard']),
                       FloatField(float(feat_dict['has_same_dbpedia_synonym'])),

                       FloatField(feat_dict['dbpedia_synonym_jaccard']),
                       FloatField(feat_dict['max_dbpedia_synonym_jaccard']),
                       FloatField(float(feat_dict['has_overlapping_synonym'])),
                       FloatField(feat_dict['all_synonym_jaccard']),
                       FloatField(feat_dict['max_all_synonym_jaccard']),

                       FloatField(float(feat_dict['has_same_root_word'])),
                       FloatField(feat_dict['root_word_jaccard'])
                       ]
        return feature_vec

    @overrides
    def text_to_instance(self,  # type: ignore
                         s_ent: dict,
                         t_ent: dict,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ

        # sample n from list l, keeping only entries with len less than max_len
        # if n is greater than the length of l, just return l
        def sample_n(l, n, max_len):
            l = [i for i in l if len(i) <= max_len]
            if not l:
                return ['00000']
            if len(l) <= n:
                return l
            return random.sample(l, n)

        fields: Dict[str, Field] = {}

        fields['engineered_features'] = ListField(
            self._get_features(s_ent, t_ent)
        )

        # add entity name fields
        fields['s_ent_name'] = TextField(
            self._tokenizer.tokenize('00000 ' + s_ent['canonical_name']), self._name_token_indexer
        )
        fields['t_ent_name'] = TextField(
            self._tokenizer.tokenize('00000 ' + t_ent['canonical_name']), self._name_token_indexer
        )

        s_aliases = sample_n(s_ent['aliases'], 16, 128)
        t_aliases = sample_n(t_ent['aliases'], 16, 128)

        # add entity alias fields
        fields['s_ent_alias'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 ' + a), self._name_token_indexer)
             for a in s_aliases]
        )
        fields['t_ent_alias'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 ' + a), self._name_token_indexer)
             for a in t_aliases]
        )

        # add entity definition fields
        fields['s_ent_def'] = TextField(
            self._tokenizer.tokenize(s_ent['definition']), self._token_only_indexer
        ) if len(s_ent['definition']) > 5 else self._empty_token_text_field
        fields['t_ent_def'] = TextField(
            self._tokenizer.tokenize(t_ent['definition']), self._token_only_indexer
        ) if len(t_ent['definition']) > 5 else self._empty_token_text_field

        # add boolean label (0 = no match, 1 = match)
        fields['label'] = BooleanField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'OntologyMatchingDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        name_token_indexer = TokenIndexer.dict_from_params(params.pop('name_token_indexer', {}))
        token_only_indexer = TokenIndexer.dict_from_params(params.pop('token_only_indexer', {}))
        params.assert_empty(cls.__name__)
        return OntologyMatchingDatasetReader(tokenizer=tokenizer,
                                             name_token_indexer=name_token_indexer,
                                             token_only_indexer=token_only_indexer)

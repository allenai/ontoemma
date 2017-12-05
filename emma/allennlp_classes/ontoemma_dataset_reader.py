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

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

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

        self.STOP = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en')
        self.token_dict = dict()

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

    def _tokenize(self, s):
        return string_utils.tokenize_string(s, self.tokenizer, self.STOP)

    def _tokenize_list(self, l):
        return [self._tokenize(i) for i in l]

    @staticmethod
    def _order_sublists(l):
        return [tuple(sorted(i)) for i in l]

    @staticmethod
    def _char_tokenize(s, ngram_size):
        return string_utils.get_character_n_grams(s, ngram_size)

    def _char_tokenize_list(self, l, ngram_size):
        return [self._char_tokenize(i, ngram_size) for i in l]

    def _stem_tokens(self, t):
        return [self.stemmer.stem(i) for i in t]

    def _lemmatize_tokens(self, t):
        return [self.lemmatizer.lemmatize(i) for i in t]

    def _stem_list(self, l):
        return [self._stem_tokens(t) for t in l]

    def _lemmatize_list(self, l):
        return [self._lemmatize_tokens(t) for t in l]

    @staticmethod
    def _acronym(t):
        return ''.join([i[0] for i in t])

    def _acronym_list(self, l):
        return [self._acronym(t) for t in l]

    @staticmethod
    def _jaccard(a, b):
        return string_utils.get_jaccard_similarity(set(a), set(b))

    def _max_jaccard(self, alist, blist):
        max_jacc = 0.0
        for a, b in itertools.product(alist, blist):
            jacc = self._jaccard(a, b)
            if jacc == 1.0:
                return 1.0
            if jacc > max_jacc:
                max_jacc = jacc
        return max_jacc

    @staticmethod
    def _overlaps(a, b):
        return not set(a).isdisjoint(b)

    def _form_dict_entry(self, ent):
        dict_entry = dict()
        dict_entry['name_tokens'] = self._tokenize(ent['canonical_name'])
        dict_entry['stemmed_name_tokens'] = self._stem_tokens(dict_entry['name_tokens'])
        dict_entry['lemmatized_name_tokens'] = self._lemmatize_tokens(dict_entry['name_tokens'])
        dict_entry['name_char_4grams'] = self._char_tokenize(ent['canonical_name'], 4)
        dict_entry['name_char_5grams'] = self._char_tokenize(ent['canonical_name'], 5)
        dict_entry['alias_tokens'] = self._tokenize_list(ent['aliases'])
        dict_entry['alias_char_4grams'] = self._char_tokenize_list(ent['aliases'], 4)
        dict_entry['alias_char_5grams'] = self._char_tokenize_list(ent['aliases'], 5)
        dict_entry['acronyms'] = self._acronym_list(dict_entry['alias_tokens'])
        dict_entry['alias_token_set'] = self._order_sublists(dict_entry['alias_tokens'])
        dict_entry['def_tokens'] = self._tokenize(ent['definition'])
        dict_entry['wiki_ent_tokens'] = self._tokenize_list(ent['wiki_entities'])
        dict_entry['mesh_syn_tokens'] = self._tokenize_list(ent['mesh_synonyms'])
        dict_entry['dbpedia_syn_tokens'] = self._tokenize_list(ent['dbpedia_synonyms'])
        dict_entry['parse_root'] = self._dependency_parse(ent['canonical_name'])
        return dict_entry

    def _get_dict_entry(self, ent):
        if ent['research_entity_id'] not in self.token_dict:
            self.token_dict[ent['research_entity_id']] = self._form_dict_entry(ent)
        return self.token_dict[ent['research_entity_id']]

    def _get_features(self, s_ent: dict, t_ent: dict):
        """
        Calculate features between two entities s_ent and t_ent from source and target KBs respectively
        :param s_ent: entity from source KB
        :param t_ent: entity from target KB
        :return:
        """

        # FOR TRAINING
        s_ent['mesh_synonyms'] = s_ent['mesh_synonynms']
        t_ent['mesh_synonyms'] = t_ent['mesh_synonynms']

        s_info = self._get_dict_entry(s_ent)
        t_info = self._get_dict_entry(t_ent)

        has_same_canonical_name = (s_ent['canonical_name'] == t_ent['canonical_name'])
        has_same_canonical_name_tokens = (s_info['name_tokens'] == t_info['name_tokens'])
        has_same_canonical_name_token_set = (set(s_info['name_tokens']) == set(t_info['name_tokens']))
        has_same_stemmed_name_tokens = (s_info['stemmed_name_tokens'] == t_info['stemmed_name_tokens'])
        has_same_stemmed_name_token_set = (set(s_info['stemmed_name_tokens']) == set(t_info['stemmed_name_tokens']))
        has_same_lemmatized_name_tokens = (s_info['lemmatized_name_tokens'] == t_info['lemmatized_name_tokens'])
        has_same_lemmatized_name_token_set = (set(s_info['lemmatized_name_tokens']) == set(t_info['lemmatized_name_tokens']))

        name_char_4gram_jaccard = self._jaccard(s_info['name_char_4grams'], t_info['name_char_4grams'])
        name_char_5gram_jaccard = self._jaccard(s_info['name_char_5grams'], t_info['name_char_5grams'])

        has_alias_in_common = self._overlaps(s_ent['aliases'], t_ent['aliases'])
        has_alias_tokens_in_common = self._overlaps(s_info['alias_tokens'], t_info['alias_tokens'])
        has_alias_token_set_in_common = self._overlaps(s_info['alias_token_set'], t_info['alias_token_set'])

        alias_token_jaccard = self._jaccard(s_info['alias_token_set'], t_info['alias_token_set'])
        max_alias_token_jaccard = self._max_jaccard(s_info['alias_token_set'], t_info['alias_token_set'])
        max_alias_4gram_jaccard = self._max_jaccard(s_info['alias_char_4grams'], t_info['alias_char_4grams'])
        max_alias_5gram_jaccard = self._max_jaccard(s_info['alias_char_5grams'], t_info['alias_char_5grams'])

        has_same_acronym = self._overlaps(s_info['acronyms'], t_info['acronyms']) or \
                           self._overlaps(s_info['acronyms'], t_ent['aliases']) or \
                           self._overlaps(s_ent['aliases'], t_info['acronyms'])

        definition_token_jaccard = self._jaccard(s_info['def_tokens'], t_info['def_tokens'])

        has_same_wiki_entity = self._overlaps(s_ent['wiki_entities'], t_ent['wiki_entities'])
        wiki_entity_jaccard = self._jaccard(s_ent['wiki_entities'], t_ent['wiki_entities'])
        max_wiki_entity_jaccard = self._max_jaccard(s_info['wiki_ent_tokens'], t_info['wiki_ent_tokens'])

        has_same_mesh_synonym = self._overlaps(s_ent['mesh_synonyms'], t_ent['mesh_synonyms'])
        mesh_synonym_jaccard = self._jaccard(s_ent['mesh_synonyms'], t_ent['mesh_synonyms'])
        max_mesh_synonym_jaccard = self._max_jaccard(s_info['mesh_syn_tokens'], t_info['mesh_syn_tokens'])

        has_same_dbpedia_synonym = self._overlaps(s_ent['dbpedia_synonyms'], t_ent['dbpedia_synonyms'])
        dbpedia_synonym_jaccard = self._jaccard(s_ent['dbpedia_synonyms'], t_ent['dbpedia_synonyms'])
        max_dbpedia_synonym_jaccard = self._max_jaccard(s_info['dbpedia_syn_tokens'], t_info['dbpedia_syn_tokens'])

        s_all = s_ent['aliases'] + s_ent['wiki_entities'] + s_ent['mesh_synonyms'] + s_ent['dbpedia_synonyms']
        t_all = t_ent['aliases'] + t_ent['wiki_entities'] + t_ent['mesh_synonyms'] + t_ent['dbpedia_synonyms']

        s_all_tokens = s_info['alias_tokens'] + s_info['wiki_ent_tokens'] + \
                       s_info['mesh_syn_tokens'] + s_info['dbpedia_syn_tokens']
        t_all_tokens = t_info['alias_tokens'] + t_info['wiki_ent_tokens'] + \
                       t_info['mesh_syn_tokens'] + t_info['dbpedia_syn_tokens']

        has_overlapping_synonym = self._overlaps(s_all, t_all)
        all_synonym_jaccard = self._jaccard(s_all, t_all)
        max_all_synonym_jaccard = self._max_jaccard(s_all_tokens, t_all_tokens)

        has_same_root_word = (s_info['parse_root'][0] == t_info['parse_root'][0])
        root_word_jaccard = self._jaccard(s_info['parse_root'][1], t_info['parse_root'][1])

        # form feature vector
        feature_vec = [FloatField(float(has_same_canonical_name)),
                       FloatField(float(has_same_canonical_name_tokens)),
                       FloatField(float(has_same_canonical_name_token_set)),
                       FloatField(float(has_same_stemmed_name_tokens)),
                       FloatField(float(has_same_stemmed_name_token_set)),

                       FloatField(float(has_same_lemmatized_name_tokens)),
                       FloatField(float(has_same_lemmatized_name_token_set)),
                       FloatField(name_char_4gram_jaccard),
                       FloatField(name_char_5gram_jaccard),
                       FloatField(float(has_alias_in_common)),

                       FloatField(float(has_alias_tokens_in_common)),
                       FloatField(float(has_alias_token_set_in_common)),
                       FloatField(alias_token_jaccard),
                       FloatField(max_alias_token_jaccard),
                       FloatField(max_alias_4gram_jaccard),

                       FloatField(max_alias_5gram_jaccard),
                       FloatField(float(has_same_acronym)),
                       FloatField(definition_token_jaccard),
                       FloatField(float(has_same_wiki_entity)),
                       FloatField(wiki_entity_jaccard),

                       FloatField(max_wiki_entity_jaccard),
                       FloatField(float(has_same_mesh_synonym)),
                       FloatField(mesh_synonym_jaccard),
                       FloatField(max_mesh_synonym_jaccard),
                       FloatField(float(has_same_dbpedia_synonym)),

                       FloatField(dbpedia_synonym_jaccard),
                       FloatField(max_dbpedia_synonym_jaccard),
                       FloatField(float(has_overlapping_synonym)),
                       FloatField(all_synonym_jaccard),
                       FloatField(max_all_synonym_jaccard),

                       FloatField(float(has_same_root_word)),
                       FloatField(root_word_jaccard)
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
            self._get_features(self._form_dict_entry(s_ent), self._form_dict_entry(t_ent))
        )

        # add entity name fields
        fields['s_ent_name'] = TextField(
            self._tokenizer.tokenize('00000 ' + self.s_ent['canonical_name']), self._name_token_indexer
        )
        fields['t_ent_name'] = TextField(
            self._tokenizer.tokenize('00000 ' + self.t_ent['canonical_name']), self._name_token_indexer
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
            self._tokenizer.tokenize(self.s_ent['definition']), self._token_only_indexer
        ) if len(s_ent['definition']) > 5 else self._empty_token_text_field
        fields['t_ent_def'] = TextField(
            self._tokenizer.tokenize(self.t_ent['definition']), self._token_only_indexer
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

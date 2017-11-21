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
from allennlp.data.fields import Field, TextField, ListField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.instance import Instance
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from emma.allennlp_classes.boolean_field import BooleanField


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
                 name_token_indexers: Dict[str, TokenIndexer] = None,
                 token_only_indexer: Dict[str, TokenIndexer] = None) -> None:
        self._name_token_indexers = name_token_indexers or \
                               {'tokens': SingleIdTokenIndexer(namespace="tokens"),
                                'token_characters': TokenCharactersIndexer(namespace="token_characters")}
        self._token_only_indexer = token_only_indexer or \
                               {'tokens': SingleIdTokenIndexer(namespace="tokens")}
        self._tokenizer = tokenizer or WordTokenizer()
        self._empty_token_text_field = TextField(self._tokenizer.tokenize('00000'), self._token_only_indexer)
        self._empty_list_token_text_field = ListField(
            [TextField(self._tokenizer.tokenize('00000'), self._token_only_indexer)]
        )

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

    @overrides
    def text_to_instance(self,  # type: ignore
                         s_ent: dict,
                         t_ent: dict,
                         label: str = None) -> Instance:

        # sample n from list l, keeping only entries with len less than max_len
        # if n is greater than the length of l, just return l
        def sample_n(l, n, max_len):
            l = [i for i in l if len(i) <= max_len]
            if not l:
                return ['00000']
            if len(l) <= n:
                return l
            return random.sample(l, n)

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # tokenize names
        s_name_tokens = self._tokenizer.tokenize('00000 00000 00000 00000 ' + s_ent['canonical_name'])
        t_name_tokens = self._tokenizer.tokenize('00000 00000 00000 00000 ' + t_ent['canonical_name'])

        # add entity name fields
        fields['s_ent_name'] = TextField(s_name_tokens, self._name_token_indexers)
        fields['t_ent_name'] = TextField(t_name_tokens, self._name_token_indexers)

        s_aliases = sample_n(s_ent['aliases'], 16, 128)
        t_aliases = sample_n(t_ent['aliases'], 16, 128)

        # add entity alias fields
        fields['s_ent_aliases'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 00000 00000 00000 ' + a), self._name_token_indexers)
             for a in s_aliases]
        )
        fields['t_ent_aliases'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 00000 00000 00000 ' + a), self._name_token_indexers)
             for a in t_aliases]
        )

        # add entity definition fields
        fields['s_ent_def'] = TextField(
            self._tokenizer.tokenize(s_ent['definition']), self._token_only_indexer
        ) if s_ent['definition'] else self._empty_token_text_field
        fields['t_ent_def'] = TextField(
            self._tokenizer.tokenize(t_ent['definition']), self._token_only_indexer
        ) if t_ent['definition'] else self._empty_token_text_field

        # add entity context fields
        s_contexts = sample_n(s_ent['other_contexts'], 16, 256)
        t_contexts = sample_n(t_ent['other_contexts'], 16, 256)

        fields['s_ent_context'] = ListField(
            [TextField(self._tokenizer.tokenize(c), self._token_only_indexer)
             for c in s_contexts]
        )
        fields['t_ent_context'] = ListField(
            [TextField(self._tokenizer.tokenize(c), self._token_only_indexer)
             for c in t_contexts]
        )

        # add entity parent fields
        s_parents = sample_n(s_ent['par_relations'], 8, 128)
        t_parents = sample_n(t_ent['par_relations'], 8, 128)

        fields['s_ent_parents'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 00000 00000 00000 ' + i), self._name_token_indexers)
             for i in s_parents]
        )
        fields['t_ent_parents'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 00000 00000 00000 ' + i), self._name_token_indexers)
             for i in t_parents]
        )

        # add entity children fields
        s_children = sample_n(s_ent['chd_relations'], 8, 128)
        t_children = sample_n(t_ent['chd_relations'], 8, 128)

        fields['s_ent_children'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 00000 00000 00000 ' + i), self._name_token_indexers)
             for i in s_children]
        )
        fields['t_ent_children'] = ListField(
            [TextField(self._tokenizer.tokenize('00000 00000 00000 00000 ' + i), self._name_token_indexers)
             for i in t_children]
        )

        # add boolean label (0 = no match, 1 = match)
        fields['label'] = BooleanField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'OntologyMatchingDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        name_token_indexers = TokenIndexer.dict_from_params(params.pop('name_token_indexers', {}))
        token_only_indexer = TokenIndexer.dict_from_params(params.pop('token_only_indexer', {}))
        params.assert_empty(cls.__name__)
        return OntologyMatchingDatasetReader(tokenizer=tokenizer,
                                             name_token_indexers=name_token_indexers,
                                             token_only_indexer=token_only_indexer)
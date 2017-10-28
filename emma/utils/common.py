import os
import re
from abc import abstractmethod, ABC
import logging

import copy
import tqdm
from emma.utils import file_util
from sklearn.feature_extraction.text import CountVectorizer
import spacy

SPLIT_RE = re.compile('[^a-zA-Z0-9]+')
MAX_NGRAM_LEN = 5
WORD_EMBEDDING_THRESHOLD = 0.6
BILOU_FIELD_NAME = 'biluo_entities'
BILOU_FIELD_NAME_MENTIONS = 'mention_label'
BILOU_TAG_OTHER = 'O'
BILOU_TAG_BEGIN = 'B'
BILOU_TAG_UNIT = 'U'
BILOU_TAG_LAST = 'L'
BILOU_TAG_INTER = 'I'

nlp = spacy.load("en")
RESTRICTED_POS_TAGS = {'PUNCT', 'SYM', 'DET', 'NUM', 'SPACE', 'PART'}


def global_tokenizer(text, restrict_by_pos=False, lowercase=False, filter_empty_token=True):
    if restrict_by_pos:
        token_list = [
            w.text for w in nlp(text) if w.pos_ not in RESTRICTED_POS_TAGS
        ]
    else:
        token_list = [w.text for w in nlp(text)]

    if lowercase:
        token_list = [w.lower() for w in token_list]

    if filter_empty_token:
        token_list = [w for w in token_list if len(w) > 0]

    return token_list


def clean_wiki_text(raw_text):
    def _replace_equal_num_spaces(match):
        match_len = len(match.group(1))
        return ' ' * match_len

    citation_stripped_text = re.sub(
        '(\[\d+\]|\[citation needed\])', _replace_equal_num_spaces, raw_text
    )
    return citation_stripped_text


class Entity(str):
    """
    An entity is string with some additional metadata.

    Currently, this consists of
        * a score in an arbtitrary range, which may be used to compare
            and order entities.
        * an optional type

    """

    def __new__(cls, v: str, score: float, typ: str=None):
        obj = str.__new__(cls, v)
        obj.score = score
        return obj

    def __repr__(self):
        return 'Entity(%s, %s, %s)' % (str(self), self.score, self.typ)


class Corpus(ABC):
    """
    A base class to represent a corpus.
    Limitation: Currently loads up the entire dataset in memory and stores in `data`
    """

    def __init__(self, data):
        self.data = data

    @abstractmethod
    def all_text(self):
        """
        A list of strings where each item corresponds to 1 document that need to be prcessed and
        vectorized
        :return: list
        """
        return []

    @abstractmethod
    def labels(self):
        """
        A list of labels in this corpus -- e.g. Binary (0/1) or Categorical
        :return: list
        """
        return []

    @abstractmethod
    def num_items(self):
        """
        Total number of items in the corpus -- e.g. number of sentences or mentions
        :return:
        """
        return 0


class Vocab(object):
    """
    Creating a vocabulary for the input corpus. Includes options to limit number of tokens by
    selected filter type. Default is to use CountVectorizer (and can be extended to other types)
    """

    def __init__(self, vocab, labels):
        self.vocabulary = vocab
        self.size = len(vocab)
        self.labels = labels

    def size(self):
        return len(self.vocabulary)

    @staticmethod
    def pkl_file(save_dir, prefix=""):
        return os.path.join(save_dir, prefix + "vocab.pickle")

    @staticmethod
    def save_vocab(save_dir, vocab, prefix=""):
        file_util.write_pickle(Vocab.pkl_file(save_dir, prefix), vocab)

    @staticmethod
    def count_vectorizer_fit(
        all_text,
        corpus,
        max_features=None,
        max_df_frac=1.0,
        min_df=0,
        to_lower=True
    ):
        if max_features is not None:
            count_vectorizer = CountVectorizer(
                max_df=max_df_frac,
                max_features=max_features,
                stop_words=None,
                lowercase=to_lower
            )
        else:
            count_vectorizer = CountVectorizer(
                max_df=max_df_frac,
                min_df=min_df / len(corpus.data),
                lowercase=to_lower
            )

        count_vectorizer.fit(tqdm.tqdm(all_text))
        return sorted(list(count_vectorizer.vocabulary_.keys()))

    @classmethod
    def build(
        cls,
        corpus: Corpus,
        filter_type='count',
        max_features=50000,
        to_lower=True
    ):
        all_text = corpus.all_text()

        if filter_type == 'count':
            vocab = cls.count_vectorizer_fit(
                all_text, corpus, max_features=max_features, to_lower=to_lower
            )

        labels = corpus.labels()
        return cls(vocab, labels)


class WordIndexer(object):
    """
    Assigns unique ID starting from 1 for each word in the vocab and lables in the training set
    """

    def __init__(self, to_lower):
        self._word_to_id = dict()
        self._label_to_id = dict()
        self.to_lower = to_lower

    def _transform_word(self, word):
        try:
            return word.lower() if self.to_lower else word
        except AttributeError:
            return word.lower()

    @classmethod
    def build(cls, vocab: Vocab, to_lower):
        indexer = cls(to_lower=to_lower)
        [indexer.add_word(word) for word in list(vocab.vocabulary)]
        [indexer.add_label(label) for label in vocab.labels]
        return indexer

    def add_word(self, word):
        word = self._transform_word(word)
        if word not in self._word_to_id:
            self._word_to_id[word] = len(self._word_to_id) + 1

    def word_to_id(self, word):
        word = self._transform_word(word)
        return self._word_to_id.get(word)

    def add_label(self, label):
        if label not in self._label_to_id:
            self._label_to_id[label] = len(self._label_to_id) + 1

    def get_label_id(self, label):
        return self._label_to_id.get(label)

    def seq_to_id(self, words):
        return list(
            filter(None.__ne__, [self.word_to_id(word) for word in words])
        )


class Batcher(object):
    """
    A generic class to batch items in the `Corpus`. Initialized by a `Corpus` object
    """

    def __init__(self, corpus: Corpus, batch_size=256):
        self.corpus = corpus
        self.batch_size = batch_size

    def batchify(self):
        """
        Produces batches of data in the corpus
        :return: yields one batch at a time
        """
        batch = []
        while True:
            for i in range(len(self.corpus.data)):
                if len(batch) == self.batch_size:
                    yield batch
                    del batch[:]
                else:
                    batch.append(self.corpus.data[i])


class Data(object):
    """
    A base class that manages the corpus, indexer, vocab and batcher for particular task. Each
    task e.g. claim extraction or entity linking will extend from this class and define how the
    elements in a batch are converted to features using the indexer.
    """

    def __init__(
        self,
        corpus: Corpus,
        indexer: WordIndexer,
        vocab: Vocab,
        batch_size=256
    ):
        self.corpus = corpus
        self.indexer = indexer
        self.vocab = vocab
        self.batch_size = batch_size
        self.batcher = Batcher(self.corpus, self.batch_size)

    @abstractmethod
    def generator(self):
        """
        Returns a generator that yields a batch of data and corresponding labels
        :return:
        """
        yield [], []


class CoNLLIO(object):
    '''
    Read and write CoNLL formatted files
    '''

    # See https://docs.google.com/document/d/1cmrIYdCqMk6fH2Cxukxz3XGaoMgQlvGK2UTM64fS8UU/edit#heading=h.5t953ig2kqmj
    # for a decription of the data format.
    # See https://docs.google.com/document/u/1/d/1dN644bNuuROhxHOcompGznFU0uzdN9AEQjmj_MisbRs/edit#heading=h.5t953ig2kqmj
    # for a description of the data format for entity linking

    # We'll make the assumption that the files are relatively small so there is no need for
    # streaming reads or writes.

    # to get the document ID
    RE_DOC_ID = re.compile(r'\((.+)\)')

    # various useful transformer methods for reading data
    @staticmethod
    def _o_mapper(x):
        # replace _ with 'O'
        if x == '_':
            return 'O'
        else:
            return x

    @staticmethod
    def _keyphrase_mapper(x):
        # replace the KeyEntity type with Entity type
        x = CoNLLIO._o_mapper(x)
        if x.endswith('KeyEntity'):
            return '{0}-{1}'.format(x.split('-', 1)[0], 'Entity')
        else:
            return x

    @staticmethod
    def _fix_split_entities(labels):
        # convert entity spans that straddle sentence boundaries to
        # two individual entity spans
        if labels[0][0] == 'I':
            labels[0] = 'B' + '-' + labels[0].split('-', 1)[1]
        elif labels[0][0] == 'L':
            labels[0] = 'U' + '-' + labels[0].split('-', 1)[1]

        if labels[-1][0] == 'B':
            labels[-1] = 'U' + '-' + labels[-1].split('-', 1)[1]
        elif labels[-1][0] == 'I':
            labels[-1] = 'L' + '-' + labels[-1].split('-', 1)[1]

    @staticmethod
    def _make_transform(labels, func):
        ret = [func(lab) for lab in labels]
        CoNLLIO._fix_split_entities(ret)
        return ret

    TRANSFORMERS = {
        'keyphrase_o_fixspans': {
            'mention_label': lambda x: CoNLLIO._make_transform(
                x, CoNLLIO._keyphrase_mapper)
        },
        'o_fixspans': {
            'mention_label': lambda x: CoNLLIO._make_transform(
                x, CoNLLIO._o_mapper)
        }
    }

    @staticmethod
    def _get_doc_id(line):
        # Try to get the document id.
        mo = CoNLLIO.RE_DOC_ID.search(line)
        if mo:
            return mo.groups()[0]
        else:
            return None

    @staticmethod
    def _parse_sentence(raw_sentence, schema, transformers):
        ret = []
        for token in raw_sentence:
            if len(token) != len(schema):
                raise ValueError('Invalid line, {0}'.format(' '.join(token)))
            parsed = dict(zip(schema, token))
            ret.append(parsed)

        # now call the transformers
        for col, func in transformers.items():
            values = func([token[col] for token in ret])
            for token, val in zip(ret, values):
                token[col] = val

        return ret

    @staticmethod
    def _reset_bilou(docs):
        for doc in docs:
            for sentence in doc['sents']:
                for token in sentence:
                    token[BILOU_FIELD_NAME_MENTIONS] = 'O'
                    token[BILOU_FIELD_NAME] = 'O'
            doc['mentions'] = []
        return docs

    @staticmethod
    def union(in_filenames, out_filename, schema=None):
        '''
        Aggregates (union) predictions in the input filenames and writes the result
        to output_filename. All input files must have the same schema.
        This method makes the assumption that the labels within each input filename
        are consistent. The following label sequence is consistent:
        [B I I L O O U U B L O], while the following label sequences are not:
        [I-Entity L-Entity O] # multi-token entities should start with B
        [B-Entity I-Entity O] # multi-token entities should end with L
        [B-Entity L-KeyEntity O] # incompatible types in a multi-token entity.

        Limitations:
        - This method assumes that the linking annotations within each
          input file do not overlap, because this complicates the merging logic
          considerably and most of the datasets we are working with don't have
          overlapping mentions.
        - This method preserves the same relations in the first input conll file.
          if any are present.
        - This method assumes that the labels are encoded using the BILUO scheme.
          B = beginning
          I = inside
          L = last
          U = unit
          O = other
        '''
        if len(in_filenames) < 2:
            logging.error(
                'The in_filenames parameter in the CoNLLIO.union method must '
                + 'be assigned a list of two or more CoNLL filenames.'
            )
            raise RuntimeError(
                'The in_filenames parameter in the CoNLLIO.union method must '
                + 'be assigned a list of two or more CoNLL filenames.'
            )

        first_file = True
        all_files_contents = []
        for in_filename in in_filenames:
            one_file_contents, one_file_schema = CoNLLIO.read(
                in_filename, schema
            )
            # If no schema is specified, use the inferred schema of the first input file
            # as the default schema.
            if first_file and schema is None:
                schema = one_file_schema
            # If the default schema (or the schema of the first input file) is not the same
            # as the schema of other input files, throw an exception.
            elif schema != one_file_schema:
                logging.error(
                    'CoNLLIO.union requires that all files in the in_filenames'
                    + 'parameter have the same schema.'
                )
                raise RuntimeError(
                    'CoNLLIO.union requires that all files in the in_filenames'
                    + 'parameter have the same schema.'
                )

            # Aggregate contents.
            all_files_contents.append(one_file_contents)

            first_file = False

        # Initialize the union prediction using the first file.
        aggregate_docs = CoNLLIO._reset_bilou(
            copy.deepcopy(all_files_contents[0])
        )

        bilou_field = BILOU_FIELD_NAME_MENTIONS
        if BILOU_FIELD_NAME in schema:
            bilou_field = BILOU_FIELD_NAME

        # Iterate through each sentence, obtain its mentions and create a merged list of
        # mentions. Then assign BILOU tokens to appropriate tokens in the aggregate_content
        for doc_index, aggregate_doc in enumerate(aggregate_docs):
            for sent_index, aggregate_doc_sentence in enumerate(
                aggregate_doc['sents']
            ):
                sentence_mentions_union = []
                for single_file_docs in all_files_contents:
                    sentence_mentions_union.extend(
                        [
                            m for m in single_file_docs[doc_index]['mentions']
                            if m['sentence_ind'] == sent_index
                        ]
                    )

                sentence_mentions_union = sorted(
                    sentence_mentions_union,
                    key=lambda x: x['token_end_ind'] - x['token_start_ind'],
                    reverse=True
                )
                # Get a list of unique list of mentions of the sentence from all files, sorted by
                # length. If overlapping mentions are found, the union picks the longest
                # mention.
                for mention in sentence_mentions_union:
                    label = mention['entity']
                    # Before assigning BILOU labels to tokens, make sure none of the tokens is
                    # part of another mention. Overlapping mentions are not allowed.
                    if all(
                        [
                            aggregate_doc_sentence[t_id]['mention_label'] ==
                            'O'
                            for t_id in range(
                                mention['token_start_ind'],
                                mention['token_end_ind']
                            )
                        ]
                    ):
                        start = mention['token_start_ind']
                        end = mention['token_end_ind']
                        mention_len = end - start
                        if mention_len == 1:
                            aggregate_doc_sentence[start][
                                bilou_field
                            ] = 'U-{}'.format(label)
                        elif mention_len > 1:
                            aggregate_doc_sentence[start][
                                bilou_field
                            ] = 'B-{}'.format(label)
                            for t_id in range(start + 1, end - 1):
                                aggregate_doc_sentence[t_id][
                                    bilou_field
                                ] = 'I-{}'.format(label)
                            aggregate_doc_sentence[end - 1][
                                bilou_field
                            ] = 'L-{}'.format(label)
                    else:
                        logging.warning(
                            "Found overlapping mentions at doc_index: {}, sent_index: {}".
                            format(doc_index, sent_index)
                        )

        CoNLLIO.write(aggregate_docs, out_filename, schema)

    @staticmethod
    def read(file_name, schema=None, column_transformers={}):
        '''
        Read the file.  The schema is inferred from the header in the file, or can be over-ridden
        by passing schema, a list of column names.

        column_transformers: a dict of column name -> callable functions
            used a hook to transform the raw value for each column.  Use
            cases are changing type (e.g. str -> int), implementing label
            transformations, etc.
        '''
        with file_util.open(file_name, 'r') as fin:
            lines = fin.read().strip().split('\n')

        # Get the schema.
        header = lines[0]
        if header.startswith('-DOCSTART-'):
            if schema is None:
                raise ValueError(
                    "Didn't find a schema in the file. "
                    "Schema can be specified as an argument in the CoNLLIO.read method."
                )
        else:
            # the line after a schema line must be blank or -DOCSTART-
            if lines[1].strip() != '':
                raise ValueError(
                    'The header line must be followed by a blank line'
                )
            file_schema = header.strip().split()
            if schema is not None:
                if len(schema) != len(file_schema):
                    raise ValueError(
                        'The provided schema is not consistent with the number '
                        'of columns in the file.'
                    )
            else:
                schema = file_schema

        if header.startswith('-DOCSTART-'):
            start = 0
        else:
            start = 1

        ret = []
        raw_sentence = []
        doc_id = None
        sents = []
        for line in lines[start:]:
            if line.startswith('-DOCSTART-'):
                if len(raw_sentence) > 0:
                    # clear out the last sentence
                    parsed = CoNLLIO._parse_sentence(
                        raw_sentence, schema, column_transformers
                    )
                    sents.append(parsed)
                    raw_sentence = []

                # add sentences to the return for the previous document
                if len(sents) > 0:
                    ret.append({'doc_id': doc_id, 'sents': sents})
                    sents = []

                # update the doc_id
                doc_id = CoNLLIO._get_doc_id(line)

            elif line.strip() == '':
                # end of a sentence
                if len(raw_sentence) > 0:
                    parsed = CoNLLIO._parse_sentence(
                        raw_sentence, schema, column_transformers
                    )
                    sents.append(parsed)
                    raw_sentence = []
            else:
                # a token in a sentence
                raw_sentence.append(line.strip().split())

        # the last sentence
        if len(raw_sentence) > 0:
            parsed = CoNLLIO._parse_sentence(
                raw_sentence, schema, column_transformers
            )
            sents.append(parsed)
        if len(sents) > 0:
            ret.append({'doc_id': doc_id, 'sents': sents})

        if BILOU_FIELD_NAME in schema:
            # process entities and mentions
            return CoNLLIO._process_entities(
                ret, schema, BILOU_FIELD_NAME
            ), schema
        else:
            return CoNLLIO._process_entities(
                ret, schema, BILOU_FIELD_NAME_MENTIONS
            ), schema

    @staticmethod
    def write(data, file_name, schema):
        '''
        Write the data to the file_name, with columns specified by the schema.
        '''
        with file_util.open(file_name, 'w') as fout:
            header = '\t'.join(schema) + '\n\n'
            fout.write(header)

            for doc in data:
                doc_start = '-DOCSTART-'
                doc_id = doc['doc_id']
                if doc_id is not None:
                    doc_start += ' ({0})'.format(doc_id)
                fout.write(doc_start + '\n\n')

                for sentence in doc['sents']:
                    for token in sentence:
                        line = '\t'.join(
                            ['{}'.format(token[col]) for col in schema]
                        ) + '\n'
                        fout.write(line)
                    # blank line after the sentence
                    fout.write('\n')

    @staticmethod
    def _process_entities(result, schema, bilou_field_name):
        def _parse_bilou(s):
            if s == 'O':
                return []
            biluo_entities = {}
            for ent in s.split('|'):
                bilou_letter, entity = ent.split('-', 1)
                if bilou_letter not in biluo_entities:
                    biluo_entities[bilou_letter] = []
                if 'EG:' in entity:
                    # The CRAFT ontologies do not come with EG. All EG entities are therefore
                    # ignored. This is the same as Tsai16
                    continue
                biluo_entities[bilou_letter].append(entity)
            return biluo_entities

        def _chunkify(sentence):
            """
            This method chunks the tokens in a sentence to contain a continuous set of non O
            bilou tags. Each chunk can have multiple mentions.
            :param sentence: The input sentence to chunk
            :return:
            """
            bilou_chunk = []
            for idx, token in enumerate(sentence):
                if token[bilou_field_name] == BILOU_TAG_OTHER:
                    if len(bilou_chunk) > 0:
                        yield bilou_chunk
                        bilou_chunk = []
                else:
                    token['bilou_map'] = _parse_bilou(token[bilou_field_name])
                    bilou_chunk.append((idx, token))
            if len(bilou_chunk) > 0:
                yield bilou_chunk

        def _is_contiguous(token_id_seq):
            token_id_seq = [int(x) for x in token_id_seq]
            conseq_diff = [
                token_id_seq[idx + 1] - e
                for idx, e in enumerate(token_id_seq)
                if idx < len(token_id_seq) - 1
            ]
            return set(conseq_diff) == set([1])

        def _parse_chunk(chunk):
            """
            This method processes each chunk into a list of mentions. It applies a greedy
            approach within the chunk. It creates a mention object for each UNIT tag. When a
            BEGIN label is found, we iteratively process the following tokens looking for either
            LAST or INTERMEDIATE tags (in that order) for the selected entity. This specified
            order makes this approach "greedy" -- i.e. greedily find a sequence of tokens that
            are linked to an entity.
            :param chunk:
            :return:
            """
            mentions = []
            idx = 0
            while idx < len(chunk):
                token_id, token_dict = chunk[idx]
                bilou_map = token_dict['bilou_map']
                if BILOU_TAG_UNIT in bilou_map:
                    for u_entity in bilou_map[BILOU_TAG_UNIT]:
                        mentions.append(
                            {
                                'mention':
                                    token_dict[schema[1]],
                                'entity':
                                    u_entity,
                                'token_start_ind':
                                    int(token_dict[schema[0]]) - 1,
                                'token_end_ind':
                                    int(token_dict[schema[0]])
                            }
                        )
                if BILOU_TAG_BEGIN in bilou_map:
                    # Find entities that start at this token and iterate over the following
                    # tokens looking for L and I tags for the identified entity.
                    for b_entity in bilou_map[BILOU_TAG_BEGIN]:
                        token_str = token_dict[schema[1]]
                        idx2 = idx + 1
                        token_id_seq = [token_dict[schema[0]]]
                        while idx2 < len(chunk):
                            token_id_2, token_dict_2 = chunk[idx2]
                            if BILOU_TAG_LAST in token_dict_2['bilou_map'] and b_entity in \
                                    token_dict_2['bilou_map'][BILOU_TAG_LAST]:
                                token_str += ' ' + token_dict_2[schema[1]]
                                token_dict_2['bilou_map'][BILOU_TAG_LAST
                                                         ].remove(b_entity)
                                token_id_seq.append(token_dict_2[schema[0]])
                                break
                            elif BILOU_TAG_INTER in token_dict_2['bilou_map'] and b_entity in \
                                    token_dict_2['bilou_map'][BILOU_TAG_INTER]:
                                token_str += ' ' + token_dict_2[schema[1]]
                                token_dict_2['bilou_map'][BILOU_TAG_INTER
                                                         ].remove(b_entity)
                                token_id_seq.append(token_dict_2[schema[0]])
                            idx2 += 1
                        # Store the token sequence for the mention and make sure it consists of
                        # contiguous list of tokens. See a test case in conll_el_failure.txt that
                        # should fail since a mention is over discontinuous list of tokens.
                        if _is_contiguous(token_id_seq):
                            mentions.append(
                                {
                                    'mention':
                                        token_str,
                                    'entity':
                                        b_entity,
                                    'token_start_ind':
                                        int(token_dict[schema[0]]) - 1,
                                    'token_end_ind':
                                        int(token_dict_2[schema[0]]),
                                }
                            )
                        else:
                            raise ValueError(
                                'Found BEGIN tag, but could not parse entity: {'
                                '0}.token_id={1}'.format(
                                    b_entity, token_dict[schema[0]]
                                )
                            )
                idx += 1
            return mentions

        for doc in result:
            doc['mentions'] = []
            for i_sent, sentence in enumerate(doc['sents']):
                sentence_mentions = []
                for chunk in _chunkify(sentence):
                    sentence_mentions.extend(_parse_chunk(chunk))
                for m in sentence_mentions:
                    m['sentence_ind'] = i_sent
                doc['mentions'].extend(sentence_mentions)

        return result

    @staticmethod
    def from_text(text):
        doc = {}
        doc['doc_id'] = hash(text)

        processed_doc = nlp(text)
        sentences = []
        for sentence in processed_doc.sents:
            tokens = []
            for idx, token in enumerate(sentence):
                token = {
                    'token_position': str(idx + 1),
                    'surface_form': str(token),
                    'mention_label': 'O',
                    'bilou_entities': 'O'
                }
                tokens.append(token)
            sentences.append(tokens)
        doc['sents'] = sentences
        schema = [
            'token_position', 'surface_form', 'mention_label', 'bilou_entities'
        ]

        return doc, schema

    @staticmethod
    def extract_mentions(docs, schema):
        return CoNLLIO._process_entities(
            docs, schema, BILOU_FIELD_NAME_MENTIONS
        )

    @staticmethod
    def extract_entities(docs, schema):
        return CoNLLIO._process_entities(docs, schema, BILOU_FIELD_NAME)

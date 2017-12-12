import itertools
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import emma.utils.string_utils as string_utils
import emma.constants as constants


# class for generating sparse features between entities of two KBs
class EngineeredFeatureGenerator:
    def __init__(self, s_token_to_idf: dict = None, t_token_to_idf: dict = None):
        """
        Initializae feature generator; token to idf dicts are taken from the candidate selector
        :param s_token_to_idf:
        :param t_token_to_idf:
        """
        self.STOP = set(stopwords.words('english'))

        if s_token_to_idf and t_token_to_idf:
            s_tokens_to_void = [k for k, v in s_token_to_idf.items() if v < constants.IDF_LIMIT]
            t_tokens_to_void = [k for k, v in t_token_to_idf.items() if v < constants.IDF_LIMIT]
            self.idf_stop = set(s_tokens_to_void + t_tokens_to_void)
            self.STOP = self.STOP.union(self.idf_stop)

        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en')
        self.token_dict = dict()

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
        """
        Tokenize string s
        :param s:
        :return:
        """
        return string_utils.tokenize_string(s, self.tokenizer, self.STOP)

    def _tokenize_list(self, l):
        """
        Tokenize each string in list l
        :param l:
        :return:
        """
        return [self._tokenize(i) for i in l]

    @staticmethod
    def _order_sublists(l):
        """
        Return sorted list of string tuples
        :param l:
        :return:
        """
        return [tuple(sorted(i)) for i in l]

    @staticmethod
    def _char_tokenize(s, ngram_size):
        """
        Generate character n-grams over string s
        :param s:
        :param ngram_size:
        :return:
        """
        return string_utils.get_character_n_grams(s, ngram_size)

    def _char_tokenize_list(self, l, ngram_size):
        """
        Generate character n-grams for all tokens in each sublist of input list l
        :param l:
        :param ngram_size:
        :return:
        """
        return [self._char_tokenize(i, ngram_size) for i in l]

    def _stem_tokens(self, t):
        """
        Stem each token in list t
        :param t:
        :return:
        """
        return [self.stemmer.stem(i) for i in t]

    def _lemmatize_tokens(self, t):
        """
        Lemmatize each token in list t
        :param t:
        :return:
        """
        return [self.lemmatizer.lemmatize(i) for i in t]

    def _stem_list(self, l):
        """
        Stem all tokens in each sublist of input list l
        :param l:
        :return:
        """
        return [self._stem_tokens(t) for t in l]

    def _lemmatize_list(self, l):
        """
        Lemmatize tokens in each sublist of input list l
        :param l:
        :return:
        """
        return [self._lemmatize_tokens(t) for t in l]

    @staticmethod
    def _acronym(t):
        """
        Returns acronym of input string t
        :param t:
        :return:
        """
        return ''.join([i[0] for i in t])

    def _acronym_list(self, l):
        """
        Returns list of acronyms for each item in list l
        :param l:
        :return:
        """
        return [self._acronym(t) for t in l]

    @staticmethod
    def _jaccard(a, b):
        """
        Return jaccard similarity between lists a and b
        :param a:
        :param b:
        :return:
        """
        return string_utils.get_jaccard_similarity(set(a), set(b))

    def _max_jaccard(self, alist, blist):
        """
        Returns max jaccard similarity between sublists in alist and blist
        :param alist:
        :param blist:
        :return:
        """
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
        """
        Returns whether set a overlaps set b
        :param a:
        :param b:
        :return:
        """
        return not set(a).isdisjoint(b)

    def _form_dict_entry(self, ent):
        """
        Form a dictionary of entity attributes
        :param ent:
        :return:
        """
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
        """
        Retrieve entity tokens from dict if exist; otherwise, compute and add to token dict
        :param ent:
        :return:
        """
        if ent['research_entity_id'] not in self.token_dict:
            self.token_dict[ent['research_entity_id']] = self._form_dict_entry(ent)
        return self.token_dict[ent['research_entity_id']]

    @staticmethod
    def _validate_entity(ent):
        """
        Validate entity data structure
        :param ent:
        :return:
        """
        if 'mesh_synonynms' in ent:
            ent['mesh_synonyms'] = ent['mesh_synonynms']
        if 'mesh_synonyms' not in ent:
            ent['mesh_synonyms'] = []
        if 'dbpedia_synonyms' not in ent:
            ent['dbpedia_synonyms'] = []
        if 'wiki_entities' not in ent:
            ent['wiki_entities'] = []
        return ent

    def calculate_features(self, s_ent: dict, t_ent: dict):
        """
        Calculate features between two entities s_ent and t_ent from source and target KBs respectively
        :param s_ent: entity from source KB
        :param t_ent: entity from target KB
        :return:
        """
        # validate entities have all necessary fields
        s_ent = self._validate_entity(s_ent)
        t_ent = self._validate_entity(t_ent)

        if s_ent and t_ent:

            # tokenize and process entity fields
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

            # create feature dictionary
            features = dict()

            features['has_same_canonical_name'] = has_same_canonical_name
            features['has_same_canonical_name_tokens'] = has_same_canonical_name_tokens
            features['has_same_canonical_name_token_set'] = has_same_canonical_name_token_set
            features['has_same_stemmed_name_tokens'] = has_same_stemmed_name_tokens
            features['has_same_stemmed_name_token_set'] = has_same_stemmed_name_token_set

            features['has_same_lemmatized_name_tokens'] = has_same_lemmatized_name_tokens
            features['has_same_lemmatized_name_token_set'] = has_same_lemmatized_name_token_set
            features['name_char_4gram_jaccard'] = name_char_4gram_jaccard
            features['name_char_5gram_jaccard'] = name_char_5gram_jaccard
            features['has_alias_in_common'] = has_alias_in_common

            features['has_alias_tokens_in_common'] = has_alias_tokens_in_common
            features['has_alias_token_set_in_common'] = has_alias_token_set_in_common
            features['alias_token_jaccard'] = alias_token_jaccard
            features['max_alias_token_jaccard'] = max_alias_token_jaccard
            features['max_alias_4gram_jaccard'] = max_alias_4gram_jaccard

            features['max_alias_5gram_jaccard'] = max_alias_5gram_jaccard
            features['has_same_acronym'] = has_same_acronym
            features['definition_token_jaccard'] = definition_token_jaccard
            features['has_same_wiki_entity'] = has_same_wiki_entity
            features['wiki_entity_jaccard'] = wiki_entity_jaccard

            features['max_wiki_entity_jaccard'] = max_wiki_entity_jaccard
            features['has_same_mesh_synonym'] = has_same_mesh_synonym
            features['mesh_synonym_jaccard'] = mesh_synonym_jaccard
            features['max_mesh_synonym_jaccard'] = max_mesh_synonym_jaccard
            features['has_same_dbpedia_synonym'] = has_same_dbpedia_synonym

            features['dbpedia_synonym_jaccard'] = dbpedia_synonym_jaccard
            features['max_dbpedia_synonym_jaccard'] = max_dbpedia_synonym_jaccard
            features['has_overlapping_synonym'] = has_overlapping_synonym
            features['all_synonym_jaccard'] = all_synonym_jaccard
            features['max_all_synonym_jaccard'] = max_all_synonym_jaccard

            features['has_same_root_word'] = has_same_root_word
            features['root_word_jaccard'] = root_word_jaccard

            return features

        else:
            raise Exception("Input entities invalid...")


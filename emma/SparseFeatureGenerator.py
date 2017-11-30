import os
import sys
import pickle
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.metrics.distance import edit_distance
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import emma.utils.string_utils as string_utils
import emma.constants as constants

# class for generating sparse features between entities of two KBs
class SparseFeatureGenerator:
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

    @staticmethod
    def _normalize_ent(ent: dict):
        """
        Given an entity, normalize its name, aliases, definition, and parent/child names
        :param ent:
        :return:
        """
        norm_ent = dict()
        norm_ent['research_entity_id'] = ent['research_entity_id']
        norm_ent['canonical_name'] = string_utils.normalize_string(ent['canonical_name'])
        norm_ent['aliases'] = [string_utils.normalize_string(a) for a in ent['aliases']]
        norm_ent['definition'] = string_utils.normalize_string(ent['definition'])
        norm_ent['wiki_entities'] = [string_utils.normalize_string(s) for s in ent['wiki_entities']]
        norm_ent['mesh_synonyms'] = [string_utils.normalize_string(s) for s in ent['mesh_synonyms']]
        norm_ent['dbpedia_synonyms'] = [string_utils.normalize_string(s) for s in ent['dbpedia_synonyms']]
        norm_ent['par_relations'] = set([string_utils.normalize_string(i) for i in ent['par_relations']])
        norm_ent['chd_relations'] = set([string_utils.normalize_string(i) for i in ent['chd_relations']])
        norm_ent['sib_relations'] = set([string_utils.normalize_string(i) for i in ent['sib_relations']])
        norm_ent['syn_relations'] = set([string_utils.normalize_string(i) for i in ent['syn_relations']])
        return norm_ent

    def _compute_tokens(self, ent: dict):
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

    def calculate_features(self, s_ent: dict, t_ent: dict):
        """
        Calculate features between two entities s_ent and t_ent from source and target KBs respectively
        :param s_ent: entity from source KB
        :param t_ent: entity from target KB
        :return:
        """

        s_ent = self._normalize_ent(s_ent)
        t_ent = self._normalize_ent(t_ent)

        if s_ent['research_entity_id'] in self.token_dict:
            s_name_tokens, s_stem_tokens, s_lemm_tokens, \
            s_char_tokens, s_alias_tokens, s_def_tokens = self.token_dict[s_ent['research_entity_id']]
        else:
            s_name_tokens, s_stem_tokens, s_lemm_tokens, \
            s_char_tokens, s_alias_tokens, s_def_tokens = self._compute_tokens(s_ent)
            self.token_dict[s_ent['research_entity_id']] = (s_name_tokens, s_stem_tokens, s_lemm_tokens,
                                                            s_char_tokens, s_alias_tokens, s_def_tokens)

        if t_ent['research_entity_id'] in self.token_dict:
            t_name_tokens, t_stem_tokens, t_lemm_tokens, \
            t_char_tokens, t_alias_tokens, t_def_tokens = self.token_dict[t_ent['research_entity_id']]
        else:
            t_name_tokens, t_stem_tokens, t_lemm_tokens, \
            t_char_tokens, t_alias_tokens, t_def_tokens = self._compute_tokens(t_ent)
            self.token_dict[t_ent['research_entity_id']] = (t_name_tokens, t_stem_tokens, t_lemm_tokens,
                                                            t_char_tokens, t_alias_tokens, t_def_tokens)

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
        has_siblings = (len(s_ent['sib_relations']) > 0 and len(t_ent['sib_relations']) > 0)
        has_synonyms = (len(s_ent['syn_relations']) > 0 and len(t_ent['syn_relations']) > 0)

        percent_parents_in_common = 0.0
        percent_children_in_common = 0.0
        percent_siblings_in_common = 0.0
        percent_synonyms_in_common = 0.0

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

        if has_siblings:
            max_siblings_in_common = (len(s_ent['sib_relations']) + len(t_ent['sib_relations'])) / 2
            percent_siblings_in_common = len(
                s_ent['sib_relations'].intersection(t_ent['sib_relations'])
            ) / max_siblings_in_common

        if has_synonyms:
            max_synonyms_in_common = (len(s_ent['syn_relations']) + len(t_ent['syn_relations'])) / 2
            percent_synonyms_in_common = len(
                s_ent['syn_relations'].intersection(t_ent['syn_relations'])
            ) / max_synonyms_in_common

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

        num_mesh_syn_shared = len(set(s_ent['mesh_synonyms']).intersection(set(t_ent['mesh_synonyms'])))
        num_total_mesh_syn = len(set(s_ent['mesh_synonyms']).union(set(t_ent['mesh_synonyms'])))
        shares_mesh_synonym = (num_mesh_syn_shared > 0)
        mesh_syn_jaccard_sim = num_mesh_syn_shared / num_total_mesh_syn

        num_dbp_syn_shared = len(set(s_ent['dbpedia_synonyms']).intersection(set(t_ent['dbpedia_synonyms'])))
        num_total_dbp_syn = len(set(s_ent['dbpedia_synonyms']).union(set(t_ent['dbpedia_synonyms'])))
        shares_dbpedia_synonym = (num_dbp_syn_shared > 0)
        dbpedia_syn_jaccard_sim = num_dbp_syn_shared / num_total_dbp_syn

        num_wiki_ent_shared = len(set(s_ent['wiki_entities']).intersection(set(t_ent['wiki_entities'])))
        num_total_wiki_ent = len(set(s_ent['wiki_entities']).union(set(t_ent['wiki_entities'])))
        shares_wikipedia_entity = (num_wiki_ent_shared > 0)
        wikipedia_ent_jaccard_sim = num_wiki_ent_shared / num_total_wiki_ent

        features = dict()

        features['has_same_canonical_name'] = has_same_canonical_name
        features['has_same_stemmed_name'] = has_same_stemmed_name
        features['has_same_lemmatized_name'] = has_same_lemmatized_name
        features['has_same_char_tokens'] = has_same_char_tokens
        features['has_alias_in_common'] = has_alias_in_common

        features['name_token_jaccard_similarity'] = name_token_jaccard_similarity
        features['inverse_name_token_edit_distance'] = inverse_name_token_edit_distance
        features['name_stem_jaccard_similarity'] = name_stem_jaccard_similarity
        features['inverse_name_stem_edit_distance'] = inverse_name_stem_edit_distance
        features['name_lemm_jaccard_similarity'] = name_lemm_jaccard_similarity

        features['inverse_name_lemm_edit_distance'] = inverse_name_lemm_edit_distance
        features['name_char_jaccard_similarity'] = name_char_jaccard_similarity
        features['inverse_name_char_edit_distance'] = inverse_name_char_edit_distance
        features['max_alias_token_jaccard'] = max_alias_token_jaccard
        features['min_alias_edit_distance'] = min_alias_edit_distance

        features['percent_parents_in_common'] = percent_parents_in_common
        features['percent_children_in_common'] = percent_children_in_common
        features['percent_siblings_in_common'] = percent_siblings_in_common
        features['percent_synonyms_in_common'] = percent_synonyms_in_common
        features['has_same_acronym'] = has_same_acronym

        features['has_same_name_root_word'] = has_same_name_root_word
        features['has_same_name_chunk_heads'] = has_same_name_chunk_heads
        features['name_chunk_heads_jaccard_similarity'] = name_chunk_heads_jaccard_similarity
        features['has_same_alias_root_word'] = has_same_alias_root_word
        features['has_same_alias_chunk_heads'] = has_same_alias_chunk_heads

        features['alias_chunk_heads_jaccard_similarity'] = alias_chunk_heads_jaccard_similarity
        features['def_jaccard_similarity'] = def_jaccard_similarity
        features['shares_mesh_synonym'] = shares_mesh_synonym
        features['mesh_syn_jaccard_sim'] = mesh_syn_jaccard_sim
        features['shares_dbpedia_synonym'] = shares_dbpedia_synonym

        features['dbpedia_syn_jaccard_sim'] = dbpedia_syn_jaccard_sim
        features['shares_wikipedia_entity'] = shares_wikipedia_entity
        features['wikipedia_ent_jaccard_sim'] = wikipedia_ent_jaccard_sim

        return features


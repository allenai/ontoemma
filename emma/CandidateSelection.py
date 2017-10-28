import os
import sys
from collections import defaultdict
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from emma.kb.kb_utils_refactor import KBEntity, KnowledgeBase
import emma.utils.string_utils as string_utils
import emma.constants as constants


# class that generates match candidates for entities in a source kb from a target kb
class CandidateSelection:
    def __init__(self, source_kb: KnowledgeBase, target_kb: KnowledgeBase):
        """
        Initialize and load/build candidate map
        :param s: source KB as KnowledgeBase object
        :param t: target KB as KnowledgeBase object
        :param fpath: file path for candidate file
        """
        self.STOP = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')

        self.source_kb = source_kb
        self.target_kb = target_kb

        self.s_ent_num = len(self.source_kb.entities)
        self.t_ent_num = len(self.target_kb.entities)

        self.s_token_to_ents = defaultdict(set)
        self.t_token_to_ents = defaultdict(set)

        self.s_ent_to_tokens = dict()
        self.t_ent_to_tokens = dict()

        self.s_token_to_idf = dict()
        self.t_token_to_idf = dict()

        self._build_map()

        self.EVAL_TOP_KS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        self.EVAL_OUTPUT_FILE = None
        self.EVAL_MISSED_FILE = None

    def _generate_token_map(self, ents: List[KBEntity]):
        """
        Generates token-to-entity and entity-to-token map for an input list
        of KBEntity objects
        :param ents: list of KBEntity objects
        :return: token-to-entity dict and entity-to-token dict
        """
        # maps entity id key to word tokens in entity
        ent_to_tokens = dict()

        # maps token key to entities that have that token
        token_to_ents = defaultdict(set)

        for ent in ents:
            ent_id = ent.research_entity_id

            # tokenize all names and definitions
            name_tokens = []
            char_tokens = []
            for name in ent.aliases:
                name_tokens += string_utils.tokenize_string(name, self.tokenizer, self.STOP)
                char_tokens += [
                    ''.join(c)
                    for c in
                    string_utils.get_character_n_grams(string_utils.normalize_string(name), constants.NGRAM_SIZE)
                ]

            def_tokens = string_utils.tokenize_string(ent.definition, self.tokenizer, self.STOP)

            # combine tokens
            tokens = set(name_tokens).union(set(char_tokens)).union(set(def_tokens))

            # add to ent-to-token map
            ent_to_tokens[ent_id] = tokens

            # add to token-to-ent map
            for tok in tokens:
                token_to_ents[tok].add(ent_id)

            # generate n-grams for all aliases
            for ng in char_tokens:
                token_to_ents[ng].add(ent_id)
        return token_to_ents, ent_to_tokens

    def _build_map(self):
        """
        Builds a mapping between word tokens and identifiers. Entities that
        share a word token are given as candidate pairs. Candidate selections
        are saved to file.
        :return:
        """
        # generate token maps for s and t
        self.s_token_to_ents, self.s_ent_to_tokens = self._generate_token_map(
            self.source_kb.entities
        )
        self.t_token_to_ents, self.t_ent_to_tokens = self._generate_token_map(
            self.target_kb.entities
        )

        # keep only tokens shared by both s and t
        s_token_keys = set(self.s_token_to_ents.keys())
        t_token_keys = set(self.t_token_to_ents.keys())
        keep_tokens = s_token_keys.intersection(t_token_keys)

        self.s_token_to_ents = {
            k: self.s_token_to_ents[k]
            for k in keep_tokens
        }
        self.t_token_to_ents = {
            k: self.t_token_to_ents[k]
            for k in keep_tokens
        }

        # generate maps for idf scores
        self.s_token_to_idf = {
            k: string_utils.get_idf(self.s_ent_num, len(self.s_token_to_ents[k]))
            for k in keep_tokens
        }
        self.t_token_to_idf = {
            k: string_utils.get_idf(self.t_ent_num, len(self.t_token_to_ents[k]))
            for k in keep_tokens
        }
        return

    def select_candidates(self, s_ent_id):
        """
        Returns sorted target candidates for an input source entity
        :param s_ent_id: entity research_entity_id from source kb
        :return:
        """
        s_tokens = self.s_ent_to_tokens.get(s_ent_id)
        t_ent_ids = defaultdict(float)

        # get matches in t for each token in s_ent
        for token in s_tokens:
            # check if token exists in both KBs and IDF score over limit
            if self.s_token_to_ents.get(token) and self.t_token_to_ents.get(token) \
                    and self.s_token_to_idf[token] >= constants.IDF_LIMIT \
                    and self.t_token_to_idf[token] >= constants.IDF_LIMIT:
                for t_match in self.t_token_to_ents[token]:
                    t_ent_ids[t_match] += self.t_token_to_idf[token]

        sorted_t = sorted(t_ent_ids, key=lambda k: t_ent_ids[k], reverse=True)
        return sorted_t

    def eval(self, gold_mappings):
        """
        Evaluate the yield and recall of the generated candidates compared against a gold alignment
        :param gold_mappings: tuple pairs of research_entity_ids from source and target ontologies
        :return:
        """
        gold_count = len(gold_mappings)
        cand_counts = [0] * len(self.EVAL_TOP_KS)
        pos_counts = [0] * len(self.EVAL_TOP_KS)

        if self.EVAL_OUTPUT_FILE is not None and not os.path.exists(self.EVAL_OUTPUT_FILE):
            with open(self.EVAL_OUTPUT_FILE, 'w') as outf:
                outf.write(
                    "KB1\tKB2\tKB1_ents\tKB2_ents\tgold\ttop_k\tcand_count\tprecision@k\trecall@k\n"
                )

        missed = []
        keep_missed = False
        if self.EVAL_MISSED_FILE is not None:
            keep_missed = True

        for s_ent_id in set([i[0] for i in gold_mappings]):
            candidates = self.select_candidates(s_ent_id)
            for k_ind, k in enumerate(self.EVAL_TOP_KS):
                cand_counts[k_ind] += len(candidates[:k])
                for t_ent_id in candidates[:k]:
                    if (s_ent_id, t_ent_id) in gold_mappings:
                        pos_counts[k_ind] += 1
                if keep_missed and k == constants.KEEP_TOP_K_CANDIDATES:
                    correct_matches = set([t_id for (s_id, t_id) in gold_mappings if s_id == s_ent_id])
                    t_missed = correct_matches.difference(set(candidates[:k]))
                    missed += [(s_ent_id, t_id) for t_id in t_missed]

        precisions = [p / c for p, c in zip(pos_counts, cand_counts)]
        recalls = [p / gold_count for p in pos_counts]

        # print evaluation results
        sys.stdout.write("Source KB: %s\n" % self.source_kb.name)
        sys.stdout.write("Target KB: %s\n" % self.target_kb.name)
        sys.stdout.write("Source entities: %i\n" % len(self.source_kb.entities))
        sys.stdout.write("Target entities: %i\n" % len(self.target_kb.entities))
        sys.stdout.write("Number of positive gold alignments: %i\n" % gold_count)
        sys.stdout.write("Top ks: %s\n" % ','.join([str(i) for i in self.EVAL_TOP_KS]))
        sys.stdout.write("Candidate count: %s\n" % ','.join([str(i) for i in cand_counts]))
        sys.stdout.write("Precision @ k: %s\n" % ','.join(['{:.2f}'.format(i) for i in precisions]))
        sys.stdout.write("Recall @ k: %s\n" % ','.join(['{:.2f}'.format(i) for i in recalls]))

        # write evaluation results to file
        if self.EVAL_OUTPUT_FILE is not None and os.path.exists(self.EVAL_OUTPUT_FILE):
            with open(self.EVAL_OUTPUT_FILE, 'a') as outf:
                outf.write(
                    "%s\t%s\t%i\t%i\t%i\t%s\t%s\t%s\t%s\n" % (
                        self.source_kb.name, self.target_kb.name,
                        len(self.source_kb.entities), len(self.target_kb.entities),
                        gold_count,
                        ','.join([str(i) for i in self.EVAL_TOP_KS]),
                        ','.join([str(i) for i in cand_counts]),
                        ','.join(['{:.2f}'.format(i) for i in precisions]),
                        ','.join(['{:.2f}'.format(i) for i in recalls])
                    )
                )

        # write missed alignments to file
        if keep_missed:
            with open(self.EVAL_MISSED_FILE, 'w') as outf:
                for (s_id, t_id) in missed:
                    s_aliases = self.source_kb.get_entity_by_research_entity_id(s_id).aliases
                    t_aliases = self.target_kb.get_entity_by_research_entity_id(t_id).aliases
                    outf.write("%s\t%s\t%s\t%s\n" % (
                        s_id, t_id, ','.join(s_aliases), ','.join(t_aliases)
                    ))
        return



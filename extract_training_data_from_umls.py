import os, sys
import jsonlines
import itertools
import random
import glob
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

from emma.utils.base import App
from emma.utils.traits import Unicode
from emma.paths import StandardFilePath
from emma.kb.kb_utils_refactor import KBEntity, KBRelation, KnowledgeBase
from emma.CandidateSelection import CandidateSelection
import emma.constants as constants
from emma.OntoEmma import OntoEmma


# class for extracting concept mappings from UMLS
class UMLSExtractor(App):
    """
    Class for extracting data from UMLS
    -- Some useful abbreviations --
    CUI: concept unique identifier
    AUI: atom unique identifier
    SAB: source kb name
    CODE: id in source kb
    STR: string name value
    TS: preferred status (P = preferred name, S = non-preferred)
    STT: string type (PF = preferred form, VO = variant,
                        VC = case variant, VW = word-order variant,
                        VCW = case and word-order variant)
    REL: relationship abbreviation
    RELA: relation attribute
    """

    paths = StandardFilePath()

    UMLS_DIR = paths.ontoemma_umls_subset_dir
    OUTPUT_DIR = paths.ontoemma_umls_output_dir
    OUTPUT_KB_DIR = paths.ontoemma_kb_dir
    TRAINING_DIR = paths.ontoemma_training_dir
    CONTEXT_DIR = os.path.join(paths.ontoemma_root_dir, 'kb_context')

    # name sort order (by preferred name status)
    TTY_sort_order = {"MH": 0, "NM": 0,           # main heading, supplementary concept name
                      "PT": 1, "PEP": 1, "PCE": 1,   # preferred terms
                      "ET": 2, "ETAL": 2, "CE": 2, "SY": 2, "SYN": 2, "NA": 2, "ETCF": 2, "ETCLIN": 2,  # entry terms, aliases, synonyms
                      "AB": 3, "ACR": 3}          # abbreviations and acronyms

    TTY_sort_order = defaultdict(lambda: 4, TTY_sort_order)

    # string literals from UMLS
    UMLS_EXPANDED_FORM_STR = 'expanded_form'
    UMLS_REL_INVERSE_STR = 'rel_inverse'
    UMLS_REL_STR = "REL"
    UMLS_NOCODE_STR = 'NOCODE'

    umls_training_data = [
    ]  # mappings between different pairs of kbs true and false

    umls_kbs = dict()  # collection of KBs as KnowledgeBase

    def main(self, args):

        # header for mapping provenance
        self.umls_header = 'UMLS' + self.UMLS_DIR.split('/')[-3]

        self.concept_file = os.path.join(self.UMLS_DIR, 'MRCONSO.RRF')
        self.definition_file = os.path.join(self.UMLS_DIR, 'MRDEF.RRF')
        self.hierarchy_file = os.path.join(self.UMLS_DIR, 'MRHIER.RRF')
        self.relation_file = os.path.join(self.UMLS_DIR, 'MRREL.RRF')
        self.document_file = os.path.join(self.UMLS_DIR, 'MRDOC.RRF')
        self.semtype_file = os.path.join(self.UMLS_DIR, 'MRSTY.RRF')

        self.done_file = os.path.join(self.OUTPUT_DIR, "processed.txt")

        # sys.stdout.write("Extracting concepts...\n")
        # concepts = self.extract_concepts()
        # sys.stdout.write("Number of concepts: %i\n" % len(concepts))
        #
        # sys.stdout.write("Extracting concept mappings...\n")
        # self.extract_mappings(concepts)
        #
        # sys.stdout.write("Write mappings to file...\n")
        # self.write_mappings_to_file()
        #
        # sys.stdout.write("Collapsing concepts to KB entities...\n")
        # kb_entities, aui_to_research_entity_id_dict = self.collapse_concepts(concepts)
        #
        # sys.stdout.write("Extracting definitions...\n")
        # kb_entities = self.extract_definitions(kb_entities, aui_to_research_entity_id_dict)
        #
        # sys.stdout.write("Extracting relations...\n")
        # relations = self.extract_relationships(aui_to_research_entity_id_dict)
        # sys.stdout.write("Number of relations: %i\n" % len(relations))
        #
        # sys.stdout.write("Adding relations to entities...\n")
        # kb_entities = self.append_relations_to_entities(kb_entities, relations)
        #
        # sys.stdout.write("Creating knowledgebases...\n")
        # self.create_umls_kbs(kb_entities)

        self.load_mappings_from_file()

        sys.stdout.write("Sampling negative mappings...\n")
        self.extract_negative_mappings()

        sys.stdout.write("Splitting all training data...\n")
        self.split_training_data()

        sys.stdout.write("DONE.\n")

        return

    def extract_concepts(self):
        """
        Parse UMLS MRCONSO (concepts) file and extract concepts
        :return: dict of concept terms from different kbs,
                    key: CUI, value: [SAB, CODE, AUI, TS, STT, STR]
        """
        concepts = defaultdict(list)
        with open(self.concept_file, 'r') as f:
            for l in f:
                # order of segs in UMLS MRCONSO file
                umls_cui, umls_lat, umls_ts, umls_lui, umls_stt, \
                    umls_sui, umls_ispref, umls_aui, umls_saui, umls_scui, \
                    umls_sdui, umls_sab, umls_tty, umls_code, umls_str, \
                    umls_srl, umls_suppress, umls_cvf, _ = l.split('|')
                if umls_sab in constants.TRAINING_KBS \
                        and umls_code != self.UMLS_NOCODE_STR:
                    concepts[umls_cui].append(
                        [
                            umls_sab, umls_code, umls_aui, umls_tty, umls_str
                        ]
                    )
        return concepts

    def extract_mappings(self, concepts):
        """
        Parse UMLS concepts and extract cross-db mappings
        :param concepts: dict produced by extract_concepts,
                    key: CUI, value: [SAB, CODE, AUI, TS, STT, STR]
        """

        # all mappings from UMLS
        mappings = defaultdict(list)

        for cui, entries in concepts.items():
            # keep only cross-db mappings
            cui_str = '{}:{}'.format(self.umls_header, cui)
            uids = set([tuple(i[:2]) for i in entries])
            pairs = [
                sorted([p, q]) for p, q in itertools.combinations(uids, 2)
                if (p[0] in constants.TRAINING_KBS) and
                (q[0] in constants.TRAINING_KBS) and (p[0] != q[0])
            ]
            for p, q in pairs:
                p_id = '{}:{}'.format(p[0], p[1])
                q_id = '{}:{}'.format(q[0], q[1])
                mappings[(p[0], q[0])].append([p_id, q_id, 1, cui_str])

        for k in mappings:
            mappings[k].sort()
            mappings[k] = list(m for m, _ in itertools.groupby(mappings[k]))

        self.umls_training_data = mappings
        return

    def collapse_concepts(self, concepts):
        """
        Collapse TS, STT, and STR fields from concepts to preferred name and other names
        :param concepts: dict of concepts and name occurrences
        :return: concept dict with names collapsed,
                key: CUI, value: [SAB, CODE, AUIs, pref_name, [aliases]]
        """
        cui_to_ents = defaultdict(list)
        for cui in concepts:
            for umls_sab, umls_code, umls_aui, umls_tty, umls_str in concepts[cui]:
                cui_to_ents[(umls_sab, umls_code)].append([umls_aui, umls_tty, umls_str])
        sys.stdout.write("CUIS map to %i entities\n" % len(cui_to_ents))

        entities = defaultdict(dict)
        aui_to_entity_id = dict()
        for (sab, code), entries in cui_to_ents.items():

            # extract KB entity information
            ent_dict = dict()
            ent_dict['research_entity_id'] = '{}:{}'.format(sab, code)
            entries.sort(key=lambda val: self.TTY_sort_order[val[1]])
            ent_dict['auis'] = [i[0] for i in entries]
            ent_dict['canonical_name'] = entries[0][2]
            ent_dict['aliases'] = [a.lower() for a in set([n[2] for n in entries])]
            ent_dict['definition'] = []
            ent_dict['relations'] = []
            entities[sab][code] = ent_dict

            # map AUIs to research entity id
            for aui in ent_dict['auis']:
                aui_to_entity_id[aui] = (sab, code)
        return entities, aui_to_entity_id

    def extract_definitions(self, entities, rid_mapdict):
        """
        Extract definitions from UMLS MRDEF and append to concepts
        :param entities: list of concepts with names
        :param rib_mapdict: mapping dict from AUIs to research_entity_ids
        :return: concepts with appended definitions
        """
        # read UMLS MRDEF file and parse data to extract concept definitions
        with open(self.definition_file, 'r') as f:
            for l in f:
                # order of segs in UMLS MRDEF file
                umls_cui, umls_aui, umls_atui, umls_satui, umls_sab, \
                    umls_def, umls_suppress, umls_cvf, _ = l.split('|')

                aui_match = rid_mapdict.get(umls_aui)
                if aui_match:
                    (sab, code) = aui_match
                    entities[sab][code]['definition'].append(umls_def)
        return entities

    def extract_relationships(self, rid_mapdict):
        """
        Read UMLS MRREL and extract relationships
        :param rid_mapdict: mapping dict from AUIs to research_entity_ids
        :return: a list of relationships between CUIS
                    [SAB, CUI1, AUI1, CUI2, AUI2, relationship]
        """
        rel_symmetric = []
        with open(self.document_file, 'r') as f:
            for l in f:
                # order of segs: DOCKEY, VALUE, TYPE, EXPL
                umls_dockey, umls_value, umls_type, umls_expl, _ = l.split('|')
                if (umls_dockey == self.UMLS_REL_STR and umls_type == self.UMLS_REL_INVERSE_STR) and \
                        (umls_value == umls_expl):
                    rel_symmetric.append(umls_value)

        relations = []
        with open(self.relation_file, 'r') as f:
            for l in f:
                # order of segs in UMLS MRREL file
                umls_cui1, umls_aui1, umls_stype1, umls_rel, \
                    umls_cui2, umls_aui2, umls_stype2, umls_rela, \
                    umls_rui, umls_srui, umls_sab, umls_sl, \
                    umls_rg, umls_dir, umls_suppress, umls_cvf, _ = l.split('|')

                if umls_sab in constants.TRAINING_KBS:
                    ent_id1 = rid_mapdict.get(umls_aui1)
                    ent_id2 = rid_mapdict.get(umls_aui2)
                    if ent_id1 and ent_id2:
                        if umls_rel and umls_rel != 'NULL':
                            relations.append([ent_id1, ent_id2, umls_rel, umls_rel in rel_symmetric])
        return relations

    def append_relations_to_entities(self, entities, relations):
        """
        Add relationship ids to entities
        :param entities: dict of kb entities
        :param relations: list of relations between research_entity_ids
        :return: add relations to kb entities
        """
        for rel in relations:
            (sab, code) = rel[0]
            entities[sab][code]['relations'].append(rel)
        return entities

    def create_umls_kbs(self, entities):
        """
        From entity list, create several KnowledgeBase objects with entities from different KBs
        :param entities: dict of entities
        :return:
        """
        for kb_name in constants.TRAINING_KBS:
            sys.stdout.write("\tCreating KB %s\n" % kb_name)
            kb = KnowledgeBase()
            kb.name = kb_name

            entities_to_add = entities[kb_name]

            for ent_id, ent_val in entities_to_add.items():
                new_ent = KBEntity(
                    ent_val['research_entity_id'], ent_val['canonical_name'],
                    ent_val['aliases'], ' '.join(ent_val['definition'])
                )
                for ent1_id, ent2_id, rel_type, symmetric in ent_val['relations']:
                    rel_id1 = '{}:{}'.format(ent1_id[0], ent1_id[1])
                    rel_id2 = '{}:{}'.format(ent2_id[0], ent2_id[1])
                    new_rel = KBRelation(
                        rel_type, [rel_id1, rel_id2], symmetric
                    )
                    kb.add_relation(new_rel)
                    rel_ind = len(kb.relations) - 1
                    new_ent.relation_ids.append(rel_ind)
                kb.add_entity(new_ent)

            # write plain KB to json
            out_fname = 'kb-{}.json'.format(kb_name)
            kb.dump(kb, os.path.join(self.OUTPUT_KB_DIR, out_fname))

            # add context to kb and write to file
            self.add_context_to_kb(kb)
        return

    @staticmethod
    def string_equiv(ent1, ent2):
        """
        Returns string equivalence between two entities' aliases
        :param ent1:
        :param ent2:
        :return:
        """
        e1_aliases = set(
            [a.lower().replace('_', ' ').replace('-', '') for a in ent1.aliases]
        )
        e2_aliases = set(
            [a.lower().replace('_', ' ').replace('-', '') for a in ent2.aliases]
        )
        if len(e1_aliases.intersection(e2_aliases)) > 0:
            return True

        return False

    def sample_negative_mappings(self, kb1, kb2, tp_mappings):
        """
        Given two KBs and true positive mapping, sample easy and hard negatives
        for training data
        :param kb1: source KB
        :param kb2: target KB
        :param tp_mappings: true positive mappings
        :return: negative pairs (0 for hard negatives, -1 for easy negatives)
        """
        cand_sel = CandidateSelection(kb1, kb2)

        sys.stdout.write('\t\tExtracting candidates...\n')
        kb2_ent_ids = [e.research_entity_id for e in kb2.entities]
        tps = set([tuple(i[:2]) for i in tp_mappings])

        cand_negs = []
        rand_negs = []

        # sample negatives for each true positive (TP)
        for tp in tps:
            # get candidates for source entity
            cands = cand_sel.select_candidates(tp[0])[:constants.KEEP_TOP_K_CANDIDATES]
            if tp[1] in cands:
                cands.remove(tp[1])
            # sample hard negatives
            cand = cands[:min(constants.NUM_HARD_NEGATIVE_PER_POSITIVE, len(cands))]
            cand_negs += [tuple([tp[0], c]) for c in cand]
            # sample easy negatives
            rand = random.sample(kb2_ent_ids, constants.NUM_EASY_NEGATIVE_PER_POSITIVE)
            rand_negs += [tuple([tp[0], r]) for r in rand]

        # filter negatives
        hard_negatives = set(cand_negs).difference(tps)
        easy_negatives = set(rand_negs).difference(tps
                                                  ).difference(hard_negatives)

        # append negative pairs together with labels: (0 = hard negative, -1 = easy negative)
        neg_pairs = []
        for neg in hard_negatives:
            neg_pairs.append([neg[0], neg[1], 0, self.umls_header])
        for neg in easy_negatives:
            neg_pairs.append([neg[0], neg[1], -1, self.umls_header])

        return neg_pairs

    def extract_negative_mappings(self):
        """
        sample negative pairings from entities
        :param mappings: positive mappings
        :param entities: entities grouped by kb
        :return:
        """
        for kb_names, kb_training_data in self.umls_training_data.items():

            # Format file names
            kb1_fname = 'kb-{}.json'.format(kb_names[0])
            kb2_fname = 'kb-{}.json'.format(kb_names[1])
            training_fname = '{}-{}.tsv'.format(kb_names[0], kb_names[1])

            kb1_path = os.path.join(self.OUTPUT_KB_DIR, kb1_fname)
            kb2_path = os.path.join(self.OUTPUT_KB_DIR, kb2_fname)
            training_path = os.path.join(
                self.OUTPUT_DIR, 'training', 'nonequiv', training_fname
            )

            # initialize KBs
            s_kb = KnowledgeBase()
            t_kb = KnowledgeBase()

            # load KBs
            sys.stdout.write("\tLoading %s and %s\n" % kb_names)
            s_kb = s_kb.load(kb1_path)
            t_kb = t_kb.load(kb2_path)

            kb_training_keep = []
            # keep only non-equiv mappings
            for s_id, t_id, score, _ in kb_training_data:
                if not self.string_equiv(s_kb.get_entity_by_research_entity_id(s_id),
                                         t_kb.get_entity_by_research_entity_id(t_id)):
                    kb_training_keep.append((s_id, t_id, score, _))

            sys.stdout.write("\t\tKeeping %i out of %i positive mappings.\n" % (
                len(kb_training_keep), len(kb_training_data))
            )

            # sample negatives using candidate selection module
            sys.stdout.write(
                "\t\tSampling negatives between %s and %s\n" % kb_names
            )
            neg_mappings = self.sample_negative_mappings(
                s_kb, t_kb, kb_training_keep
            )

            # write negative mappings to training data file
            if neg_mappings:
                # write positive and negative training mappings to disk
                self.write_mapping_to_file(
                    training_path, kb_training_data + neg_mappings
                )

                # append kb pair to done file
                with open(self.done_file, 'a') as outf:
                    outf.write('%s\n' % training_path)
        return

    @staticmethod
    def _kb_entity_to_training_json(ent, kb):
        """
        Given entity and its origin KB, return a json representation of the entiy with extracted parent, children,
        synonym, and sibling relations represented by canonical_name
        :param ent:
        :param kb:
        :return:
        """
        parent_ids = [kb.relations[rel_id].entity_ids[1]
                      for rel_id in ent.relation_ids
                      if kb.relations[rel_id].relation_type in constants.UMLS_PARENT_REL_LABELS]

        child_ids = [kb.relations[rel_id].entity_ids[1]
                     for rel_id in ent.relation_ids
                     if kb.relations[rel_id].relation_type in constants.UMLS_CHILD_REL_LABELS]

        synonym_ids = [kb.relations[rel_id].entity_ids[1]
                       for rel_id in ent.relation_ids
                       if kb.relations[rel_id].relation_type in constants.UMLS_SYNONYM_REL_LABELS]

        sibling_ids = [kb.relations[rel_id].entity_ids[1]
                       for rel_id in ent.relation_ids
                       if kb.relations[rel_id].relation_type in constants.UMLS_SIBLING_REL_LABELS]

        parents = [kb.get_entity_by_research_entity_id(i).canonical_name
                   for i in parent_ids if i in kb.research_entity_id_to_entity_index]

        children = [kb.get_entity_by_research_entity_id(i).canonical_name
                    for i in child_ids if i in kb.research_entity_id_to_entity_index]

        synonyms = [kb.get_entity_by_research_entity_id(i).canonical_name
                    for i in synonym_ids if i in kb.research_entity_id_to_entity_index]

        siblings = [kb.get_entity_by_research_entity_id(i).canonical_name
                    for i in sibling_ids if i in kb.research_entity_id_to_entity_index]

        return {
            'research_entity_id': ent.research_entity_id,
            'canonical_name': ent.canonical_name,
            'aliases': ent.aliases,
            'definition': ent.definition,
            'other_contexts': ent.other_contexts,
            'par_relations': parents,
            'chd_relations': children,
            'syn_relations': synonyms,
            'sib_relations': siblings
        }

    def split_training_data(self):
        """
        Process and split data into training development and test sets
        :return:
        """
        all_kb_names = constants.TRAINING_KBS + constants.DEVELOPMENT_KBS
        training_file_dir = os.path.join(self.OUTPUT_DIR, 'training', 'nonequiv')

        output_training_data = os.path.join(self.TRAINING_DIR, 'nonequiv', 'ontoemma.context.train')
        output_development_data = os.path.join(self.TRAINING_DIR, 'nonequiv', 'ontoemma.context.dev')
        output_test_data = os.path.join(self.TRAINING_DIR, 'nonequiv', 'ontoemma.context.test')

        context_files = glob.glob(os.path.join(self.OUTPUT_KB_DIR, '*context.json'))
        context_kbs = [os.path.basename(f).split('-')[1] for f in context_files]
        training_files = glob.glob(os.path.join(training_file_dir, '*.tsv'))
        file_names = [os.path.splitext(os.path.basename(f))[0] for f in training_files]

        training_labels = []
        training_dat = []

        emma = OntoEmma()

        for fname, fpath in zip(file_names, training_files):
            (kb1_name, kb2_name) = fname.split('-')
            if kb1_name in all_kb_names and kb2_name in all_kb_names \
                    and kb1_name in context_kbs and kb2_name in context_kbs:
                sys.stdout.write("Processing %s and %s\n" % (kb1_name, kb2_name))
                kb1 = emma.load_kb(
                    os.path.join(self.OUTPUT_KB_DIR, 'kb-{}-context.json'.format(kb1_name))
                )
                kb2 = emma.load_kb(
                    os.path.join(self.OUTPUT_KB_DIR, 'kb-{}-context.json'.format(kb2_name))
                )
                alignment = emma.load_alignment(fpath)

                for (e1, e2, score) in alignment:
                    kb1_ent = kb1.get_entity_by_research_entity_id(e1)
                    kb2_ent = kb2.get_entity_by_research_entity_id(e2)
                    training_labels.append(int(score))
                    training_dat.append({
                        "source_entity": self._kb_entity_to_training_json(kb1_ent, kb1),
                        "target_entity": self._kb_entity_to_training_json(kb2_ent, kb2)
                    })
            else:
                sys.stdout.write("Skipping %s and %s\n" % (kb1_name, kb2_name))

        training_dat, test_dat, training_labels, test_labels = train_test_split(
            training_dat,
            training_labels,
            stratify=training_labels,
            test_size=constants.TEST_PART
        )

        training_dat, development_dat, training_labels, development_labels = train_test_split(
            training_dat,
            training_labels,
            stratify=training_labels,
            test_size=constants.DEVELOPMENT_PART
        )

        training_labels = self._replace_negative_labels(training_labels)
        development_labels = self._replace_negative_labels(development_labels)
        test_labels = self._replace_negative_labels(test_labels)

        with jsonlines.open(output_training_data, mode='w') as writer:
            for label, dat in zip(training_labels, training_dat):
                writer.write({"label": label,
                              "source_ent": dat["source_entity"],
                              "target_ent": dat["target_entity"]})

        with jsonlines.open(output_development_data, mode='w') as writer:
            for label, dat in zip(development_labels, development_dat):
                writer.write({"label": label,
                              "source_ent": dat["source_entity"],
                              "target_ent": dat["target_entity"]})

        with jsonlines.open(output_test_data, mode='w') as writer:
            for label, dat in zip(test_labels, test_dat):
                writer.write({"label": label,
                              "source_ent": dat["source_entity"],
                              "target_ent": dat["target_entity"]})
        return

    @staticmethod
    def _replace_negative_labels(l):
        """
        replace negative values in list with 0
        :param l:
        :return:
        """
        for i, n in enumerate(l):
            if n == -1:
                l[i] = 0
        return l

    @staticmethod
    def write_mapping_to_file(fpath, mappings):
        with open(fpath, 'w') as outf:
            for p, q, tp, cui in mappings:
                outf.write("%s\t%s\t%i\t%s\n" % (p, q, int(tp), cui))
        return

    def write_mappings_to_file(self):
        """
        Write mappings to file
        Format for mappings is three-column tab-delimited
        research_entity_id from KB1, research_entity_id from KB2, provenance of mapping (UMLS CUI)
        :return:
        """
        for k, v in self.umls_training_data.items():
            fname = '{}-{}.tsv'.format(k[0], k[1])
            with open(os.path.join(self.OUTPUT_DIR, 'mappings', fname),
                      'w') as outf:
                for p, q, tp, cui in v:
                    outf.write("%s\t%s\t%i\t%s\n" % (p, q, int(tp), cui))
        return

    def load_mappings_from_file(self):
        """
        Load mappings from directory
        :return:
        """
        done_list = []
        with open(self.done_file, 'r') as f:
            for l in f:
                done_list.append(l.strip())

        self.umls_training_data = dict()
        mapping_files = glob.glob(
            os.path.join(self.OUTPUT_DIR, 'mappings', '*.tsv')
        )
        for fpath in mapping_files:
            if fpath not in done_list:
                fname, fext = os.path.splitext(os.path.basename(fpath))
                names = fname.split('-')
                v = []
                with open(fpath, 'r') as f:
                    for l in f:
                        parts = l.strip().split('\t')
                        if parts[2] == '1':
                            v.append(parts)
                self.umls_training_data[tuple(names)] = v
        return

    def add_context_to_kb(self, kb):
        """
        Iterates through KBs and add context if it exists; save to new json file
        :return:
        """
        if kb.name in constants.TRAINING_KBS + constants.DEVELOPMENT_KBS:
            context_path = os.path.join(self.CONTEXT_DIR, '{}-contexts.pickle'.format(kb.name))
            output_path = os.path.join(self.OUTPUT_KB_DIR, 'kb-{}-context.json'.format(kb.name))

            sys.stdout.write("Loading context dict\n")
            context_dict = pickle.load(open(context_path, 'rb'))

            sys.stdout.write("Adding context to entities\n")

            counter = 0
            for ent_name, contexts in context_dict.items():
                if contexts:
                    counter += 1
                    ent_matches = kb.get_entity_by_canonical_name(ent_name)
                    for ent in ent_matches:
                        ent.other_contexts = list([c for c in contexts if c != ""])

            sys.stdout.write("%i of %i entities w/ context\n" % (counter, len(kb.entities)))

            sys.stdout.write("Writing enriched KB to file\n")
            kb.dump(kb, output_path)
        else:
            sys.stdout.write("%s not a training KB\n" % kb.name)

        return

UMLSExtractor.run(__name__)

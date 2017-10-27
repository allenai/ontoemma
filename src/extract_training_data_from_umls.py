import os, sys
import csv
import itertools
import random
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split

from base import App
from base.traits import Unicode
from scigraph.paths import StandardFilePath
from scigraph.kb.kb_utils_refactor import KBEntity, KBRelation, KnowledgeBase
from scigraph.ontology_matching.CandidateSelection import CandidateSelection
import scigraph.ontology_matching.constants as constants


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

    paths = StandardFilePath(release_root='/net/nfs.corp/s2-research/scigraph/data/', version='')

    UMLS_DIR = paths.ontoemma_umls_subset_dir
    OUTPUT_DIR = paths.ontoemma_umls_output_dir
    OUTPUT_KB_DIR = paths.ontoemma_kb_dir
    TRAINING_DIR = paths.ontoemma_training_dir

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

    # TRAINING/DEVELOPMENT SPLIT
    TRAINING_PART = 0.7
    DEVELOPMENT_PART = 0.3

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

        sys.stdout.write("Extracting concepts...\n")
        concepts = self.extract_concepts()
        sys.stdout.write("Number of concepts: %i\n" % len(concepts))

        sys.stdout.write("Extracting concept mappings...\n")
        self.extract_mappings(concepts)

        sys.stdout.write("Write mappings to file...\n")
        self.write_mappings_to_file()

        sys.stdout.write("Collapsing concepts to KB entities...\n")
        kb_entities, aui_to_research_entity_id_dict = self.collapse_concepts(concepts)

        sys.stdout.write("Extracting definitions...\n")
        kb_entities = self.extract_definitions(kb_entities, aui_to_research_entity_id_dict)

        sys.stdout.write("Extracting relations...\n")
        relations = self.extract_relationships(aui_to_research_entity_id_dict)
        sys.stdout.write("Number of relations: %i\n" % len(relations))

        sys.stdout.write("Adding relations to entities...\n")
        kb_entities = self.append_relations_to_entities(kb_entities, relations)

        sys.stdout.write("Creating knowledgebases...\n")
        self.create_umls_kbs(kb_entities)

        # self.load_mappings_from_file()

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

            # write KB to json
            out_fname = 'kb-{}.json'.format(kb_name)
            kb.dump(kb, os.path.join(self.OUTPUT_KB_DIR, out_fname))

        return

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
            # sample hard negatives
            cand = random.sample(cands, min(5, len(cands)))
            cand_negs += [tuple([tp[0], c]) for c in cand]
            # sample easy negatives
            rand = random.sample(kb2_ent_ids, 5)
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
                self.OUTPUT_DIR, 'training', training_fname
            )

            # initialize KBs
            s_kb = KnowledgeBase()
            t_kb = KnowledgeBase()

            # load KBs
            sys.stdout.write("\tLoading %s and %s\n" % kb_names)
            s_kb = s_kb.load(kb1_path)
            t_kb = t_kb.load(kb2_path)

            # sample negatives using candidate selection module
            sys.stdout.write(
                "\t\tSampling negatives between %s and %s\n" % kb_names
            )
            neg_mappings = self.sample_negative_mappings(
                s_kb, t_kb, kb_training_data
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

    def split_training_data(self):
        """
        Process and split training data
        :return:
        """
        # alignment files generated from UMLS
        align_paths = glob.glob(
            os.path.join(self.OUTPUT_DIR, 'training', '*.tsv')
        )
        alignments = []

        for fpath in align_paths:
            print(fpath)
            a = []
            # read alignments out of file
            with open(fpath, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for s_ent, t_ent, label, provenance in reader:
                    a.append([s_ent, t_ent, int(label), provenance])

            # keep data if long enough
            if len(a) >= constants.MIN_TRAINING_SET_SIZE:
                alignments += a

        # alignment labels (1=pos, 0=hard neg, -1=easy neg)
        labels = [i[2] for i in alignments]

        # use stratified sampling to split alignment data
        training_pairs, development_pairs, training_labels, development_labels = train_test_split(
            alignments,
            labels,
            stratify=labels,
            test_size=self.DEVELOPMENT_PART
        )

        # replace easy negative labels with negative labels (-1 -> 0)
        training_labels = self._replace_negative_labels(training_labels)
        development_labels = self._replace_negative_labels(development_labels)

        # write training set to file
        with open(os.path.join(self.TRAINING_DIR, 'training_data.tsv'),
                  'w') as outf:
            for p, l in zip(training_pairs, training_labels):
                outf.write('%s\t%s\t%i\t%s\n' % (p[0], p[1], l, p[3]))

        # write dev set to file
        with open(os.path.join(self.TRAINING_DIR, 'development_data.tsv'),
                  'w') as outf:
            for p, l in zip(development_pairs, development_labels):
                outf.write('%s\t%s\t%i\t%s\n' % (p[0], p[1], l, p[3]))
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
            os.path.join(self.OUTPUT_DIR, 'training', '*.tsv')
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


UMLSExtractor.run(__name__)

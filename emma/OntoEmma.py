import os
import sys
import csv
import json
import tqdm
import math
import itertools
import requests
import jsonlines
import numpy as np
import pickle
from collections import defaultdict
from lxml import etree
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError

from emma.OntoEmmaLRModel import OntoEmmaLRModel
from emma.OntoEmmaRFModel import OntoEmmaRFModel
from emma.kb.kb_utils_refactor import KBEntity, KnowledgeBase
from emma.kb.kb_load_refactor import KBLoader
from emma.CandidateSelection import CandidateSelection
from emma.SparseFeatureGenerator import SparseFeatureGenerator
from emma.paths import StandardFilePath
import emma.constants as constants

from allennlp.commands.train import train_model_from_file
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.commands.evaluate import evaluate as evaluate_allennlp
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

from torch.cuda import device


# class for training an ontology matcher and aligning input ontologies
class OntoEmma:
    def __init__(self):
        paths = StandardFilePath()
        mesh_syn_file = os.path.join(paths.ontoemma_synonym_dir, 'mesh_synonyms.pickle')
        dbpedia_syn_file = os.path.join(paths.ontoemma_synonym_dir, 'dbpedia_synonyms.pickle')

        self.mesh_synonyms = pickle.load(open(mesh_syn_file, 'rb'))
        self.dbpedia_synonyms = pickle.load(open(dbpedia_syn_file, 'rb'))

    @staticmethod
    def load_kb(kb_path):
        """
        Load KnowledgeBase specified at kb_path
        :param kb_path: path to knowledge base
        :return:
        """
        sys.stdout.write("\tLoading %s...\n" % kb_path)

        assert kb_path is not None
        assert kb_path != ''

        kb_name = os.path.basename(kb_path)

        kb = KnowledgeBase()

        # load kb
        if kb_path.endswith('.json') or kb_path.endswith(
            '.pickle'
        ) or kb_path.endswith('.pkl'):
            kb = kb.load(kb_path)
        elif kb_path.endswith('.obo') or kb_path.endswith('.OBO'):
            kb = KBLoader.import_obo_kb(kb_name, kb_path)
        elif kb_path.endswith('.owl') or kb_path.endswith('.rdf') or \
            kb_path.endswith('.OWL') or kb_path.endswith('.RDF'):
            kb = KBLoader.import_owl_kb(kb_name, kb_path)
        elif kb_path.endswith('.ttl') or kb_path.endswith('.n3'):
            sys.stdout.write('This program cannot parse your file type.\n')
            raise NotImplementedError()
        else:
            val = URLValidator()
            try:
                val(kb_path)
            except ValidationError:
                raise

            response = requests.get(kb_path, stream=True)
            response.raise_for_status()
            temp_file = 'temp_file_ontoemma.owl'
            with open(temp_file, 'wb') as outf:
                for block in response.iter_content(1024):
                    outf.write(block)
            kb = KBLoader.import_owl_kb('', temp_file)
            os.remove(temp_file)

        sys.stdout.write("\tEntities: %i\n" % len(kb.entities))

        return kb

    def add_synonyms(self, kb):
        """
        Add synonyms to kb from MeSH and DBpedia
        :param kb:
        :return:
        """
        counter = 0
        for ent in kb.entities:
            syns = []
            for a in ent.aliases:
                syns += self.mesh_synonyms[a]
                syns += self.dbpedia_synonyms[a]
            if syns:
                ent.aliases = list(set(ent.aliases + syns))
                counter += 1
        sys.stdout.write('%i entities added synonyms.\n' % counter)
        return kb

    @staticmethod
    def _load_alignment_from_tsv(gold_path):
        """
        Parse alignments from tsv gold alignment file path.
        File format given by format specified by
        https://docs.google.com/document/d/1VSeMrpnKlQLrJuh9ffkq7u7aWyQuIcUj4E8dUclReXM
        :param gold_path: path to gold alignment file
        :return:
        """
        mappings = []
        for s_ent, t_ent, label, _ in csv.reader(
            open(gold_path, 'r'), delimiter='\t'
        ):
            mappings.append((s_ent, t_ent, float(label)))
        return mappings

    @staticmethod
    def _load_alignment_from_json(gold_path):
        """
        Parse alignments from json gold alignment file path.
        :param gold_path: path to gold alignment file
        :return:
        """
        mappings = []

        # open data file and read lines
        with open(gold_path, 'r') as f:
            for line in tqdm.tqdm(f):
                dataline = json.loads(line)
                s_ent = dataline['source_ent']
                t_ent = dataline['target_ent']
                label = dataline['label']
                mappings.append((s_ent['research_entity_id'], t_ent['research_entity_id'], float(label)))
        return mappings

    @staticmethod
    def _load_alignment_from_rdf(gold_path):
        """
        Parse alignments from rdf gold alignment file path
        :param gold_path: path to gold alignment file
        :return:
        """
        mappings = []

        # parse the file
        tree = etree.parse(gold_path)
        root = tree.getroot()
        ns = root.nsmap
        ns['alignment'
           ] = 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
        maps = root.find('alignment:Alignment',
                         ns).findall('alignment:map', ns)
        resource = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'

        # parse matches
        for m in maps:
            cell = m.find('alignment:Cell', ns)
            ent1 = cell.find('alignment:entity1', ns).get(resource)
            ent2 = cell.find('alignment:entity2', ns).get(resource)
            meas = cell.find('alignment:measure', ns).text
            mappings.append((ent1, ent2, meas))

        return set(mappings)

    def load_alignment(self, gold_path):
        """
        Load alignments from gold file.
        File format specified by format specified by
        https://docs.google.com/document/d/1VSeMrpnKlQLrJuh9ffkq7u7aWyQuIcUj4E8dUclReXM
        :param gold_path: path to gold alignment file
        :return:
        """
        sys.stdout.write("\tLoading %s\n" % gold_path)
        assert os.path.exists(gold_path)
        fname, fext = os.path.splitext(gold_path)
        if fext == '.tsv':
            return self._load_alignment_from_tsv(gold_path)
        elif fext == '.rdf':
            return self._load_alignment_from_rdf(gold_path)
        else:
            try:
                return self._load_alignment_from_json(gold_path)
            except:
                raise NotImplementedError(
                    "Unknown input alignment file type. Cannot parse."
                )

    @staticmethod
    def _alignments_to_pairs_and_labels(file_path):
        """
        Convert jsonlines alignments to pairs and labels
        File format of input file specified by
        https://docs.google.com/a/allenai.org/document/d/1t8cwpTRqcscFEZOQJrtTMAhjAYzlA_demc9GCY0xYaU
        :param file_path:
        :return:
        """
        pairs = []
        labels = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                pairs.append([obj['source_ent'], obj['target_ent']])
                labels.append(obj['label'])
        return pairs, labels

    @staticmethod
    def _form_json_entity(ent_to_json: KBEntity, kb: KnowledgeBase):
        """
        Forms json representation of entity from kb
        :param ent_to_json:
        :param kb:
        :return:
        """
        all_rels = [kb.relations[r_id] for r_id in ent_to_json.relation_ids]
        par_ents = [
            kb.get_entity_by_research_entity_id(r.entity_ids[1]) for r in all_rels
            if r.relation_type in constants.UMLS_PARENT_REL_LABELS
        ]
        chd_ents = [
            kb.get_entity_by_research_entity_id(r.entity_ids[1]) for r in all_rels
            if r.relation_type in constants.UMLS_CHILD_REL_LABELS
        ]
        return {
            'research_entity_id': ent_to_json.research_entity_id,
            'canonical_name': ent_to_json.canonical_name,
            'aliases': ent_to_json.aliases,
            'definition': ent_to_json.definition,
            'other_contexts': ent_to_json.other_contexts,
            'par_relations': [e.canonical_name for e in par_ents],
            'chd_relations': [e.canonical_name for e in chd_ents]
        }

    def _apply_model_train(self, model, model_path, config_file):
        """
        Apply loaded model to config_file data and save
        :param model:
        :param model_path:
        :param config_file:
        :return:
        """
        # read model config
        with open(config_file, 'r') as f:
            config = json.load(f)

        # parse parameters
        training_data_path = config['train_data_path']
        dev_data_path = config['validation_data_path']

        # load training and dev data
        training_pairs, training_labels = self._alignments_to_pairs_and_labels(training_data_path)
        dev_pairs, dev_labels = self._alignments_to_pairs_and_labels(dev_data_path)

        sys.stdout.write('Training data size: %i\n' % len(training_labels))

        # generate features for training pairs
        feat_gen_train = SparseFeatureGenerator()
        training_features = []
        for s_ent, t_ent in tqdm.tqdm(training_pairs, total=len(training_pairs)):
            training_features.append(
                feat_gen_train.calculate_features(s_ent, t_ent)
            )

        sys.stdout.write('Development data size: %i\n' % len(dev_labels))

        # generate features for development pairs
        feat_gen_dev = SparseFeatureGenerator()
        dev_features = []
        for s_ent, t_ent in tqdm.tqdm(dev_pairs, total=len(dev_pairs)):
            dev_features.append(
                feat_gen_dev.calculate_features(s_ent, t_ent)
            )

        model.train(training_features, training_labels)

        training_accuracy = model.score_accuracy(training_features, training_labels)
        sys.stdout.write(
            "Accuracy on training data set: %.2f\n" % training_accuracy
        )

        dev_accuracy = model.score_accuracy(dev_features, dev_labels)
        sys.stdout.write(
            "Accuracy on development data set: %.2f\n" % dev_accuracy
        )

        model.save(model_path)
        return

    def _train_lr(self, model_path: str, config_file: str):
        """
        Train a logistic regression model
        :param model_path:
        :param config_file:
        :return:
        """
        model = OntoEmmaLRModel()
        self._apply_model_train(model, model_path, config_file)
        return

    def _train_rf(self, model_path: str, config_file: str):
        """
        Train a random forest model
        :param model_path:
        :param config_file:
        :return:
        """
        model = OntoEmmaRFModel()
        self._apply_model_train(model, model_path, config_file)
        return

    def _train_nn(self, model_path: str, config_file: str):
        """
        Train a neural network model
        :param model_path:
        :param config_file:
        :return:
        """
        # import allennlp ontoemma classes (to register -- necessary, do not remove)
        from emma.allennlp_classes.ontoemma_dataset_reader import OntologyMatchingDatasetReader
        from emma.allennlp_classes.ontoemma_model import OntoEmmaNN

        with open(config_file) as json_data:
            configuration = json.load(json_data)

        cuda_device = configuration['trainer']['cuda_device']

        if cuda_device >= 0:
            with device(cuda_device):
                train_model_from_file(config_file, model_path)
        else:
            train_model_from_file(config_file, model_path)
        return

    def train(
        self, model_type: str, model_path: str, config_file: str
    ):
        """
        Train model
        :param model_type: type of model (nn, lr etc)
        :param model_path: path to ontoemma model
        :param config_file: path to training data, dev data, config for nn
        :return:
        """
        assert model_type in constants.IMPLEMENTED_MODEL_TYPES
        assert config_file is not None
        assert os.path.exists(config_file)
        sys.stdout.write("Training {} model...\n".format(constants.IMPLEMENTED_MODEL_TYPES[model_type]))

        if model_type == "nn":
            self._train_nn(model_path, config_file)
        elif model_type == "lr":
            self._train_lr(model_path, config_file)
        elif model_type == "rf":
            self._train_rf(model_path, config_file)

        sys.stdout.write("done.\n")
        return

    def _evaluate_lr(self, model_path: str, evaluation_data_file: str):
        """

        :param model_path:
        :param evaluation_data_file:
        :return:
        """
        # load model from disk
        model = OntoEmmaLRModel()
        model.load(model_path)

        # load evaluation data
        eval_pairs, eval_labels = self._alignments_to_pairs_and_labels(evaluation_data_file)

        sys.stdout.write('Evaluation data size: %i\n' % len(eval_labels))

        # initialize feature generator
        feat_gen = SparseFeatureGenerator()
        eval_features = []
        for s_ent, t_ent in tqdm.tqdm(eval_pairs, total=len(eval_pairs)):
            eval_features.append(
                feat_gen.calculate_features(s_ent, t_ent)
            )

        # compute metrics
        tp, fp, tn, fn = (0, 0, 0, 0)
        precision, recall, accuracy, f1_score = (0.0, 0.0, 0.0, 0.0)

        for features, label in zip(eval_features, eval_labels):
            prediction = model.predict_entity_pair(features)
            if prediction[0][1] > constants.MAX_SCORE_THRESHOLD and label == 1:
                tp += 1
            elif prediction[0][1] > constants.MAX_SCORE_THRESHOLD and label == 0:
                fp += 1
            elif prediction[0][0] > constants.MAX_SCORE_THRESHOLD and label == 1:
                fn += 1
            else:
                tn += 1

        if tp + fp > 0:
            precision = tp / (tp + fp)
        if tp + fn > 0:
            recall = tp / (tp + fn)
        if tp + fp + fn + tn > 0:
            accuracy = (tp + tn)/(tp + fp + fn + tn)
        if precision + recall > 0.0:
            f1_score = (2 * precision * recall / (precision + recall))

        metrics = {'precision': precision,
                   'recall': recall,
                   'accuracy': accuracy,
                   'f1_score': f1_score}
        return metrics

    def _evaluate_nn(self, model_path: str, evaluation_data_file: str, cuda_device: int):
        """

        :param model_path:
        :param evaluation_data_file:
        :param cuda_device:
        :return:
        """
        # import allennlp ontoemma classes (to register -- necessary, do not remove)
        from emma.allennlp_classes.ontoemma_dataset_reader import OntologyMatchingDatasetReader
        from emma.allennlp_classes.ontoemma_model import OntoEmmaNN

        # Load from archive
        archive = load_archive(model_path, cuda_device)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        model.eval()

        # Load the evaluation data
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
        evaluation_data_path = evaluation_data_file
        dataset = dataset_reader.read(evaluation_data_path)

        # compute metrics
        dataset.index_instances(model.vocab)
        iterator = DataIterator.from_params(config.pop("iterator"))
        metrics = evaluate_allennlp(model, dataset, iterator, cuda_device)

        return metrics

    def evaluate(self, model_type: str, model_path: str, evaluation_data_file: str, cuda_device: int):
        """
        Evaluate trained model on some dataset specified in the config file
        :param model_type
        :param model_path:
        :param config_file:
        :param cuda_device: GPU device number
        :return:
        """
        assert model_type in constants.IMPLEMENTED_MODEL_TYPES
        assert os.path.exists(model_path)
        assert evaluation_data_file is not None
        assert os.path.exists(evaluation_data_file)

        metrics = dict()

        if model_type == "nn":
            metrics = self._evaluate_nn(model_path, evaluation_data_file, cuda_device)
        elif model_type == "lr":
            metrics = self._evaluate_lr(model_path, evaluation_data_file)

        sys.stdout.write('Metrics:\n')
        for key, metric in metrics.items():
            sys.stdout.write("\t%s: %s\n" % (key, metric))
        return

    @staticmethod
    def _get_region_around_ent(start_ent, kb):
        """
        Compute region around entity in kb, returning a dictionary of paths for each entity in the region
        :param ent:
        :param kb:
        :return:
        """
        regions = dict()
        regions[start_ent.research_entity_id] = []

        steps = 0
        next_step = [start_ent]

        while steps < constants.NUM_STEPS_FOR_KB_REGION:
            this_step = next_step
            next_step = []
            for current_ent in this_step:
                rels = [kb.relations[rel_id] for rel_id in current_ent.relation_ids]
                rel_tuples = [(r.relation_type, r.entity_ids[1]) for r in rels]
                for t, next_ent in rel_tuples:
                    if next_ent not in regions:
                        regions[next_ent] = regions[current_ent.research_entity_id][:]
                        regions[next_ent].append((current_ent.research_entity_id, t))
                        next_step.append(kb.get_entity_by_research_entity_id(next_ent))
            steps += 1
        return regions

    @staticmethod
    def _get_distance_weight(path1, path2):
        """
        Calculate weight based on two path lengths; if path lengths are both zero, the weight is 1
        :param path1:
        :param path2:
        :return:
        """
        return math.exp(-(len(path1) + len(path2)) / 2)

    @staticmethod
    def _get_rep_similarity(rep1, rep2):
        """
        Compute similarity between two tensors; currently just returns cosine similarity
        :param rep1:
        :param rep2:
        :return:
        """
        r1 = np.array(rep1)
        r2 = np.array(rep2)

        normalized_r1 = r1 / np.linalg.norm(r1)
        normalized_r2 = r2 / np.linalg.norm(r2)

        return sum((normalized_r1 * normalized_r2) / (np.linalg.norm(normalized_r1) * np.linalg.norm(normalized_r2)))

    @staticmethod
    def _align_string_equiv(s_kb, t_kb):
        """
        Align entities in two KBs using string equivalence
        :param s_kb:
        :param t_kb:
        :return:
        """
        alignment = []
        s_aliases = dict()
        t_aliases = dict()
        s_matched = set([])
        t_matched = set([])
        for s_ent in s_kb.entities:
            s_aliases[s_ent.research_entity_id] = set(
                [a.lower().replace('_', ' ').replace('-', '') for a in s_ent.aliases]
            )
        for t_ent in t_kb.entities:
            t_aliases[t_ent.research_entity_id] = set(
                [a.lower().replace('_', ' ').replace('-', '') for a in t_ent.aliases]
            )

        for s_id, t_id in itertools.product(s_aliases, t_aliases):
            if len(s_aliases[s_id].intersection(t_aliases[t_id])) > 0:
                alignment.append((s_id, t_id, 1.0))
                s_matched.add(s_id)
                t_matched.add(t_id)

        s_remaining = set([e.research_entity_id for e in s_kb.entities]).difference(s_matched)
        t_remaining = set([e.research_entity_id for e in t_kb.entities]).difference(t_matched)

        return alignment, s_remaining, t_remaining

    def _compute_global_similarities(self, local_scores, s_kb, t_kb):
        """
        Compute global similarities based on scores in local_scores
        :param local_scores:
        :param s_kb:
        :param t_kb:
        :return:
        """
        # iteratively calculate global similarity scores
        for i in range(0, constants.GLOBAL_SIMILARITY_ITERATIONS):
            global_scores = dict()
            for (s_ent_id, t_ent_id), score in local_scores.items():
                s_ent = s_kb.get_entity_by_research_entity_id(s_ent_id)
                t_ent = t_kb.get_entity_by_research_entity_id(t_ent_id)

                # generate regions around s_ent and t_ent not included s_ent and t_ent
                s_region = self._get_region_around_ent(s_ent, s_kb)
                t_region = self._get_region_around_ent(t_ent, t_kb)

                # sum regional contributions to similarity
                global_sum = 0.0
                distance_weights = 0.0
                for s_neighbor_id, t_neighbor_id in itertools.product(s_region, t_region):
                    if len(s_region[s_neighbor_id]) == len(t_region[t_neighbor_id]):
                        if (s_neighbor_id, t_neighbor_id) in local_scores:
                            d_weight = self._get_distance_weight(s_region[s_neighbor_id], t_region[t_neighbor_id])
                            distance_weights += d_weight
                            global_sum += d_weight * local_scores[(s_neighbor_id, t_neighbor_id)]
                global_score = global_sum / distance_weights
                global_scores[(s_ent_id, t_ent_id)] = global_score

            # set local scores to newly computed global scores
            local_scores = global_scores

        # keep all alignments about minimum score threshold
        temp_alignments = defaultdict(list)
        for (s_ent_id, t_ent_id), score in local_scores.items():
            if score >= constants.MIN_SCORE_THRESHOLD:
                temp_alignments[s_ent_id].append((t_ent_id, score))

        # select best match for each source entity
        global_matches = []
        for s_ent_id, matches in temp_alignments.items():
            if len(matches) > 0:
                m_sort = sorted(matches, key=lambda p: p[1], reverse=True)
                if m_sort[0][1] >= constants.MAX_SCORE_THRESHOLD:
                    global_matches.append((s_ent_id, m_sort[0][0], m_sort[0][1]))

        sys.stdout.write('Global matches: %i\n' % len(global_matches))
        return global_matches

    def _apply_model_align(self, model, s_kb, t_kb, cand_sel):
        """
        Align kbs with model
        :param model:
        :param source_kb:
        :param target_kb:
        :param cand_sel:
        :return:
        """
        alignment, s_ent_ids, t_ent_ids = self._align_string_equiv(s_kb, t_kb)
        sys.stdout.write("%i alignments with string equivalence\n" % len(alignment))

        feat_gen = SparseFeatureGenerator(cand_sel.s_token_to_idf,
                                          cand_sel.t_token_to_idf)

        sys.stdout.write("Making predictions...\n")
        local_scores = dict()
        s_ent_tqdm = tqdm.tqdm(s_ent_ids,
                               total=len(s_ent_ids))
        for s_ent_id in s_ent_tqdm:
            s_ent = s_kb.get_entity_by_research_entity_id(s_ent_id)
            for t_ent_id in cand_sel.select_candidates(
                    s_ent_id
            )[:constants.KEEP_TOP_K_CANDIDATES]:
                if t_ent_id in t_ent_ids:
                    t_ent = t_kb.get_entity_by_research_entity_id(t_ent_id)
                    features = [feat_gen.calculate_features(self._form_json_entity(s_ent, s_kb),
                                                            self._form_json_entity(t_ent, t_kb))]
                    score = model.predict_entity_pair(features)
                    if score[0][1] >= constants.MIN_SCORE_THRESHOLD:
                        local_scores[(s_ent_id, t_ent_id)] = score[0][1]

        global_matches = self._compute_global_similarities(local_scores, s_kb, t_kb)

        return alignment + global_matches


    def _align_lr(self, model_path, source_kb, target_kb, candidate_selector):
        """
        Align using logistic regression model
        :param source_kb:
        :param target_kb:
        :param candidate_selector:
        :return:
        """
        sys.stdout.write("Loading model...\n")
        model = OntoEmmaLRModel()
        model.load(model_path)
        return self._apply_model_align(model, source_kb, target_kb, candidate_selector)

    def _align_rf(self, model_path, source_kb, target_kb, candidate_selector):
        """
        Align using logistic regression model
        :param source_kb:
        :param target_kb:
        :param candidate_selector:
        :return:
        """
        sys.stdout.write("Loading model...\n")
        model = OntoEmmaRFModel()
        model.load(model_path)
        return self._apply_model_align(model, source_kb, target_kb, candidate_selector)

    def _align_nn(self, model_path, source_kb, target_kb, candidate_selector, cuda_device, batch_size=128):
        """
        Align using neural network model
        :param source_kb:
        :param target_kb:
        :param candidate_selector:
        :param cuda_device: GPU device number
        :return:
        """

        from emma.allennlp_classes.ontoemma_dataset_reader import OntologyMatchingDatasetReader
        from emma.allennlp_classes.ontoemma_model import OntoEmmaNN
        from emma.allennlp_classes.ontoemma_predictor import OntoEmmaPredictor

        alignment, s_ent_ids, t_ent_ids = self._align_string_equiv(source_kb, target_kb)
        sys.stdout.write("%i alignments with string equivalence\n" % len(alignment))

        sys.stdout.write("Adding synonyms to KBs from MeSH and DBpedia...\n")
        source_kb = self.add_synonyms(source_kb)
        target_kb = self.add_synonyms(target_kb)

        # Load similarity predictor
        if cuda_device > 0:
            with device(cuda_device):
                archive = load_archive(model_path, cuda_device=cuda_device)
        else:
            archive = load_archive(model_path, cuda_device=cuda_device)

        predictor = Predictor.from_archive(archive, 'ontoemma-predictor')

        sys.stdout.write("Making predictions...\n")
        s_ent_tqdm = tqdm.tqdm(s_ent_ids,
                               total=len(s_ent_ids))
        local_scores = dict()

        batch_json_data = []

        for s_ent_id in s_ent_tqdm:
            s_ent = source_kb.get_entity_by_research_entity_id(s_ent_id)
            for t_ent_id in candidate_selector.select_candidates(s_ent_id)[:constants.KEEP_TOP_K_CANDIDATES]:
                if t_ent_id not in t_ent_ids:
                    continue

                t_ent = target_kb.get_entity_by_research_entity_id(t_ent_id)

                json_data = {
                    'source_ent': self._form_json_entity(s_ent, source_kb),
                    'target_ent': self._form_json_entity(t_ent, target_kb),
                    'label': 0
                }
                batch_json_data.append(json_data)

                if len(batch_json_data) == batch_size:
                    results = predictor.predict_batch_json(batch_json_data, cuda_device)

                    for ent_data, output in zip(batch_json_data, results):
                        if output['score'][0] >= constants.MIN_SCORE_THRESHOLD:
                            local_scores[
                                (ent_data['source_ent']['research_entity_id'], ent_data['target_ent']['research_entity_id'])
                            ] = output['score'][0]

                    batch_json_data = []

        if batch_json_data:
            results = predictor.predict_batch_json(batch_json_data, cuda_device)
            for ent_data, output in zip(batch_json_data, results):
                if output['score'][0] >= constants.MIN_SCORE_THRESHOLD:
                    local_scores[
                        (ent_data['source_ent']['research_entity_id'], ent_data['target_ent']['research_entity_id'])
                    ] = output['score'][0]

        sys.stdout.write("Computing global similarities...\n")
        global_matches = self._compute_global_similarities(local_scores)

        return alignment + global_matches

    def align(self,
              model_type, model_path,
              s_kb_path, t_kb_path,
              gold_path, output_path,
              cuda_device=-1, missed_path=None):
        """
        Align two input ontologies
        :param model_type: type of model
        :param model_path: path to ontoemma model
        :param s_kb_path: path to source KB
        :param t_kb_path: path to target KB
        :param gold_path: path to gold alignment between source and target KBs
        :param output_path: path to write output alignment
        :param cuda_device: GPU device number
        :param missed_path: optional parameter for outputting missed alignments
        :return:
        """
        assert model_type in constants.IMPLEMENTED_MODEL_TYPES
        assert os.path.exists(model_path)
        assert s_kb_path is not None
        assert t_kb_path is not None

        alignment_scores = None

        sys.stdout.write("Loading KBs...\n")
        s_kb = self.load_kb(s_kb_path)
        t_kb = self.load_kb(t_kb_path)

        sys.stdout.write("Building candidate indices...\n")
        cand_sel = CandidateSelection(s_kb, t_kb)

        alignment = []
        if model_type == 'lr':
            alignment = self._align_lr(model_path, s_kb, t_kb, cand_sel)
        elif model_type == 'nn':
            alignment = self._align_nn(model_path, s_kb, t_kb, cand_sel, cuda_device)
        elif model_type == 'rf':
            alignment = self._align_rf(model_path, s_kb, t_kb, cand_sel)

        if missed_path is None and output_path is not None:
            missed_path = output_path + '.ontoemma.missed'

        if gold_path is not None and os.path.exists(gold_path):
            sys.stdout.write("Evaluating against gold standard...\n")
            alignment_scores = self.compare_alignment_to_gold(gold_path, alignment, s_kb, t_kb, missed_path)

        if output_path is not None:
            sys.stdout.write("Writing results to file...\n")
            self.write_alignment(output_path, alignment, s_kb_path, t_kb_path)

        return alignment_scores

    def compare_alignment_to_gold(self, gold_path, alignment, s_kb, t_kb, missed_file):
        """
        Make predictions on features and evaluate against gold
        :param gold_path: path to gold alignment file
        :param alignment: OntoEmma-produced alignment
        :param s_kb: source kb
        :param t_kb: target kb
        :param missed_file: file to write missed data
        :return:
        """
        gold_positives = set(
            [
                (s_ent, t_ent)
                for s_ent, t_ent, score in self.load_alignment(gold_path)
                if score is not None and score != '' and float(score) > 0.0
            ]
        )
        sys.stdout.write(
            'Positive alignments in gold standard: %i\n' % len(gold_positives)
        )

        alignment_positives = set(
            [(s_ent, t_ent) for s_ent, t_ent, score in alignment]
        )
        sys.stdout.write(
            'Positive alignments detected by OntoEmma: %i\n' %
            len(alignment_positives)
        )

        missed = gold_positives.difference(alignment_positives)

        if missed_file:
            dir_name, file_name = os.path.split(missed_file)
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                except OSError:
                    raise(
                        OSError,
                        "Missed file directory does not exist and OntoEmma cannot make it.\n"
                    )
            with open(missed_file, 'w') as outf:
                for s_ent, t_ent in missed:
                    try:
                        s_names = s_kb.get_entity_by_research_entity_id(
                            s_ent
                        ).aliases
                        t_names = t_kb.get_entity_by_research_entity_id(
                            t_ent
                        ).aliases
                        outf.write(
                            '%s\t%s\t%s\t%s\n' % (
                                s_ent, t_ent, ','.join(s_names),
                                ','.join(t_names)
                            )
                        )
                    except AttributeError:
                        outf.write('%s\t%s\t%s\t%s\n' % (s_ent, t_ent, '', ''))

        precision = 0.0
        recall = 0.0
        f1_score = 0.0

        if len(alignment_positives) > 0:
            precision = len(alignment_positives.intersection(gold_positives)
                            ) / len(alignment_positives)
            recall = len(alignment_positives.intersection(gold_positives)
                         ) / len(gold_positives)
            if precision + recall > 0.0:
                f1_score = (2 * precision * recall / (precision + recall))

        sys.stdout.write('Precision: %.2f\n' % precision)
        sys.stdout.write('Recall: %.2f\n' % recall)
        sys.stdout.write('F1-score: %.2f\n' % f1_score)

        return precision, recall, f1_score

    @staticmethod
    def _write_alignment_to_tsv(output_path, alignment):
        """
        Write matches to tsv file according to alignment file specifications.
        Format specified by https://docs.google.com/document/d/1VSeMrpnKlQLrJuh9ffkq7u7aWyQuIcUj4E8dUclReXM
        :param output_path: path specifying the output file
        :param alignment: alignment to write to file
        :return:
        """
        with open(output_path, 'w') as outf:
            for s_ent, t_ent, pred in sorted(
                alignment, key=lambda x: x[2], reverse=True
            ):
                outf.write(
                    "%s\t%s\t%s\t%s\n" % (s_ent, t_ent, pred, "OntoEmma")
                )
        return

    @staticmethod
    def _write_alignment_to_rdf(output_path, alignment, s_kb_path, t_kb_path):
        """
        Write matches to RDF file, specified by OAEI
        :param output_path: path specifying the output file
        :param alignment: alignment to write to file
        :param s_kb_path: path to source KB
        :param t_kb_path: path to target KB
        :return:
        """
        with open(output_path, 'w') as outf:
            outf.write("<?xml version='1.0' encoding='utf-8'?>\n")
            outf.write(
                "<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment'\n"
            )
            outf.write(
                "\t xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' \n"
            )
            outf.write("\t xmlns:xsd='http://www.w3.org/2001/XMLSchema#' \n")
            outf.write("\t alignmentSource='extracted_from_UMLS'>\n\n")
            outf.write("<Alignment>\n")
            outf.write("\t<xml>yes</xml>\n")
            outf.write("\t<level>0</level>\n")
            outf.write("\t<type>??</type>\n")
            outf.write("\t<onto1>" + s_kb_path + "</onto1>\n")
            outf.write("\t<onto2>" + t_kb_path + "</onto2>\n")
            outf.write("\t<uri1>" + s_kb_path + "</uri1>\n")
            outf.write("\t<uri2>" + t_kb_path + "</uri2>\n")

            for s_ent, t_ent, pred in sorted(
                alignment, key=lambda x: x[2], reverse=True
            ):
                outf.write("\t<map>\n")
                outf.write("\t\t<Cell>\n")
                outf.write("\t\t\t<entity1 rdf:resource=\"" + s_ent + "\"/>\n")
                outf.write("\t\t\t<entity2 rdf:resource=\"" + t_ent + "\"/>\n")
                outf.write(
                    "\t\t\t<measure rdf:datatype=\"http://www.w3.org/2001/XMLSchema#float\">"
                    + '{0:.2f}'.format(pred) + "</measure>\n"
                )
                outf.write("\t\t\t<relation>=</relation>\n")
                outf.write("\t\t</Cell>\n")
                outf.write("\t</map>\n\n")

            outf.write("</Alignment>\n")
            outf.write("</rdf:RDF>")
        return

    def write_alignment(self, output_path, alignment, s_kb_path, t_kb_path):
        """
        Write alignments to file
        :param output_path: path specifying the output file
        :param alignment: alignment to write to file
        :param s_kb_path: path to source KB
        :param t_kb_path: path to target KB
        :return:
        """
        dir_name, file_name = os.path.split(output_path)
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError:
                sys.stdout.write(
                    "WARNING: Output directory does not exist and OntoEmma cannot make it.\n"
                )
                sys.stdout.write(
                    "Output file will be written to the current directory.\n"
                )
                output_path = os.path.join(os.getcwd(), file_name)

        fname, fext = os.path.splitext(output_path)
        if fext == '.tsv':
            self._write_alignment_to_tsv(output_path, alignment)
        elif fext == '.rdf':
            self._write_alignment_to_rdf(
                output_path, alignment, s_kb_path, t_kb_path
            )
        else:
            raise NotImplementedError(
                "Unknown output file type. Cannot write alignment to file."
            )

        def eval_cs(self, s_kb_path, t_kb_path, gold_path, output_path, missed_path):
            """
            Evaluate candidate selection module
            :param s_kb_path: source kb path
            :param t_kb_path: target kb path
            :param gold_path: gold alignment file path
            :param output_path: output path for evaluation results
            :param missed_path: output path for missed alignments
            :return:
            """
            sys.stdout.write("Loading KBs...\n")
            s_kb = self.load_kb(s_kb_path)
            t_kb = self.load_kb(t_kb_path)

            sys.stdout.write("Loading gold alignment...\n")
            gold_alignment = self.load_alignment(gold_path)
            positive_alignments = [(i[0], i[1]) for i in gold_alignment]
            sys.stdout.write("\tNumber of gold alignments: %i\n" % len(positive_alignments))

            sys.stdout.write("Starting candidate selection...\n")
            cand_sel = CandidateSelection(s_kb, t_kb)
            cand_sel.EVAL_OUTPUT_FILE = output_path
            cand_sel.EVAL_MISSED_FILE = missed_path

            sys.stdout.write("Evaluating candidate selection...\n")
            cand_sel.eval(positive_alignments)
            return
import os
import sys
import csv
import time
import json
import itertools
import requests
from lxml import etree
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError

from allennlp.commands import main
from emma.OntoEmmaLRModel import OntoEmmaLRModel
from emma.kb.kb_utils_refactor import KnowledgeBase
from emma.kb.kb_load_refactor import KBLoader
from emma.CandidateSelection import CandidateSelection
from emma.FeatureGenerator import FeatureGenerator
from emma.paths import StandardFilePath
import emma.constants as constants


# class for training an ontology matcher and aligning input ontologies
class OntoEmma:
    def __init__(self):

        paths = StandardFilePath(base_dir='/net/nfs.corp/s2-research/scigraph/data/')
        self.kb_dir = paths.ontoemma_kb_dir

        self.kb_file_paths = dict()
        self.kb_pairs = set([])
        self._get_kb_fnames()

        sys.stdout.write("Configured variables:\n")
        sys.stdout.write("\tScore threshold: %.2f\n" % constants.SCORE_THRESHOLD)
        sys.stdout.write(
            "\tTop k candidates kept: %i\n" % constants.KEEP_TOP_K_CANDIDATES
        )

    def _get_kb_fnames(self):
        """
        Parse KB filenames from KB directory and generate KB pairs
        :return:
        """
        for kb_name in constants.TRAINING_KBS:
            self.kb_file_paths.setdefault(
                kb_name,
                os.path.join(self.kb_dir, 'kb-{}.json'.format(kb_name))
            )
        self.kb_pairs = itertools.combinations(constants.TRAINING_KBS, 2)
        return

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
            raise NotImplementedError(
                "Unknown input alignment file type. Cannot parse."
            )

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

        if model_type == "nn":
            sys.stdout.write("Training {} model...\n".format(constants.IMPLEMENTED_MODEL_TYPES[model_type]))

            from emma.allennlp_classes.ontoemma_dataset_reader import OntologyMatchingDatasetReader
            from emma.allennlp_classes.ontoemma_model import OntoEmmaNN

            # trim remaining arguments
            sys.argv = ['allennlp.run', 'train', config_file, '--serialization_dir', model_path]

            main(prog="python -m allennlp.run")

            sys.stdout.write("done.\n")
            sys.exit(0)

        # create model
        if model_type == "lr":
            model = OntoEmmaLRModel()
        else:
            raise (NotImplementedError, "Unknown model type")

        # read model config
        with open(config_file, 'r') as f:
            config = json.load(f)

        # parse parameters
        training_data_path = config['train_data_path']
        dev_data_path = config['validation_data_path']

        # load training data
        training_data = self.load_alignment(training_data_path)
        training_pairs = [(ent1, ent2) for ent1, ent2, _ in training_data]
        training_labels = [label for _, _, label in training_data]

        training_ordered_indices = []
        training_features = []

        # load development data
        dev_data = self.load_alignment(dev_data_path)
        dev_pairs = [(ent1, ent2) for ent1, ent2, _ in dev_data]
        dev_labels = [label for _, _, label in dev_data]

        dev_ordered_indices = []
        dev_features = []

        sys.stdout.write('Training data size: %i\n' % len(training_data))
        sys.stdout.write('Development data size: %i\n' % len(dev_data))

        # initialize KBs
        s_kb = KnowledgeBase()
        t_kb = KnowledgeBase()

        # iterate through kb pairs
        for s_kb_name, t_kb_name in self.kb_pairs:
            training_matches = [
                i for i, p in enumerate(training_pairs)
                if p[0].startswith(s_kb_name) and p[1].startswith(t_kb_name)
            ]
            dev_matches = [
                i for i, p in enumerate(dev_pairs)
                if p[0].startswith(s_kb_name) and p[1].startswith(t_kb_name)
            ]
            # load kbs if matches not empty
            if len(training_matches) > constants.MIN_TRAINING_SET_SIZE:
                training_ordered_indices += training_matches
                dev_ordered_indices += dev_matches
                sys.stdout.write(
                    "\tCalculating features for pairs between %s and %s\n" %
                    (s_kb_name, t_kb_name)
                )

                # load KBs if necessary
                if s_kb.name != s_kb_name:
                    s_kb = s_kb.load(self.kb_file_paths[s_kb_name])
                if t_kb.name != t_kb_name:
                    t_kb = t_kb.load(self.kb_file_paths[t_kb_name])

                # initialize feature generator with pair of KBs
                feat_gen = FeatureGenerator(s_kb, t_kb)

                # calculate features for training pairs
                for i in training_matches:
                    s_ent_id, t_ent_id = training_pairs[i]
                    training_features.append(
                        feat_gen.calculate_features(s_ent_id, t_ent_id)
                    )

                # calculate features for development pairs
                for i in dev_matches:
                    s_ent_id, t_ent_id = dev_pairs[i]
                    dev_features.append(
                        feat_gen.calculate_features(s_ent_id, t_ent_id)
                    )

        sys.stdout.write("Training...\n")

        training_labels = [
            training_labels[i] for i in training_ordered_indices
        ]

        model.train(training_features, training_labels)

        training_accuracy = model.score_accuracy(
            training_features, training_labels
        )
        sys.stdout.write(
            "Accuracy on training data set: %.2f\n" % training_accuracy
        )

        dev_labels = [dev_labels[i] for i in dev_ordered_indices]

        dev_accuracy = model.score_accuracy(dev_features, dev_labels)
        sys.stdout.write(
            "Accuracy on development data set: %.2f\n" % dev_accuracy
        )

        model.save(model_path)

        return

    def _align_lr(self, model_path, source_kb, target_kb, candidate_selector, feature_generator):
        """
        Align using logistic regression model
        :param source_kb:
        :param target_kb:
        :param candidate_selector:
        :param feature_generator:
        :return:
        """
        # Initialize
        alignment = []

        sys.stdout.write("Loading model...\n")
        model = OntoEmmaLRModel()
        model.load(model_path)

        sys.stdout.write("Making predictions...\n")

        for index, s_ent in enumerate(source_kb.entities):
            # show progress to user so that they feel good.
            if index == 1:
                sys.stdout.write('\n')
            if index % 10 == 1:
                sys.stdout.write('\rpredicted alignments for {} out of {} source entities.'.format(
                    index, len(source_kb.entities)))
            s_ent_id = s_ent.research_entity_id
            for t_ent_id in candidate_selector.select_candidates(
                    s_ent_id
            )[:constants.KEEP_TOP_K_CANDIDATES]:
                features = [feature_generator.calculate_features(s_ent_id, t_ent_id)]
                score = model.predict_entity_pair(features)
                if score[0][1] >= constants.SCORE_THRESHOLD:
                    alignment.append((s_ent_id, t_ent_id, score[0][1]))

        return alignment

    def _align_nn(self, model_path, source_kb, target_kb, candidate_selector):
        """
        Align using neural network model
        :param source_kb:
        :param target_kb:
        :param candidate_selector:
        :return:
        """
        raise(NotImplementedError, "Yet to connect AllenNLP model to OntoEmma")
        # TODO: connect to allennlp ontoemma model
        # write candidates to file
        # call allennlp ontoemma on output candidate file to generate predictions
        # extract positive alignments

    def align(self, model_type, model_path, s_kb_path, t_kb_path, gold_path, output_path, missed_path=None):
        """
        Align two input ontologies
        :param model_type: type of model
        :param model_path: path to ontoemma model
        :param s_kb_path: path to source KB
        :param t_kb_path: path to target KB
        :param gold_path: path to gold alignment between source and target KBs
        :param output_path: path to write output alignment
        :param missed_path: optional parameter for outputting missed alignments
        :return:
        """
        alignment_scores = None

        sys.stdout.write("Loading KBs...\n")
        s_kb = self.load_kb(s_kb_path)
        t_kb = self.load_kb(t_kb_path)

        sys.stdout.write("Building candidate indices...\n")
        cand_sel = CandidateSelection(s_kb, t_kb)

        if model_type == 'lr':
            feat_gen = FeatureGenerator(s_kb, t_kb)
            alignment = self._align_lr(model_path, s_kb, t_kb, cand_sel, feat_gen)
        elif model_type == 'nn':
            alignment = self._align_nn(model_path, s_kb, t_kb, cand_sel)
        else:
            raise(NotImplementedError, "Model type has not been implemented.")

        if missed_path is None and output_path is not None:
            missed_path = output_path + '.ontoemma.missed'

        if gold_path is not None and os.path.exists(gold_path):
            sys.stdout.write("Evaluating against gold standard...\n")
            alignment_scores = self.evaluate_alignment(gold_path, alignment, s_kb, t_kb, missed_path)

        if output_path is not None:
            sys.stdout.write("Writing results to file...\n")
            self.write_alignment(output_path, alignment, s_kb_path, t_kb_path)

        return alignment_scores

    def evaluate_alignment(self, gold_path, alignment, s_kb, t_kb, missed_file):
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

        precision = None
        recall = None
        f1_score = None

        if len(alignment_positives) > 0:
            precision = len(alignment_positives.intersection(gold_positives)
                            ) / len(alignment_positives)
            recall = len(alignment_positives.intersection(gold_positives)
                         ) / len(gold_positives)
            if precision + recall > 0.0:
                f1_score = (2 * precision * recall / (precision + recall))

        if precision:
            sys.stdout.write('Precision: %.2f\n' % precision)
        if recall:
            sys.stdout.write('Recall: %.2f\n' % recall)
        if f1_score:
            sys.stdout.write('F1-score: %.2f\n' % f1_score)

        return precision, recall, f1_score

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


if __name__ == '__main__':
    t = time.time()

    matcher = OntoEmma()
    args = sys.argv[1:]
    mode = args[0]

    if mode == 'train':
        # training mode, training data specified
        model_type = args[1]
        model_path = args[2]
        config_file = args[3]
        matcher.train(
           model_type, model_path,
           config_file
        )
    elif mode == 'align':
        # alignment mode, source kb, target kb, gold alignment, and output file specified
        model_type = args[1]
        model_path = args[2]
        source_kb_path = args[3]
        target_kb_path = args[4]
        gold_file_path = None
        output_file_path = None
        if len(args) > 5:
            gold_file_path = args[5]
        if len(args) > 6:
            output_file_path = args[6]
        matcher.align(
            model_type, model_path,
            source_kb_path, target_kb_path,
            gold_file_path, output_file_path
        )
    elif mode == 'test':
        # evaluate candidate selection
        test_type = args[1]
        source_kb_path = args[2]
        target_kb_path = args[3]

        if test_type == "eval_cs":
            gold_file_path = args[4]
            output_file_path = None
            missed_data_file = None
            if len(args) > 5:
                output_file_path = args[5]
            if len(args) > 6:
                missed_data_file = args[6]
            matcher.eval_cs(
                source_kb_path, target_kb_path, gold_file_path, output_file_path, missed_data_file
            )
        else:
            sys.stdout.write("Unknown test type, exiting...\n")
    else:
        sys.stdout.write('Unknown mode, exiting...\n')
        sys.exit(1)

    sys.stdout.write("Time taken: %f\n" % (time.time() - t))
    sys.exit(0)

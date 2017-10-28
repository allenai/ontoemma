import sys
import xml.etree.ElementTree as ET

# parse matches in OAEI formatted rdf match file
# returns list of entities matched and their match scores
def get_mappings(mfile):
    mappings = []

    namespaces = {'alignment': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'}
    resource = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'

    tree = ET.parse(mfile)
    root = tree.getroot()
    maps = root.find('alignment:Alignment',namespaces).findall('alignment:map',namespaces)

    for m in maps:
        cell = m.find('alignment:Cell',namespaces)
        ent1 = cell.find('alignment:entity1',namespaces).get(resource)
        ent2 = cell.find('alignment:entity2',namespaces).get(resource)
        meas = cell.find('alignment:measure',namespaces).text # match score [0,1]
        mappings.append((ent1,ent2,meas))

    return set(mappings)

# class for comparing two ontology alignments for evaluation purposes
# evaluates similarity between alignment and baseline
# calculates precision, recall, f-score, false positives and false negatives

class EvaluateAlignments:

    # initialize evaluator
    # eval_alignment: list of pairs of aligned entities
    # alignment_file: file location from which alignments can be read
    # baseline_alignment_file: file location of baseline alignment
    # ont1: source ontology as rdflib.Graph()
    # ont2: target ontology as rdflib.Graph()
    # if eval_alignment is None, alignments will be read from alignment_file
    def __init__(self,eval_alignment,alignment_file,baseline_alignment_file,ont1,ont2):

        if eval_alignment is None:
            self.eval_alignment = get_mappings(alignment_file)
        else:
            self.eval_alignment = eval_alignment
        self.base_alignment = get_mappings(baseline_alignment_file)

        self.source_ontology = ont1
        self.target_ontology = ont2

        # initialize outputs
        self.FP = []
        self.FN = []
        self.precision = None
        self.recall = None
        self.fscore = None

    # determine output values
    def compute_mapping_stats(self):

        TP = self.eval_alignment.intersection(self.base_alignment)
        self.FP = self.eval_alignment.difference(self.base_alignment)
        self.FN = self.base_alignment.difference(self.eval_alignment)

        self.precision = len(TP)/len(self.eval_alignment)
        self.recall = len(TP)/len(self.base_alignment)
        self.fscore = 2*(self.precision*self.recall)/(self.precision+self.recall)

    # print mapping stats
    def print_mapping_stats(self):
        sys.stdout.write("Precision: %.2f\n" %self.precision)
        sys.stdout.write("Recall: %.2f\n" %self.recall)
        sys.stdout.write("F-score: %.2f\n" %self.fscore)
        sys.stdout.write("Total system mappings: %i\n" %len(self.eval_alignment))
        sys.stdout.write("Total baseline mappings: %i\n" %len(self.base_alignment))
        sys.stdout.write("False positives: %i\n" %len(self.FP))
        sys.stdout.write("False negatives: %i\n" %len(self.FN))

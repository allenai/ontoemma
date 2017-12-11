import os
import sys
import tqdm
import json
import pickle
import datetime
import jsonlines
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from typing import List, Dict

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from emma.OntoEmma import OntoEmma
import emma.utils.string_utils as string_utils
from emma.paths import StandardFilePath
from emma.kb.kb_utils_refactor import KnowledgeBase


# class for enriching entity with synonyms
class SynonymEnricher:
    def __init__(self):
        """
        Intialize
        """
        # Load synonym files
        paths = StandardFilePath()
        mesh_syn_file = os.path.join(paths.ontoemma_synonym_dir, 'mesh_synonyms.pickle')
        dbpedia_syn_file = os.path.join(paths.ontoemma_synonym_dir, 'dbpedia_synonyms.pickle')

        sys.stdout.write("Loading synonym files..\n")
        self.mesh_synonyms = pickle.load(open(mesh_syn_file, 'rb'))
        self.dbpedia_synonyms = pickle.load(open(dbpedia_syn_file, 'rb'))

    def add_synonyms_to_entity(self, aliases: List):
        """
        Return synonyms of entity
        :param aliases: entity aliases
        :return:
        """
        # normalize aliases
        norm_aliases = [string_utils.normalize_string(a) for a in aliases]

        # intialize synonym lists
        mesh_syns = []
        dbpedia_syns = []

        # get synonyms from synonym dicts
        for a in norm_aliases:
            mesh_syns += self.mesh_synonyms[a]
            dbpedia_syns += self.dbpedia_synonyms[a]

        return list(set(mesh_syns)), list(set(dbpedia_syns))


# class for querying entities in Wikipedia to extract context
class WikipediaEnricher:
    def __init__(self):
        """
        Intialize queryer
        :param kb_name: name of KB
        :param kb_path: path to KB file (any format readable by OntoEmma)
        :param out_path: output path of enriched KB
        :param restrict_to_ids: set restricting KB ids to query
        """
        # Initialize wikipedia queryer
        wikipedia.set_lang('en')
        wikipedia.set_rate_limiting(True, min_wait=datetime.timedelta(0, 0, 2000))
        self.wiki_dict = dict()

        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.STOP = set(stopwords.words('english'))

    @staticmethod
    def get_summary(query_text: str):
        """
        Get summary from hit article
        :param query_text:
        :return:
        """
        try:
            return wikipedia.summary(query_text, sentences=1)
        except DisambiguationError:
            return ''
        except PageError:
            return ''
        except Exception:
            print('Unknown exception!')
            return ''

    def search_term(self, query_text: str):
        """
        Search query text in wikipedia
        :param query_text:
        :return:
        """
        if query_text not in self.wiki_dict:
            try:
                self.wiki_dict[query_text] = wikipedia.search(query_text, suggestion=True)
            except Exception:
                self.wiki_dict[query_text] = ([], None)
        return self.wiki_dict[query_text]

    def add_definition_to_entity(self, ent_name: str):
        """
        Add synonyms to entity
        :param ent:
        :return:
        """
        wiki_ents = []
        definition = ""

        try:
            # normalize name
            norm_name = string_utils.remove_stop(ent_name, self.tokenizer, self.STOP)

            # get definition from wikipedia
            definition = self.get_summary(norm_name)

            # get suggested articles from wikipedia
            articles, suggestion = self.search_term(norm_name)
            wiki_ents = articles

            # get definition from suggested articles if no definition retrieved earlier
            if len(definition) <= 5 and suggestion:
                definition = self.get_summary(suggestion)
        except Exception:
            return wiki_ents, definition

        return wiki_ents, definition


# class for enriching data from KB or training data
class DataEnricher:
    def __init__(self, data_path: str, out_path: str):
        """
        Initialize class for enriching data. The data_path can point to a particular knowledgebase
        to be enriched, or it can point to a training data file.
        :param data_path:
        :param out_path:
        :param restrict_path: path to list of entity ids to enrich
        """
        self.data_path = data_path
        self.out_path = out_path

        # clear out file
        if os.path.exists(self.out_path):
            open(self.out_path, 'w').close()

        # initialize enricher classes
        self.syn_enricher = SynonymEnricher()
        self.wiki_enricher = WikipediaEnricher()

    def query_all_kb(self, kb: KnowledgeBase):
        """
        Iterate through KB entities, query synonyms and definition, write to file.
        :param kb:
        :return:
        """
        for ent in tqdm.tqdm(kb.entities, total=len(kb.entities)):
            mesh_syn, dbp_syn = self.syn_enricher.add_synonyms_to_entity(ent.aliases)
            wiki_ents, definition = self.wiki_enricher.add_definition_to_entity(ent.canonical_name)
            ent.additional_details['mesh_synonyms'] = mesh_syn
            ent.additional_details['dbpedia_synonyms'] = dbp_syn
            ent.additional_details['wiki_entities'] = wiki_ents
            if len(ent.definition) < 5:
                ent.definition = definition

        kb.dump(kb, self.out_path)
        return

    def update_json_ent(self, ent_in_json: Dict):
        """
        Update json entity representation with synonyms and wikipedia definition
        :param ent_in_json:
        :return:
        """
        mesh_syn, dbp_syn = self.syn_enricher.add_synonyms_to_entity(ent_in_json['aliases'])
        wiki_ents, definition = self.wiki_enricher.add_definition_to_entity(ent_in_json['canonical_name'])

        ent_in_json['mesh_synonyms'] = mesh_syn
        ent_in_json['dbpedia_synonyms'] = dbp_syn
        ent_in_json['wiki_entities'] = wiki_ents
        if len(ent_in_json['definition']) < 5:
            ent_in_json['definition'] = definition

        return ent_in_json

    def query_all_training_data(self):
        """
        Iterate through training data, query and write to file
        :return:
        """
        # open data file and read lines
        with open(self.data_path, 'r') as f:
            for line in tqdm.tqdm(f):
                try:
                    dataline = json.loads(line)
                    s_ent = dataline['source_ent']
                    t_ent = dataline['target_ent']
                    label = dataline['label']

                    s_ent_new = self.update_json_ent(s_ent)
                    t_ent_new = self.update_json_ent(t_ent)

                    line = {
                        'source_ent': s_ent_new,
                        'target_ent': t_ent_new,
                        'label': label
                    }
                    with jsonlines.Writer(open(self.out_path, 'a')) as writer:
                        writer.write(line)
                except Exception:
                    continue
        return

    def query_all(self):
        """
        Query all entities in input data file
        :return:
        """
        try:
            ontoemma = OntoEmma()
            kb = ontoemma.load_kb(self.data_path)
            self.query_all_kb(kb)
        except Exception:
            try:
                self.query_all_training_data()
            except Exception:
                raise NotImplementedError("Unknown file type, cannot enrich...")

if __name__ == '__main__':
    data_file_path = sys.argv[1]
    out_file_path = sys.argv[2]
    queryer = DataEnricher(data_path=data_file_path, out_path=out_file_path)
    queryer.query_all()






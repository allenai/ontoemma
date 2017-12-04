import os
import sys
import json
import string
import logging
import tqdm
import rdflib

from collections import defaultdict
from lxml import etree

from emma.utils import file_util
from emma.utils.string_utils import canonicalize
from emma.utils.common import global_tokenizer


# a lightweight class to represent an entity with a unified schema.
class KBEntity(object):
    def __init__(
        self,
        research_entity_id=None,
        canonical_name=None,
        aliases=[],
        definition="<s>",
        source_urls=[],
        category=None
    ):
        self.research_entity_id = research_entity_id
        self.canonical_name = canonical_name
        self.aliases = aliases
        self.definition = definition
        self.source_urls = source_urls
        self.category = category
        # relations is a list of KBRelation ids.
        self.relation_ids = []
        self.other_contexts = []
        self.additional_details = defaultdict(list)

        # Fields containing tokenized text used for training
        self.tokenized_definition = None
        self.tokenized_canonical_name = None
        self.tokenized_aliases = None

    def __repr__(self):
        return json.dumps(
            {
                'research_entity_id': self.research_entity_id,
                'canonical_name': self.canonical_name
            }
        )

    def __eq__(self, ent):
        if self.research_entity_id == ent.research_entity_id \
                and self.canonical_name == ent.canonical_name \
                and set(self.aliases) == set(ent.aliases) \
                and self.source_urls == ent.source_urls \
                and self.category == ent.category:
            return True
        else:
            return False

    @property
    def raw_ids(self):
        return self.research_entity_id.split('|')

    @property
    def raw_id(self):
        return self.research_entity_id

    @staticmethod
    def form_dict(**entries):
        entity = KBEntity()
        entity.__dict__.update(entries)
        entity.set_source_url()
        return entity

    def entity_names(self):
        return set([canonicalize(n) for n in self.aliases])

    def tokenize_properties(self):
        if self.tokenized_definition is None:
            self.tokenized_definition = global_tokenizer(
                self.definition, restrict_by_pos=True, lowercase=True
            )
        if self.tokenized_canonical_name is None:
            self.tokenized_canonical_name = global_tokenizer(
                self.canonical_name, restrict_by_pos=True, lowercase=True
            )
        if self.tokenized_aliases is None:
            self.tokenized_aliases = [
                global_tokenizer(alias, restrict_by_pos=True, lowercase=True)
                for alias in self.aliases
            ]

    def set_source_url(self):
        # Add a default value for source_urls based on
        # the research_entity_id prefix.
        if 'source_urls' not in self.__dict__ or not self.source_urls:
            if self.research_entity_id.startswith('UMLS'):
                self.source_urls = [
                    'https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/'
                ]
            elif self.research_entity_id.startswith('craftv2'):
                self.source_urls = [
                    'http://bionlp-corpora.sourceforge.net/CRAFT/'
                ]
            elif self.research_entity_id.startswith('dbpedia'):
                pass
            else:
                pass


# a lightweight class to represent a relation with a unified schema.
class KBRelation(object):
    def __init__(self, relation_type=None, entity_ids=None, symmetric=None, labels=None):
        self.relation_type = relation_type
        self.entity_ids = entity_ids
        self.labels = set() if labels is None else labels
        # symmetric is a boolean that determines whether the relation is directional.
        self.symmetric = symmetric

    def __eq__(self, rel):
        if self.relation_type == rel.relation_type \
                and self.entity_ids == rel.entity_ids \
                and self.symmetric == rel.symmetric:
            return True
        else:
            return False

    def __repr__(self):
        return json.dumps(
            {
                'entity_ids': self.entity_ids,
                'relation_type': self.relation_type
            }
        )

    @staticmethod
    def form_dict(entries):
        rel = KBRelation(
            relation_type=entries['relation_type'],
            entity_ids=entries['entity_ids'],
            symmetric=entries['symmetric'],
            labels=set(entries['labels']) if 'labels' in entries and entries['labels'] else None
        )
        return rel


# a class that represents a knowledgebase with a unified schema.
class KnowledgeBase(object):
    def __init__(self):
        self.name = ""
        self.entities = []
        self.relations = []
        self.research_entity_id_to_entity_index = dict()
        self.raw_id_to_entity_index = dict()
        self.canonical_name_to_entity_index = defaultdict(set)
        self.entity_ids_to_relation_index = defaultdict(set)
        self.null_entity = None

    def __eq__(self, other_kb):
        if self.name == other_kb.name and \
                self.entities == other_kb.entities and \
                self.relations == other_kb.relations:
            return True
        else:
            return False

    def generate_indices(self):
        """
        Generates lookup dicts for entities
        :return:
        """
        self.research_entity_id_to_entity_index = dict()
        self.raw_id_to_entity_index = dict()
        self.canonical_name_to_entity_index = defaultdict(set)
        for ent_index, entity in enumerate(self.entities):
            self.research_entity_id_to_entity_index[entity.research_entity_id
                                                   ] = ent_index
            for raw_id in entity.raw_ids:
                self.raw_id_to_entity_index[raw_id] = ent_index
            self.canonical_name_to_entity_index[entity.canonical_name
                                               ].add(ent_index)
        for rel_index, relation in enumerate(self.relations):
            self.entity_ids_to_relation_index[tuple(relation.entity_ids)
                                             ].add(rel_index)
        return

    def add_null_entity(self):
        if self.null_entity is None:
            null_entity = KBEntity(
                research_entity_id='{}:{}'.format(self.name, 'NULL'),
                canonical_name="NULL Entity",
                aliases=[],
                definition="This is a null entity."
            )
            null_entity.source_urls = ['']
            self.add_entity(null_entity)
            self.null_entity = null_entity

    def validate_entity(self, ent: KBEntity):
        """
        Check if input entity is valid
        :param ent:
        :return: bool
        """
        if ent.canonical_name is None \
                or ent.canonical_name == "" \
                or ent.research_entity_id is None \
                or ent.research_entity_id == "":
            return False
        else:
            return True

    def validate_relation(self, rel: KBRelation):
        """
        Check if input relation is valid
        :param rel:
        :return: bool
        """
        if rel.relation_type is None \
                or rel.relation_type == "" \
                or rel.entity_ids[0] is None \
                or rel.entity_ids[1] is None:
            return False
        else:
            return True

    def add_entity(self, new_entity: KBEntity):
        """
        Verify and add entity to the KB
        :param new_entity: A new entity object to be added to the KB
        :return:
        """
        if self.validate_entity(new_entity):
            self.entities.append(new_entity)
            ent_index = len(self.entities) - 1
            self.research_entity_id_to_entity_index[
                new_entity.research_entity_id
            ] = ent_index
            for raw_id in new_entity.raw_ids:
                self.raw_id_to_entity_index[raw_id] = ent_index
            self.canonical_name_to_entity_index[new_entity.canonical_name
                                               ].add(ent_index)
        else:
            raise ValueError('Entity failed validation: %s' % new_entity)
        return

    def add_relation(self, new_relation: KBRelation):
        """
        Verify and add relation to KB
        :param new_relation:
        :return:
        """
        if self.validate_relation(new_relation):
            self.relations.append(new_relation)
            rel_index = len(self.relations) - 1
            self.entity_ids_to_relation_index[tuple(new_relation.entity_ids)
                                             ].add(rel_index)
        else:
            raise ValueError('Relation failed validation: %s' % new_relation)
        return

    def merge_entities(self, research_entity_id: str, ent_to_merge: KBEntity):
        """
        Merge ent_to_merge with entity at ent1_index, which is already in KB
        :param research_entity_id: research_entity_id of entity in KB
        :param ent_to_merge: ent to merge with entity represented by ent_id
        :param ent_relations: dict of relations associated with ent_to_merge; key is relation id, value is relation
        :return:
        """
        # check if original entity exists
        if self.get_entity_by_research_entity_id(research_entity_id):
            # check if entity to merge is valid
            if self.validate_entity(ent_to_merge):
                # get index of original entity
                ent_index = self.research_entity_id_to_entity_index[
                    research_entity_id
                ]

                # form new research entity id from all raw_ids
                all_raw_ids = self.entities[ent_index
                                           ].raw_ids + ent_to_merge.raw_ids
                new_research_entity_id = '|'.join(
                    sorted(list(set(all_raw_ids)))
                )

                # merge all info from ent_to_merge into original entity
                self.entities[ent_index
                             ].research_entity_id = new_research_entity_id
                self.entities[ent_index].aliases.extend(ent_to_merge.aliases)
                if len(ent_to_merge.definition) > 0 and self.entities[
                    ent_index
                ].definition != ent_to_merge.definition:
                    self.entities[ent_index
                                 ].definition += ' ' + ent_to_merge.definition
                self.entities[ent_index
                             ].source_urls.extend(ent_to_merge.source_urls)
                self.entities[ent_index].other_contexts.extend(
                    ent_to_merge.other_contexts
                )

                # add new relation ids to entity
                self.entities[ent_index
                             ].relation_ids.extend(ent_to_merge.relation_ids)

                # update relation entity_ids
                for rel_id in self.entities[ent_index].relation_ids:
                    self.relations[rel_id].entity_ids[0
                                                     ] = new_research_entity_id

                # merge additional_details dictionaries
                for key, val in ent_to_merge.additional_details:
                    self.entities[ent_index].additional_details[key
                                                               ].extend(val)

                # recompute tokenized info
                self.entities[ent_index].tokenize_properties()

                # delete former research_entity_id and add new entry to lookup dict
                del self.research_entity_id_to_entity_index[research_entity_id]
                self.research_entity_id_to_entity_index[new_research_entity_id
                                                       ] = ent_index

                # add new raw_ids to lookup dict
                for raw_id in ent_to_merge.raw_ids:
                    self.raw_id_to_entity_index[raw_id] = ent_index
            else:
                raise ValueError(
                    'Entity to merge failed validation: %s' % ent_to_merge
                )
        else:
            raise ValueError(
                'Research entity id not found in KB: %s' % research_entity_id
            )
        return

    def merge_relations(self, rel1: KBRelation, rel2: KBRelation):
        """
        Merge rel2 with rel1, which is already in KB
        :param rel1:
        :param rel2:
        :return:
        """
        raise NotImplementedError(
            'The merge relations method has not yet been implemented.'
        )

    def get_entity_by_research_entity_id(self, research_entity_id):
        """
        Returns entity associated with input research_entity_id, or None if none found
        :param research_entity_id:
        :return:
        """
        if research_entity_id in self.research_entity_id_to_entity_index:
            return self.entities[
                self.research_entity_id_to_entity_index[research_entity_id]
            ]
        else:
            return None

    def get_entity_by_raw_id(self, raw_id):
        """
        Returns entity associated with input raw_id, or None if none found
        :param raw_id:
        :return:
        """
        if raw_id in self.raw_id_to_entity_index:
            return self.entities[self.raw_id_to_entity_index[raw_id]]
        else:
            return None

    def get_entity_by_canonical_name(self, canonical_name):
        """
        Returns list of entities associated with input canonical_name, or empty list if none found
        :param canonical_name:
        :return:
        """
        return [
            self.entities[ind]
            for ind in self.canonical_name_to_entity_index[canonical_name]
        ]

    def get_entity_index(self, ent_id):
        """
        get index in entity list
        :param ent:
        :return:
        """
        return self.entities.index(self.get_entity_by_research_entity_id(ent_id))

    def get_relation_by_research_entity_ids_and_type(
        self, research_entity_ids, relation_type
    ):
        """
        Returns relation matching research_entity_ids and relation_type or None if none found
        :param research_entity_ids: tuple of two research_entity_id
        :param relation_type:
        :return:
        """
        rel_matches = self.entity_ids_to_relation_index[tuple(
            research_entity_ids
        )]
        for r_ind in rel_matches:
            if self.relations[r_ind].relation_type == relation_type:
                return self.relations[r_ind]
        return None

    def get_relations_of_entity(self, ent: KBEntity):
        """
        Returns list of relations represented under relation_ids of input entity
        :param ent:
        :return:
        """
        return [self.relations[ind] for ind in ent.relation_ids]

    @staticmethod
    def _json_dump(kb, filename):
        """
        Dumps unified knowledgebase into a zipped json file
        :param filename: file to write to
        :return:
        """
        with file_util.open(filename, mode='wb') as outfile:
            kb_dict = {
                'name':
                    kb.name,
                'entities':
                    [
                        {
                            'research_entity_id': entity.research_entity_id,
                            'canonical_name': entity.canonical_name,
                            'aliases': entity.aliases,
                            'definition': entity.definition,
                            'source_urls': entity.source_urls,
                            'category': entity.category,
                            'relation_ids': entity.relation_ids,
                            'other_contexts': entity.other_contexts,
                            'additional_details': entity.additional_details
                        } for entity in kb.entities
                    ],
                'relations':
                    [
                        {
                            'relation_id': rel_id,
                            'relation_type': relation.relation_type,
                            'entity_ids': relation.entity_ids,
                            'symmetric': relation.symmetric,
                            'labels': list(relation.labels) if relation.labels else None,
                        } for rel_id, relation in enumerate(kb.relations)
                    ]
            }
            outfile.write(json.dumps(kb_dict).encode())

    @staticmethod
    def _json_load(filename):
        """
        Loads unified knowledgebase from json file
        :param filename:
        :return: kb: KnowledgeBase
        """
        kb = KnowledgeBase()
        kb.filename = filename
        with open(filename, mode='r') as infile:
            unified_kb = json.loads(infile.read())
            kb.name = unified_kb['name']

            # load entities
            for e in unified_kb['entities']:
                ent = KBEntity.form_dict(**e)
                ent.tokenize_properties()
                ent.set_source_url()
                kb.add_entity(ent)

            # load relations in order
            relations = [None] * len(unified_kb['relations'])
            for r in unified_kb['relations']:
                rel_id = r['relation_id']
                del r['relation_id']
                rel = KBRelation.form_dict(r)
                relations[rel_id] = rel

            kb.relations = relations

        kb.generate_indices()
        return kb

    @staticmethod
    def _pickle_dump(kb, filename):
        """
        Dumps knowledgebase to pickle file
        :param kb:
        :param filename:
        :return:
        """
        logging.info(
            'Tokenizing entity fields before writing pickle file. Also assign source URL for entities'
        )

        logging.info("Adding a special NULL Entity")
        kb.add_null_entity()

        for entity in tqdm.tqdm(kb.entities):
            entity.tokenize_properties()
            try:
                entity.set_source_url()
            except RuntimeError:
                message = 'Each entity in the KB must have a non-empty source_urls field. Some entities in {} do not, e.g., {}'.format(
                    filename, entity.research_entity_id
                )
                logging.error(message)

        file_util.write_pickle(filename, kb)

    @staticmethod
    def _pickle_load(filename):
        """
        Loads knowledgebase from pickle file
        :param filename:
        :return:
        """
        kb = file_util.read_pickle(filename)
        # Set the filename field if not already set.
        kb.filename = kb.filename if 'filename' in kb.__dict__ and kb.filename else filename
        return kb

    def load(self, filename):
        """
        Loads knowledgebase from file
        :param filename:
        :return:
        """
        fname, fext = os.path.splitext(filename)
        if fext == '.json':
            return self._json_load(filename)
        elif fext == '.pickle':
            return self._pickle_load(filename)
        else:
            raise NameError('Unknown file type')

    def dump(self, kb, filename):
        """
        Dumps knowledgebase to file
        :param kb:
        :param filename:
        :return:
        """
        fname, fext = os.path.splitext(filename)
        if fext == '.json':
            return self._json_dump(kb, filename)
        elif fext == '.pickle':
            return self._pickle_dump(kb, filename)
        else:
            print('Saving to json, unknown file type...')
            return self._json_dump(kb, filename)

import logging
import os
from lxml import etree

import rdflib
from base import file_util
import re
from scigraph.kb.kb_utils_refactor import KBEntity, KBRelation, KnowledgeBase


# class for loading knowledgebase resources
class KBLoader(object):

    # s3://ai2-s2-research/scigraph/data/raw_kbs/craft_v2/CHEBI.obo
    SEQUENCE_ONTOLOGY = 'craftv2_sequence_ontology'
    # s3://ai2-s2-research/scigraph/data/raw_kbs/craft_v2/CL.obo
    NCBI_TAXONOMY = 'craftv2_ncbi_taxonomy'
    # s3://ai2-s2-research/scigraph/data/raw_kbs/craft_v2/GO.obo
    CHEBI_TAXONOMY = 'craftv2_chebi_taxonomy'
    # s3://ai2-s2-research/scigraph/data/raw_kbs/craft_v2/NCBITaxon.obo
    GO_TAXONOMY = 'craftv2_go_taxonomy'
    # s3://ai2-s2-research/scigraph/data/raw_kbs/craft_v2/PR.obo
    PR_TAXONOMY = 'craftv2_pr_taxonomy'
    # s3://ai2-s2-research/scigraph/data/raw_kbs/craft_v2/CL.obo
    CL_TAXONOMY = 'craftv2_cl_taxonomy'
    # An unknown source of OBO taxonomy. Usually used for test
    UNK_OBO_TAXONOMY = 'craftv2_UNK_taxonomy'

    # s3://ai2-s2-research/scigraph/data/raw_kbs/mesh2017/d2017.bin
    MESH_TAXONOMY = 'mesh_taxonomy'
    # s3://ai2-s2-research/scigraph/data/raw_kbs/dbpedia201510_en/long_abstracts_en.ttl
    DBPEDIA = 'dbpedia201510_en'
    # Merged KnowledgeBase including entities from all known KBs
    MERGED = "CANONICAL"

    # set of asymmetric relations from OBO
    OBO_ASYM_RELATION_SET = {
        'part_of', 'derives_from', 'has_quality', 'has_origin', 'sequence_of',
        'has_part', 'non_functional_homolog_of', 'position_of',
        'associated_with', 'adjacent_to', 'member_of', 'variant_of',
        'is_part_of', 'is_conjugate_base_of', 'is_conjugate_acid_of',
        'is_enantiomer_of', 'has_functional_parent', 'is_tautomer_of',
        'has_parent_hydride', 'is_substituent_group_from', 'develops_from',
        'lacks_part', 'only_in_taxon'
    }

    # set of symmetric relations
    OBO_SYM_RELATION_SET = {}

    # tag that marks the start of an entity in OBO
    OBO_ENTITY_START_TAG = "[Term]"

    # tag that marks the start of an entity in MeSH
    MESH_ENTITY_START_TAG = '*NEWRECORD'

    # tag patter
    TAG_PATTERN = re.compile("\[[a-zA-Z]*\]")

    @staticmethod
    def _chunkify(lines, kw):
        """
        Chunks lines delimited by kw
        :param lines: list of strings
        :param kw: delimiting keyword
        :return:
        """
        in_chunk = False
        chunk = []
        for line in lines:
            if not line:
                continue
            if line == kw:
                in_chunk = True
                if len(chunk) > 0:
                    yield chunk
                    chunk = []
            elif KBLoader.TAG_PATTERN.match(line):
                break
            elif in_chunk:
                chunk.append(line)

        if len(chunk) > 0:
            yield chunk

    # kb_filename=s3://ai2-s2-research/scigraph/data/raw_kbs/dbpedia201510_en/long_abstracts_en.ttl
    @staticmethod
    def import_dbpedia(kb_name, kb_filename, entities_count=0):
        """
        Instantiate a KnowledgeBase object with entities and relations from dbpedia
        :param kb_name:
        :param kb_filename:
        :param entities_count:
        :return:
        """
        # initialize the KB.
        kb = KnowledgeBase()
        kb.name = kb_name
        # only the "turtle" format is allowed for this kb.
        assert ('.ttl' in kb_filename)
        kb_filename = file_util.cache_file(kb_filename)

        # parse the turtle file
        abstracts_graph = rdflib.Graph()
        abstracts_graph.parse(kb_filename, format='turtle')
        logging.warning('done parsing dbpedia .ttl files.')

        counter = 0
        for item_subject, item_property, item_object in abstracts_graph:
            entity = KBEntity()
            entity.research_entity_id = str(item_subject)
            if not entity.research_entity_id.startswith(
                'http://dbpedia.org/resource/'
            ):
                continue
            entity.canonical_name = entity.research_entity_id[len(
                'http://dbpedia.org/resource/'
            ):].replace('_', ' ')
            entity.aliases.append(entity.canonical_name)
            entity.definition = str(item_object)
            # verify and add entity to the KB.
            kb.add_entity(entity)
            counter += 1
            if counter >= entities_count > 0:
                break
        return kb

    @staticmethod
    def import_obo_kb(kb_name, kb_filename):
        """
        Create a KnowledgeBase object with entities and relations from an OBO file
        :param kb_name:
        :param kb_filename: OBO file where KB is located
        :return:
        """
        # initialize the KB
        kb = KnowledgeBase()
        kb.name = kb_name

        for chunk in KBLoader._chunkify(
            file_util.read_lines(kb_filename), KBLoader.OBO_ENTITY_START_TAG
        ):
            # instantiate an empty entity.
            entity = KBEntity()

            # list of KBRelations to add
            relations = []

            for line_index, line in enumerate(chunk):
                if line.startswith('id: '):
                    # research_entity_id
                    entity.research_entity_id = line[len('id: '):]
                elif line.startswith('name: '):
                    # canonical_name
                    entity.canonical_name = line[len('name: '):].replace(
                        '_', ' '
                    )
                    entity.aliases.append(entity.canonical_name)
                elif line.startswith('def: '):
                    # definition
                    start_offset, end_offset = line.index(
                        '"'
                    ) + 1, line.rindex('"')
                    entity.definition = line[start_offset:end_offset]
                elif line.startswith('synonym: '):
                    # other aliases
                    start_offset, end_offset = line.index(
                        '"'
                    ) + 1, line.rindex('"')
                    entity.aliases.append(line[start_offset:end_offset])
                elif line.startswith('is_a: '):
                    # is_a relationships
                    assert entity.research_entity_id
                    splits = line.strip().split(' ')
                    assert (len(splits) > 1)
                    target_research_entity_id = '{}:{}'.format(
                        kb.name, splits[1]
                    )
                    relation = KBRelation(
                        relation_type='is_a',
                        entity_ids=[
                            entity.research_entity_id,
                            target_research_entity_id
                        ],
                        symmetric=True
                    )
                    relations.append(relation)
                elif line.startswith('relationship: '):
                    # other relationships
                    assert entity.research_entity_id
                    splits = line.split(' ')
                    assert (len(splits) > 2)
                    relation_type = splits[1]
                    target_research_entity_id = '{}:{}'.format(
                        kb.name, splits[2]
                    )
                    # is the relation symmetric?
                    if relation_type in KBLoader.OBO_ASYM_RELATION_SET:
                        symmetric = False
                    elif relation_type in KBLoader.OBO_SYM_RELATION_SET:
                        symmetric = True
                    else:
                        # unknown relation type
                        logging.info('unknown relation type: ' + relation_type)
                        assert False
                    relation = KBRelation(
                        relation_type=relation_type,
                        entity_ids=[
                            entity.research_entity_id,
                            target_research_entity_id
                        ],
                        symmetric=symmetric
                    )
                    relations.append(relation)
                elif line.startswith('intersection_of: ') or \
                        line.startswith('is_obsolete: ') or \
                        line.startswith('comment: ') or \
                        line.startswith('disjoint_from: ') or \
                        line.startswith('alt_id: ') or \
                        line.startswith('xref: ') or \
                        line.startswith('property_value: has_rank') or \
                        line.startswith('subset: ') or \
                        line.startswith('xref_analog') or \
                        line.startswith('xylem') or \
                        line.startswith('related_synonym') or \
                        line.startswith('exact_synonym') or \
                        line.startswith('broad_synonym') or \
                        line.startswith('narrow_synonym') or \
                        line.startswith('namespace') or \
                        line.startswith('consider') or \
                        line.startswith('replaced_by') or \
                        line.startswith('union_of'):
                    # properties don't map naturally to the unified schema.
                    pass
                else:
                    # unknown obo property.
                    logging.info('unknown OBO property: ' + line)
                    assert False

            # add relations to entity and to kb
            for rel in relations:
                kb.add_relation(rel)
                rel_index = len(kb.relations) - 1
                entity.relation_ids.append(rel_index)

            # add entity to kb
            kb.add_entity(entity)

        return kb

    @staticmethod
    def import_mesh(name, mesh_filename):
        """
        Create a KnowledgeBase object with entities from MeSH file
        :param name:
        :param mesh_filename:
        :return:
        """
        # initialize the KB
        kb = KnowledgeBase()
        kb.name = name

        def _make_mesh_entity(entity_chunk):
            """
            Make a KBEntity from each MeSH chunk
            :param entity_chunk:
            :return:
            """
            entity = KBEntity()
            for line in entity_chunk:
                fields = line.split(" = ")
                if len(fields) != 2:
                    continue
                key, value = fields[0], fields[1]
                if key == 'UI':
                    entity.research_entity_id = value
                elif key == 'MH' or key == 'SH':
                    entity.canonical_name = value
                    entity.aliases.append(value)
                elif key == 'ENTRY' or key == 'PRINT ENTRY':
                    entity.aliases.append(value.split("|")[0])
                elif key == 'MS':
                    entity.definition = value
            return entity

        for chunk in KBLoader._chunkify(
            file_util.read_lines(mesh_filename), KBLoader.MESH_ENTITY_START_TAG
        ):
            kb.add_entity(_make_mesh_entity(chunk))
        return kb

    @staticmethod
    def import_owl_kb(kb_name, kb_filename):
        """
        Create a KnowledgeBase object with entities and relations from an OWL file
        :param kb_name:
        :param kb_filename:
        :return:
        """

        # get the description label for this resource id
        def get_label(l):
            if l.text is not None:
                return l.text
            else:
                r_id = l.get('{' + ns['rdf'] + '}resource')
                if r_id in descriptions:
                    return descriptions[r_id][0]
            return None

        assert kb_filename.endswith('.owl') or kb_filename.endswith('.rdf')

        # initialize the KB
        kb = KnowledgeBase()
        kb.name = kb_name

        # parse the file
        try:
            tree = etree.parse(kb_filename)
        except etree.XMLSyntaxError:
            p = etree.XMLParser(huge_tree=True)
            tree = etree.parse(kb_filename, parser=p)

        root = tree.getroot()
        ns = root.nsmap

        if None in ns:
            del ns[None]

        # get description dict
        descriptions = dict()
        for desc in root.findall('rdf:Description', ns):
            resource_id = desc.get('{' + ns['rdf'] + '}about')
            try:
                labels = []
                for label in desc.findall('rdfs:label', ns):
                    if label.text is not None:
                        labels.append(label.text)
                if 'skos' in ns:
                    for label in desc.findall('skos:prefLabel', ns):
                        if label.text is not None:
                            labels.append(label.text)
                if 'oboInOwl' in ns:
                    for syn in desc.findall('oboInOwl:hasExactSynonym', ns):
                        if syn.text is not None:
                            labels.append(syn.text)
                    for syn in desc.findall('oboInOwl:hasRelatedSynonym', ns) \
                            + desc.findall('oboInOwl:hasNarrowSynonym', ns) \
                            + desc.findall('oboInOwl:hasBroadSynonym', ns):
                        if syn.text is not None:
                            labels.append(syn.text)
                if len(labels) > 0:
                    descriptions[resource_id] = labels
            except AttributeError:
                continue

        # parse OWL classes
        for cl in root.findall('owl:Class', ns):
            # instantiate an entity.
            research_entity_id = cl.get('{' + ns['rdf'] + '}about')
            entity = KBEntity(research_entity_id, None, [], '')

            # list of KBRelations to add
            relations = []

            if entity.research_entity_id is not None and entity.research_entity_id != '':
                try:
                    labels = []

                    # get rdfs labels
                    for label in cl.findall('rdfs:label', ns):
                        l_text = get_label(label)
                        if l_text is not None:
                            labels.append(l_text)

                    # add labels from description
                    if entity.research_entity_id in descriptions:
                        labels += descriptions[entity.research_entity_id]

                    # get skos labels
                    if 'skos' in ns:
                        for label in cl.findall('skos:prefLabel', ns):
                            l_text = get_label(label)
                            if l_text is not None:
                                labels.append(l_text)
                        for label in cl.findall('skos:altLabel', ns):
                            l_text = get_label(label)
                            if l_text is not None:
                                labels.append(l_text)
                        for label in cl.findall('skos:hiddenLabel', ns):
                            l_text = get_label(label)
                            if l_text is not None:
                                labels.append(l_text)

                    # get synonyms
                    if 'oboInOwl' in ns:
                        for syn in cl.findall('oboInOwl:hasExactSynonym', ns):
                            l_text = get_label(syn)
                            if l_text is not None:
                                labels.append(l_text)
                        for syn in cl.findall('oboInOwl:hasRelatedSynonym', ns) \
                                + cl.findall('oboInOwl:hasNarrowSynonym', ns) \
                                + cl.findall('oboInOwl:hasBroadSynonym', ns):
                            l_text = get_label(syn)
                            if l_text is not None:
                                labels.append(l_text)

                    # set canonical_name and aliases
                    if len(labels) > 0:
                        entity.canonical_name = labels[0]
                        entity.aliases = list(
                            set([lab.lower() for lab in labels])
                        )

                    # if no name available (usually entity from external KB), replace name with id
                    if entity.canonical_name is None:
                        entity.canonical_name = entity.research_entity_id

                    # get definition
                    if 'skos' in ns:
                        for definition in cl.findall('skos:definition', ns):
                            if definition.text is not None:
                                entity.definition += definition.text.lower(
                                ) + ' '
                    if 'obo' in ns:
                        for definition in cl.findall('obo:IAO_0000115', ns):
                            if definition.text is not None:
                                entity.definition += definition.text.lower(
                                ) + ' '
                    entity.definition = entity.definition.strip()

                    # get subclass relations
                    for sc_rel in cl.findall('rdfs:subClassOf', ns):
                        target_research_entity_id = '{}:{}'.format(
                            kb.name,
                            sc_rel.get('{' + ns['rdf'] + '}resource', ns)
                        )
                        relation = KBRelation(
                            relation_type='subClassOf',
                            entity_ids=[
                                entity.research_entity_id,
                                target_research_entity_id
                            ],
                            symmetric=False
                        )
                        relations.append(relation)
                except AttributeError:
                    pass

                # add relations to entity and to kb
                for rel in relations:
                    kb.add_relation(rel)
                    rel_index = len(kb.relations) - 1
                    entity.relation_ids.append(rel_index)

                # add entity to kb
                kb.add_entity(entity)

        return kb

    @staticmethod
    def import_kb(kb_name, kb_filename):
        """
        Returns a KnowledgeBase object loaded from kb_filename. The KB
        must be one of the supported one below.
        :param kb_name:
        :param kb_filename:
        :return:
        """
        # if needed, copy the file locally and update kb_filename.
        delete_local_copy = False
        if kb_filename.startswith('s3'):
            delete_local_copy = True
            kb_filename = file_util.cache_file(kb_filename)

        kb = None
        if kb_name in {
            KBLoader.SEQUENCE_ONTOLOGY, KBLoader.NCBI_TAXONOMY,
            KBLoader.CHEBI_TAXONOMY, KBLoader.GO_TAXONOMY,
            KBLoader.PR_TAXONOMY, KBLoader.CL_TAXONOMY,
            KBLoader.UNK_OBO_TAXONOMY
        }:
            kb = KBLoader.import_obo_kb(kb_name, kb_filename)
        elif kb_name == KBLoader.MESH_TAXONOMY:
            kb = KBLoader.import_mesh(kb_name, kb_filename)
        elif kb_name == KBLoader.DBPEDIA:
            kb = KBLoader.import_dbpedia(kb_name, kb_filename)
        elif kb_name == KBLoader.MERGED:
            kb = KnowledgeBase.load(kb_filename)
        else:
            raise LookupError("Unknown kb_name: {}".format(kb_name))

        # remove the local copy of the raw kb file(s).
        if delete_local_copy:
            os.remove(kb_filename)

        # return the imported kb.
        assert (kb is not None)
        return kb

    @staticmethod
    def import_craft_kbs_from_dir(data_dir):
        """
        Import OBO KBs used in CRAFT corpus
        :param data_dir:
        :return:
        """
        pr_kb = KBLoader.import_kb(
            KBLoader.PR_TAXONOMY, os.path.join(data_dir, 'PR.obo')
        )
        go_kb = KBLoader.import_kb(
            KBLoader.GO_TAXONOMY, os.path.join(data_dir, 'GO.obo')
        )
        cl_kb = KBLoader.import_kb(
            KBLoader.CL_TAXONOMY, os.path.join(data_dir, 'CL.obo')
        )
        so_kb = KBLoader.import_kb(
            KBLoader.SEQUENCE_ONTOLOGY, os.path.join(data_dir, 'SO.obo')
        )
        ncbi_kb = KBLoader.import_kb(
            KBLoader.NCBI_TAXONOMY, os.path.join(data_dir, 'NCBITaxon.obo')
        )
        chebi_kb = KBLoader.import_kb(
            KBLoader.CHEBI_TAXONOMY, os.path.join(data_dir, 'CHEBI.obo')
        )
        return [pr_kb, go_kb, cl_kb, so_kb, ncbi_kb, chebi_kb]

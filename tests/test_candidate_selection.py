from emma.OntoEmma import OntoEmma
from emma.CandidateSelection import CandidateSelection
import emma.constants as constants
import os
import unittest

TEST_DATA = os.path.join('tests', 'data')


class TestCandidateSelector(unittest.TestCase):

    def test_candidate_selector(self):

        source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
        target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')
        input_alignment_file = os.path.join(TEST_DATA, 'test_input_alignment.tsv')

        ontoemma = OntoEmma()
        s_kb = ontoemma._normalize_kb(
            ontoemma.load_kb(source_ont_file)
        )
        t_kb = ontoemma._normalize_kb(
            ontoemma.load_kb(target_ont_file)
        )

        cand_sel = CandidateSelection(s_kb, t_kb)

        candidates = []

        for s_ent in s_kb.entities:
            s_ent_id = s_ent.research_entity_id
            for t_ent_id in cand_sel.select_candidates(s_ent_id)[:constants.KEEP_TOP_K_CANDIDATES]:
                candidates.append((s_ent_id, t_ent_id, 1.0))

        p, r, f1 = ontoemma.compare_alignment_to_gold(input_alignment_file, candidates, s_kb, t_kb, None)

        assert r >= 0.99
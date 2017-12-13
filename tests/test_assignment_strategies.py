from emma.OntoEmma import OntoEmma
import os
import unittest
import pickle

TEST_DATA = os.path.join('tests', 'data')


class TestAssignmentStrategies(unittest.TestCase):

    sim_scores_fpath = os.path.join(TEST_DATA, 'test_sim_scores.pickle')
    sim_scores = pickle.load(open(sim_scores_fpath, 'rb'))
    ontoemma = OntoEmma()

    def test_best_match(self):
        alignment = self.ontoemma._apply_best_alignment_strategy(self.sim_scores)
        assert len(alignment) == 6

    def test_all_match(self):
        alignment = self.ontoemma._apply_all_alignment_strategy(self.sim_scores)
        assert len(alignment) == 6

    def test_modh_match(self):
        source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
        target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')

        ontoemma = OntoEmma()
        s_kb = ontoemma._normalize_kb(
            ontoemma.load_kb(source_ont_file)
        )
        t_kb = ontoemma._normalize_kb(
            ontoemma.load_kb(target_ont_file)
        )
        alignment = self.ontoemma._apply_modh_alignment_strategy(self.sim_scores, s_kb, t_kb)
        assert len(alignment) == 6
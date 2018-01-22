from emma.OntoEmma import OntoEmma
import os
import unittest
import pickle

TEST_DATA = os.path.join('tests', 'data')


class TestNeighborhoodSimilarity(unittest.TestCase):

    source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
    target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')

    ontoemma = OntoEmma()

    s_kb = ontoemma.load_kb(source_ont_file)
    s_kb.normalize_kb()

    t_kb = ontoemma.load_kb(target_ont_file)
    t_kb.normalize_kb()

    sim_scores_fpath = os.path.join(TEST_DATA, 'test_sim_scores.pickle')
    sim_scores = pickle.load(open(sim_scores_fpath, 'rb'))

    def test_neighbor_similarity_null(self):
        neighborhood_sim_null = self.ontoemma._compute_neighborhood_similarities(
            self.sim_scores, self.s_kb, self.t_kb, 0
        )
        assert neighborhood_sim_null == self.sim_scores

    def test_neighborhood_similarity_oneiter(self):
        neighborhood_sim_one = self.ontoemma._compute_neighborhood_similarities(
            self.sim_scores, self.s_kb, self.t_kb, 1
        )
        assert len(neighborhood_sim_one) == len(self.sim_scores)
        assert neighborhood_sim_one != self.sim_scores

    def test_neighborhood_similarity_fiveiter(self):
        neighborhood_sim_five = self.ontoemma._compute_neighborhood_similarities(
            self.sim_scores, self.s_kb, self.t_kb, 5
        )
        assert len(neighborhood_sim_five) == len(self.sim_scores)
        assert neighborhood_sim_five != self.sim_scores

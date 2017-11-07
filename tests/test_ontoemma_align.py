from emma.OntoEmma import OntoEmma
import os
import unittest

TEST_DATA = os.path.join('tests', 'data')


class TestOntoEmmaAlign(unittest.TestCase):
    def test_align_lr(self):
        model_path = os.path.join(TEST_DATA, 'test_lr_model.pickle')
        source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
        target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')
        input_alignment_file = os.path.join(TEST_DATA, 'test_input_alignment.tsv')
        output_alignment_file = os.path.join(TEST_DATA, 'test_output_alignment.tsv')

        matcher = OntoEmma()
        p, r, f1 = matcher.align(
            'lr', model_path,
            source_ont_file, target_ont_file,
            input_alignment_file, output_alignment_file
        )
        assert p >= 1.0
        assert r >= 0.6
        assert f1 >= 0.5

    def test_train_nn(self):
        model_path = os.path.join(TEST_DATA, 'test_nn_model', 'model.tar.gz')
        source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
        target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')
        input_alignment_file = os.path.join(TEST_DATA, 'test_input_alignment.tsv')
        output_alignment_file = os.path.join(TEST_DATA, 'test_output_alignment.tsv')

        matcher = OntoEmma()
        p, r, f1 = matcher.align(
            'nn', model_path,
            source_ont_file, target_ont_file,
            input_alignment_file, output_alignment_file
        )
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

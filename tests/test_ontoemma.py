from emma.OntoEmma import OntoEmma
import os
import shutil
import unittest

TEST_DATA = os.path.join('tests', 'data')


class TestOntoEmmaAlign(unittest.TestCase):

    def test_nn(self):
        config_file = os.path.join(TEST_DATA, 'test_nn_config_file.json')
        model_path = os.path.join(TEST_DATA, 'test_nn_model')

        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        matcher = OntoEmma()
        matcher.train(
            'nn', model_path, config_file
        )

        assert(os.path.exists(os.path.join(model_path, 'model.tar.gz')))

        model_path = os.path.join(TEST_DATA, 'test_nn_model', 'model.tar.gz')
        source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
        target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')
        input_alignment_file = os.path.join(TEST_DATA, 'test_input_alignment.tsv')
        output_alignment_file = os.path.join(TEST_DATA, 'test_output_alignment.tsv')

        matcher = OntoEmma()
        p, r, f1 = matcher.align(
            'nn', model_path,
            source_ont_file, target_ont_file,
            input_alignment_file, output_alignment_file,
            -1
        )
        assert p >= 0.0
        assert r >= 0.0
        assert f1 >= 0.0

    def test_lr(self):
        model_path = os.path.join(TEST_DATA, 'test_lr_model.pickle')
        source_ont_file = os.path.join(TEST_DATA, 'test_source_ont.json')
        target_ont_file = os.path.join(TEST_DATA, 'test_target_ont.json')
        input_alignment_file = os.path.join(TEST_DATA, 'test_input_alignment.tsv')
        output_alignment_file = os.path.join(TEST_DATA, 'test_output_alignment.tsv')

        matcher = OntoEmma()
        p, r, f1 = matcher.align(
            'lr', model_path,
            source_ont_file, target_ont_file,
            input_alignment_file, output_alignment_file,
            -1
        )

        assert p >= 0.8
        assert r >= 0.6
        assert f1 >= 0.7


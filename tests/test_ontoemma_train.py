from emma.OntoEmma import OntoEmma
from emma.OntoEmmaLRModel import OntoEmmaLRModel
import os
import shutil
import unittest

TEST_DATA = os.path.join('tests', 'data')


class TestOntoEmmaTrain(unittest.TestCase):

    def test_train_nn(self):
        config_file = os.path.join(TEST_DATA, 'test_nn_config_file.json')
        model_path = os.path.join(TEST_DATA, 'test_nn_model')

        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        matcher = OntoEmma()
        matcher.train(
            'nn', model_path, config_file
        )

        assert(os.path.exists(os.path.join(model_path, 'model.tar.gz')))

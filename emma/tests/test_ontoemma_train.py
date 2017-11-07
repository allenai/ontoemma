from emma.OntoEmma import OntoEmma
from emma.OntoEmmaLRModel import OntoEmmaLRModel
import os
import shutil
import unittest

TEST_DATA = os.path.join('emma', 'tests', 'data')



class TestOntoEmmaTrain(unittest.TestCase):
    def test_train_lr(self):
        config_file = os.path.join(TEST_DATA, 'test_lr_config_file.json')
        model_path = os.path.join(TEST_DATA, 'test_lr_model.pickle')

        if os.path.exists(model_path):
            os.remove(model_path)

        matcher = OntoEmma()
        matcher.train(
            'lr', model_path, config_file
        )

        assert(os.path.exists(model_path))

        lr_model = OntoEmmaLRModel()
        lr_model.load(model_path)



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

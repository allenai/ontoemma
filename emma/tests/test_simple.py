import os
import shutil
import unittest

TEST_DATA = os.path.join('emma', 'tests', 'data')


class TestOntoEmmaTrue(unittest.TestCase):
    def test_true(self):
        assert (1+1 == 2)
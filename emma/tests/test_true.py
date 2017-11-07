from emma.OntoEmma import OntoEmma
import os
import unittest

TEST_DATA = os.path.join('emma', 'tests')


class TestOntoEmmaTrue(unittest.TestCase):
    def test_simple(self):
        assert True

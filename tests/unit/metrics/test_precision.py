import os
import unittest

from telescope.metrics.precision.metric import Precision
from tests.data import DATA_PATH


class TestPrecision(unittest.TestCase):
    labels = ["a", "b", "c"]
    precision = Precision(labels=labels)
    pred = [ "a", "b", "a", "b", "c"]
    true = [ "a", "b", "c", "a", "c"]

    def test_name_property(self):
        self.assertEqual(self.precision.name, "Precision")

    def test_score(self):

        expected_sys = (0.5 + 0.5 + 1) / 3

        result = self.precision.score([],self.pred,self.true)
        
        self.assertEqual(result.sys_score, expected_sys)
        self.assertListEqual(result.ref, self.true)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, self.pred)


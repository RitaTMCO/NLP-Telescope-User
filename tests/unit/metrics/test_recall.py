import os
import unittest

from telescope.metrics.recall.metric import Recall
from tests.data import DATA_PATH


class TestRecall(unittest.TestCase):
    labels = ["a", "b", "c"]
    recall = Recall(labels=labels)
    pred = [ "a", "b", "a", "b", "c"]
    true = [ "a", "b", "c", "a", "c"]

    def test_name_property(self):
        self.assertEqual(self.recall.name, "Recall")

    def test_score(self):

        expected_sys = (0.5 + 1 + 0.5) / 3

        result = self.recall.score([],self.pred,self.true)
        
        self.assertEqual(result.sys_score, expected_sys)
        self.assertListEqual(result.ref, self.true)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, self.pred)


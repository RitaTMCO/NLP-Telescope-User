import os
import unittest

from telescope.metrics.accuracy.metric import Accuracy
from tests.data import DATA_PATH


class TestAccuracy(unittest.TestCase):
    labels = ["a", "b", "c"]
    accuracy = Accuracy(labels=labels)
    pred = [ "a", "b", "a", "b", "c"]
    true = [ "a", "b", "c", "a", "c"]

    def test_name_property(self):
        self.assertEqual(self.accuracy.name, "Accuracy")

    def test_score(self):

        expected_seg = [0.6,0.8,0.8]
        expected_sys = 0.6

        result = self.accuracy.score([],self.pred,self.true)
        
        self.assertEqual(result.sys_score, expected_sys)
        for i in range(len(self.labels)):
            self.assertEqual(result.seg_scores[i], expected_seg[i])
        self.assertListEqual(result.ref, self.true)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, self.pred)


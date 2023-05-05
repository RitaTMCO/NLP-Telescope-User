#https://github.com/pltrdy/rouge

import unittest

from telescope.metrics.rouge_two.metric import ROUGETwo


class TestROUGETwo(unittest.TestCase):

    rouge_two = ROUGETwo(language="en")

    def test_score(self):

        cand = ["the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"]
        ref = ["this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentiTwod on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"]

        expected_sys =  0.23423422957552154

        result = self.rouge_two.score([], cand, ref)
        self.assertEqual(result.sys_score, expected_sys)
        self.assertFalse(result.seg_scores)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, cand)
        self.assertEqual(result.precision, 0.3170731707317073)
        self.assertEqual(result.recall, 0.18571428571428572)


    def test_name_property(self):
        self.assertEqual(self.rouge_two.name, "ROUGE-2")

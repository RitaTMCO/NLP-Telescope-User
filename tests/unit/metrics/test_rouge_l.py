#https://github.com/pltrdy/rouge

import unittest

from telescope.metrics.rouge_l.metric import ROUGEL


class TestROUGEL(unittest.TestCase):

    rouge_l = ROUGEL(language="en")

    def test_score(self):

        cand = ["the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"]
        ref = ["this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"]

        expected_sys =  0.44705881864636676
        expected_seg = [0.44705881864636676]

        result = self.rouge_l.score([], cand, ref)
        self.assertEqual(result.sys_score, expected_sys)
        self.assertListEqual(result.seg_scores, expected_seg)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, cand)
        self.assertEqual(result.precision, 0.5277777777777778)
        self.assertEqual(result.recall, 0.3877551020408163)


    def test_name_property(self):
        self.assertEqual(self.rouge_l.name, "ROUGE-L")

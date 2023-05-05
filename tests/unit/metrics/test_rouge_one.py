#https://github.com/pltrdy/rouge

import unittest

from telescope.metrics.rouge_one.metric import ROUGEOne


class TestROUGEOne(unittest.TestCase):

    rouge_one = ROUGEOne(language="en")

    def test_score(self):

        cand = ["the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"]
        ref = ["this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"]

        expected_sys =  0.49411764217577864

        result = self.rouge_one.score([], cand, ref)
        self.assertEqual(result.sys_score, expected_sys)
        self.assertFalse(result.seg_scores)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, cand)
        self.assertEqual(result.precision, 0.5833333333333334)
        self.assertEqual(result.recall, 0.42857142857142855)


    def test_name_property(self):
        self.assertEqual(self.rouge_one.name, "ROUGE-1")

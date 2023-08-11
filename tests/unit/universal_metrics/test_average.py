#https://github.com/pltrdy/rouge

import unittest

from telescope.universal_metrics.average import Average
from telescope.testset import MultipleTestset
from telescope.metrics.result import MetricResult, MultipleMetricResults


class TestAverage(unittest.TestCase):

    # sys_id:sys_name
    systems_names = {"Sys 1": "Sys A", "Sys 2":"Sys B", "Sys 3":"Sys C"}

    metrics = ["mock_1","mock_2","mock_3"]

    testset = MultipleTestset(
        src=["a", "b", "c"],
        ref=["a", "b", "c"],
        ref_id="Ref 1",
        systems_output={
            "Sys 1": ["a", "d", "c"],
            "Sys 2": ["a", "k", "c"],
            "Sys 3": ["a", "p", "c"]
        },
        filenames = ["src.txt","ref.txt","sys1.txt","sys2.txt","sys3.txt"]
    )

    multiple_result_1 = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.1,
                seg_scores=[1, 0.5, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 1"],
                ref=testset.ref,
                metric="mock_1",
            ),
            "Sys 2": MetricResult(
                sys_score=0.1,
                seg_scores=[1, 0.25, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 2"],
                ref=testset.ref,
                metric="mock_1",
            ),
            "Sys 3": MetricResult(
                sys_score=0.3,
                seg_scores=[1, 0.75, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 3"],
                ref=testset.ref,
                metric="mock_1",
            )
        }
    )

    multiple_result_2 = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.1,
                seg_scores=[1, 0.5, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 1"],
                ref=testset.ref,
                metric="mock_2",
            ),
            "Sys 2": MetricResult(
                sys_score=0.2,
                seg_scores=[1, 0.25, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 2"],
                ref=testset.ref,
                metric="mock_2",
            ),
            "Sys 3": MetricResult(
                sys_score=0.3,
                seg_scores=[1, 0.75, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 3"],
                ref=testset.ref,
                metric="mock_2",
            )
        }
    )

    multiple_result_3 = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.1,
                seg_scores=[1, 0.5, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 1"],
                ref=testset.ref,
                metric="mock_3",
            ),
            "Sys 2": MetricResult(
                sys_score=0.3,
                seg_scores=[1, 0.25, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 2"],
                ref=testset.ref,
                metric="mock_3",
            ),
            "Sys 3": MetricResult(
                sys_score=0.3,
                seg_scores=[1, 0.75, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 3"],
                ref=testset.ref,
                metric="mock_3",
            )
        }
    )

    multiple_metric_result = {"mock_1":multiple_result_1,"mock_2":multiple_result_2,"mock_3":multiple_result_3}

    average = Average(multiple_metric_result)


    def test_score_calculation_and_ranking(self):

        expected_sys_score = {"Sys 1": 0.1, "Sys 2":0.2, "Sys 3":0.3}
        expected_sys_rank = {"Sys 1": 3, "Sys 2": 2, "Sys 3": 1}

        result = self.average.universal_score_calculation_and_ranking(self.testset)

        for sys in list(self.systems_names.keys()):
            sys_result  = result.systems_universal_metrics_results[sys]
            self.assertListEqual(sys_result.ref, self.testset.ref)
            self.assertListEqual(sys_result.system_output, self.testset.systems_output[sys])
            self.assertListEqual(sys_result.metrics, self.metrics)
            self.assertEqual(sys_result.universal_metric, "average")
            self.assertEqual(sys_result.title, "Average")
            self.assertEqual(sys_result.rank, expected_sys_rank[sys])
            self.assertAlmostEqual(sys_result.universal_score, expected_sys_score[sys],places=5)
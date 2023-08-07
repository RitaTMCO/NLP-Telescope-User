# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

from telescope.testset import MultipleTestset
from telescope.metrics.result import MetricResult, PairwiseResult, MultipleMetricResults
from telescope.plotting import (plot_bucket_comparison,
                                plot_bucket_multiple_comparison,
                                plot_pairwise_distributions,
                                plot_multiple_distributions,
                                plot_segment_comparison,
                                plot_multiple_segment_comparison,
                                confusion_matrix_of_system,
                                confusion_matrix_focused_on_one_label,
                                analysis_labels,
                                incorrect_examples
                                )

from tests.data import DATA_PATH


class TestPlots(unittest.TestCase):

    result = PairwiseResult(
        x_result=MetricResult(
            sys_score=0.5,
            seg_scores=[0, 0.5, 1],
            src=["a", "b", "c"],
            cand=["a", "b", "c"],
            ref=["a", "b", "c"],
            metric="mock",
        ),
        y_result=MetricResult(
            sys_score=0.25,
            seg_scores=[0, 0.25, 0.5],
            src=["a", "b", "c"],
            cand=["a", "k", "c"],
            ref=["a", "b", "c"],
            metric="mock",
        ),
    )

    # sys_id:sys_name
    systems_names = {"Sys 1": "Sys A", "Sys 2":"Sys B", "Sys 3":"Sys C"}

    testset = MultipleTestset(
        src=["a", "b", "c"],
        ref=["a", "b", "c"],
        systems_output={
            "Sys 1": ["a", "d", "c"],
            "Sys 2": ["a", "k", "c"],
            "Sys 3": ["a", "p", "c"]
        },
        filenames = ["src.txt","ref.txt","sys1.txt","sys2.txt","sys3.txt"]
    )

    multiple_result = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.833,
                seg_scores=[1, 0.5, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 1"],
                ref=testset.ref,
                metric="mock",
            ),
            "Sys 2": MetricResult(
                sys_score=0.750,
                seg_scores=[1, 0.25, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 2"],
                ref=testset.ref,
                metric="mock",
            ),
            "Sys 3": MetricResult(
                sys_score=0.916,
                seg_scores=[1, 0.75, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 3"],
                ref=testset.ref,
                metric="mock",
            )
        }
    )

    multiple_result_comet = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.833,
                seg_scores=[1, 0.5, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 1"],
                ref=testset.ref,
                metric="COMET",
            ),
            "Sys 2": MetricResult(
                sys_score=0.750,
                seg_scores=[1, 0.25, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 2"],
                ref=testset.ref,
                metric="COMET",
            ),
            "Sys 3": MetricResult(
                sys_score=0.916,
                seg_scores=[1, 0.75, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 3"],
                ref=testset.ref,
                metric="COMET",
            )
        }
    )

    multiple_result_bertscore = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.833,
                seg_scores=[1, 0.5, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 1"],
                ref=testset.ref,
                metric="BERTScore",
            ),
            "Sys 2": MetricResult(
                sys_score=0.750,
                seg_scores=[1, 0.25, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 2"],
                ref=testset.ref,
                metric="BERTScore",
            ),
            "Sys 3": MetricResult(
                sys_score=0.916,
                seg_scores=[1, 0.75, 1],
                src=testset.src,
                cand=testset.systems_output["Sys 3"],
                ref=testset.ref,
                metric="BERTScore",
            )
        }
    )


    labels = ['a', 'b', 'c']

    testset_class = MultipleTestset(
        src=["a", "b", "c"],
        ref=["a", "b", "c"],
        systems_output={
            "Sys 1": ["a", "d", "c"],
            "Sys 2": ["a", "b", "c"],
            "Sys 3": ["a", "p", "c"]
        },
        filenames = ["src.txt","ref.txt","sys1.txt","sys2.txt","sys3.txt"],
    )

    multiple_result_class = MultipleMetricResults(
        systems_metric_results = {
            "Sys 1": MetricResult(
                sys_score=0.833,
                seg_scores=[1, 0.5, 1],
                src=testset_class.src,
                cand=testset_class.systems_output["Sys 1"],
                ref=testset_class.ref,
                metric="mock",
            ),
            "Sys 2": MetricResult(
                sys_score=1.000,
                seg_scores=[1, 1, 1],
                src=testset_class.src,
                cand=testset_class.systems_output["Sys 2"],
                ref=testset_class.ref,
                metric="mock",
            ),
            "Sys 3": MetricResult(
                sys_score=0.916,
                seg_scores=[1, 0.75, 1],
                src=testset_class.src,
                cand=testset_class.systems_output["Sys 3"],
                ref=testset_class.ref,
                metric="mock",
            )
        }
    )

    @classmethod
    def tearDownClass(cls):
        os.remove(DATA_PATH + "/segment-comparison.html")
        os.remove(DATA_PATH + "/scores-distribution.html")
        os.remove(DATA_PATH + "/bucket-analysis.png")
        os.remove(DATA_PATH + "/Sys A-Sys B_multiple-segment-comparison.html")
        os.remove(DATA_PATH + "/Sys B-Sys C_multiple-segment-comparison.html")
        os.remove(DATA_PATH + "/Sys C-Sys A_multiple-segment-comparison.html")
        os.remove(DATA_PATH + "/multiple-scores-distribution.html")
        os.remove(DATA_PATH + "/COMET-multiple-bucket-analysis.png")
        os.remove(DATA_PATH + "/BERTScore-multiple-bucket-analysis.png")
        os.remove(DATA_PATH + "/confusion-matrix-Sys_A.png")
        os.remove(DATA_PATH + "/Sys_B-label-a.png")
        os.remove(DATA_PATH + "/Sys_C-label-b.png")
        os.remove(DATA_PATH + "/Sys_A-label-c.png")
        os.remove(DATA_PATH + "/mock-analysis-labels-bucket.png")
        os.remove(DATA_PATH + "/incorrect-examples.csv")

    def test_segment_comparison(self):
        plot_segment_comparison(self.result, DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "segment-comparison.html"))
        )
    
    def test_multiple_segment_comparison_A_B(self):
        system_A = ["Sys 1", self.systems_names["Sys 1"]]
        system_B = ["Sys 2", self.systems_names["Sys 2"]]
        plot_multiple_segment_comparison(self.multiple_result, system_A, system_B, source=True, saving_dir = DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "Sys A-Sys B_multiple-segment-comparison.html"))
        )
    
    def test_multiple_segment_comparison_B_C(self):
        system_B = ["Sys 2", self.systems_names["Sys 2"]]
        system_C = ["Sys 3", self.systems_names["Sys 3"]]
        plot_multiple_segment_comparison(self.multiple_result, system_B, system_C, source=True, saving_dir = DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "Sys B-Sys C_multiple-segment-comparison.html"))
        )

    def test_multiple_segment_comparison_C_A(self):
        system_A = ["Sys 1", self.systems_names["Sys 1"]]
        system_C = ["Sys 3", self.systems_names["Sys 3"]]
        plot_multiple_segment_comparison(self.multiple_result, system_C, system_A, source=True, saving_dir = DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "Sys C-Sys A_multiple-segment-comparison.html"))
        )

    def test_pairwise_distributions(self):
        plot_pairwise_distributions(self.result, DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "scores-distribution.html"))
        )

    def test_multiple_distributions(self):
        plot_multiple_distributions(self.multiple_result, list(self.systems_names.values()), DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "multiple-scores-distribution.html"))
        )

    def test_bucket_comparison(self):
        plot_bucket_comparison(self.result, DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "bucket-analysis.png")))

    def test_bucket_multiple_comparison_comet(self):        
        plot_bucket_multiple_comparison(self.multiple_result_comet, list(self.systems_names.values()), DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "COMET-multiple-bucket-analysis.png")))

    def test_bucket_multiple_comparison_bertscore(self):        
        plot_bucket_multiple_comparison(self.multiple_result_bertscore, list(self.systems_names.values()), DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "BERTScore-multiple-bucket-analysis.png")))
    
    def test_confusion_matrix_of_system(self):
        confusion_matrix_of_system(self.testset_class.ref, self.testset_class.systems_output["Sys 1"], self.labels, 
                                   self.systems_names["Sys 1"], DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "confusion-matrix-Sys_A.png")))
    
    def test_confusion_matrix_focused_on_one_label_a(self):
        confusion_matrix_focused_on_one_label(self.testset_class.ref, self.testset_class.systems_output["Sys 2"], "a", 
                                              self.labels, self.systems_names["Sys 2"], DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "Sys_B-label-a.png")))

    def test_confusion_matrix_focused_on_one_label_b(self):
        confusion_matrix_focused_on_one_label(self.testset_class.ref, self.testset_class.systems_output["Sys 3"], "b", 
                                              self.labels, self.systems_names["Sys 3"], DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "Sys_C-label-b.png")))

    def test_confusion_matrix_focused_on_one_label_c(self):
        confusion_matrix_focused_on_one_label(self.testset_class.ref, self.testset_class.systems_output["Sys 1"], "c", 
                                              self.labels, self.systems_names["Sys 1"], DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "Sys_A-label-c.png")))

    def test_analysis_labels(self):
        analysis_labels(self.multiple_result_class, list(self.systems_names.values()), self.labels, DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "mock-analysis-labels-bucket.png")))
    
    def test_incorrect_examples(self):
        num = int(len(self.testset_class.ref)/4) + 1
        incorrect_examples(self.testset_class, "Sys 1", num, [], [], [], DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "incorrect-examples.csv")))
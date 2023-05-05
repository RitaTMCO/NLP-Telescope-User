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
import abc
from typing import List, Tuple, Dict
from itertools import combinations

import numpy as np
from telescope.metrics.result import BootstrapResult, MetricResult, PairwiseResult, MultipleResult
from telescope.testset import PairwiseTestset, MultipleTestset


class Metric(metaclass=abc.ABCMeta):

    """ Class attibutes to be overwriten! """

    name = None
    segment_level = True

    def __init__(self, language: str = "X", labels: List[str] = [" "]):
        if not self.language_support(language):
            raise Exception(f"{language} is not supported by {self.name}.")
        else:
            self.language = language
        self.labels = labels

    @abc.abstractmethod
    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        """ Metric scoring function. """
        pass

    @classmethod
    def language_support(cls, language: str):
        return True

    def pairwise_comparison(self, testset: PairwiseTestset):
        """ Function that scores the two candidate systems inside a paired testset. """
        x_result = self.score(testset.src, testset.system_x, testset.ref)
        y_result = self.score(testset.src, testset.system_y, testset.ref)
        return PairwiseResult(x_result, y_result)

    def multiple_comparison(self, testset: MultipleTestset):
        """ Function that scores the multiple candidate systems inside a testset. """
        ref = testset.ref
        src = testset.src
        systems_metric_results = {name: self.score(src,output,ref) for name,output in testset.systems_output.items()}
        return MultipleResult(systems_metric_results)

    @classmethod
    def bootstrap_resampling(
        cls,
        testset: PairwiseTestset,
        num_samples: int,
        sample_ratio: float,
        pairwise_result: PairwiseResult = None,
    ):

        """
        Bootstrap resampling for system-level metrics such as BLEU that have to recompute
        the system-level score for each partition

        :param testset: Testset
        :param num_samples: Number of testset splits.
        :param sample_ratio: % of the testset to be used in each partition.
        :param pairwise_result: Precomputed scores between two systems.
        :return: BootstrapResult object
        """

        def update_wins(x_score: int, y_score: int, wins: Tuple[int]):
            if y_score > x_score:
                wins[1] += 1
            elif y_score < x_score:
                wins[0] += 1
            else:
                wins[2] += 1
            return wins

        def recompute_sys_scores(pairwise_result: PairwiseResult) -> (float, float):
            if cls.segment_level and pairwise_result is not None:
                reduces_x_scr = [
                    pairwise_result.x_result.seg_scores[i] for i in reduced_ids
                ]
                reduces_y_scr = [
                    pairwise_result.y_result.seg_scores[i] for i in reduced_ids
                ]
                return (
                    sum(reduces_x_scr) / len(reduces_x_scr),
                    sum(reduces_y_scr) / len(reduces_y_scr),
                )
            else:
                result = cls(testset.target_language).pairwise_comparison(
                    PairwiseTestset(
                        reduced_src,
                        reduced_x,
                        reduced_y,
                        reduced_ref,
                        language_pair=testset.language_pair,
                        filenames=testset.filenames,
                    )
                )
                return (result.x_result.sys_score, result.y_result.sys_score)

        n = len(testset)
        ids = list(range(n))
        sample_size = max(int(n * sample_ratio), 1)

        x_scores, y_scores = [], []
        wins = [0, 0, 0]
        for _ in range(num_samples):
            # Subsample the gold and system outputs (with replacement)
            reduced_ids = np.random.choice(ids, size=sample_size, replace=True)

            # Calculate accuracy on the reduced sample and save stats
            reduced_src = [testset[i][0] for i in reduced_ids]
            reduced_x = [testset[i][1] for i in reduced_ids]
            reduced_y = [testset[i][2] for i in reduced_ids]
            reduced_ref = [testset[i][3] for i in reduced_ids]

            x_result, y_result = recompute_sys_scores(pairwise_result)
            wins = update_wins(x_result, y_result, wins)
            x_scores.append(x_result)
            y_scores.append(y_result)

        return BootstrapResult(x_scores, y_scores, wins, cls.name)


    @classmethod
    def multiple_bootstrap_resampling(
        cls,
        testset: MultipleTestset,
        num_samples: int,
        sample_ratio: float,
        system_x: str,
        system_y: str,
        language: str,
        multiple_result: MultipleResult = None,
    ):

        """
        Bootstrap resampling for system-level metrics such as BLEU that have to recompute
        the system-level score for each partition

        :param testset: Testset
        :param num_samples: Number of testset splits.
        :param sample_ratio: % of the testset to be used in each partition.
        :param multiple_result: Precomputed scores of multiple systems.
        :param system_x: Name of system x.
        :param system_y: Name of system x.
        :param ref_filename: Filename of reference.
        :return: BootstrapResult object.
        """

        def update_wins(x_score: int, y_score: int, wins: Tuple[int]):
            if y_score > x_score:
                wins[1] += 1
            elif y_score < x_score:
                wins[0] += 1
            else:
                wins[2] += 1
            return wins

        def recompute_sys_scores(multiple_result: MultipleResult) -> (float, float):
            if cls.segment_level and multiple_result is not None:
                reduces_x_scr = [
                    multiple_result.systems_metric_results[system_x].seg_scores[i] 
                    for i in reduced_ids
                ]
                reduces_y_scr = [
                    multiple_result.systems_metric_results[system_y].seg_scores[i] 
                    for i in reduced_ids
                ]
                return (
                    sum(reduces_x_scr) / len(reduces_x_scr),
                    sum(reduces_y_scr) / len(reduces_y_scr),
                )
            else:
                result = cls(language, [" "]).multiple_comparison(
                    MultipleTestset(
                        reduced_src,
                        reduced_ref,
                        reducted_n_systems_output,
                        filenames=testset.filenames,
                    )
                )
                return (result.systems_metric_results[system_x].sys_score, 
                        result.systems_metric_results[system_y].sys_score)

        n = len(testset)
        ids = list(range(n))
        sample_size = max(int(n * sample_ratio), 1)

        x_scores, y_scores = [], []
        wins = [0, 0, 0]

        for _ in range(num_samples):
            # Subsample the gold and system outputs (with replacement)
            reduced_ids = np.random.choice(ids, size=sample_size, replace=True)

            # Calculate accuracy on the reduced sample and save stats
            reduced_src = [testset.src[i] for i in reduced_ids]
            reduced_ref = [testset.ref[i] for i in reduced_ids]
            reducted_n_systems_output = { system: [output[i] for i in reduced_ids]
                for system, output in testset.systems_output.items()}


            x_result, y_result = recompute_sys_scores(multiple_result)
            wins = update_wins(x_result, y_result, wins)
            x_scores.append(x_result)
            y_scores.append(y_result)

        return BootstrapResult(x_scores, y_scores, wins, cls.name)

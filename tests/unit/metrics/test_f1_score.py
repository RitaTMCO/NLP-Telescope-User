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

from telescope.metrics.f1_score.metric import F1Score
from tests.data import DATA_PATH


class TestF1Score(unittest.TestCase):
    labels = ["a", "b", "c"]
    f1score = F1Score(labels=labels)
    pred = [ "a", "b", "a", "b", "c"]
    true = [ "a", "b", "c", "a", "c"]

    def test_name_property(self):
        self.assertEqual(self.f1score.name, "F1-score")

    def test_score(self):

        expected_sys = (0.5 + 2/3 + 2/3) / 3

        result = self.f1score.score([],self.pred,self.true)
        
        self.assertEqual(result.sys_score, expected_sys)
        self.assertListEqual(result.ref, self.true)
        self.assertListEqual(result.src, [])
        self.assertListEqual(result.cand, self.pred)


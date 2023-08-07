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
import unittest

from telescope.filters import DuplicatesFilter
from telescope.testset import PairwiseTestset, MultipleTestset


class TestDuplicatesFilter(unittest.TestCase):

    hyp = ["a", "b", "cd", "hello"]
    alt = ["A", "b", "cD", "hello"]
    src = ["A", "A", "cD", "A"]
    ref = ["a", "b", "cd", "hello"]

    src_2 = ["A", "A", "cD", "a", "hello"]
    ref_2 = ["a", "a", "cd", "a"]

    testset = PairwiseTestset(
        src, hyp, alt, ref, "de-en", ["src.de", "ref.en", "hyp.en", "alt.hyp.en"]
    )

    multiple_testset = MultipleTestset(
        src_2, ref_2, {"Sys 1":hyp, "Sys 2":alt}, ["src.de", "hyp.en", "alt.hyp.en", "ref.en"]
    )
    

    def test_sucess_filter(self):
        filter = DuplicatesFilter(self.testset)
        orig_size = len(self.testset)
        self.testset.apply_filter(filter)
        self.assertEqual(len(self.testset), 2)
        self.assertTrue(len(self.testset) < orig_size)
        src, x, y, ref = self.testset[0]
        self.assertEqual(src, "A")
        src, x, y, ref = self.testset[1]
        self.assertEqual(src, "cD")

    def test_sucess_with_src_bigger(self):
        filter = DuplicatesFilter(self.multiple_testset)
        orig_ref_size = len(self.multiple_testset)
        self.multiple_testset.apply_filter(filter)

        self.assertEqual(len(self.multiple_testset), 2)
        self.assertEqual(len(self.multiple_testset), len(self.multiple_testset.systems_output["Sys 1"]))
        self.assertEqual(len(self.multiple_testset), len(self.multiple_testset.systems_output["Sys 2"]))
        self.assertTrue(len(self.multiple_testset) < orig_ref_size)
        self.assertEqual(len(self.multiple_testset.src), 5)
        
        _, ref, x, y = self.multiple_testset[0]
        self.assertEqual(ref, "a")
        self.assertEqual(x, "a")
        self.assertEqual(y, "A")

        _, ref, x, y = self.multiple_testset[1]
        self.assertEqual(ref, "cd")
        self.assertEqual(x, "cd")
        self.assertEqual(y, "cD")
       

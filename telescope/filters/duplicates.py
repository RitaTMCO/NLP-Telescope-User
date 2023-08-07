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
from collections import Counter
from typing import List

from telescope.filters.filter import Filter
from telescope.testset import Testset


class DuplicatesFilter(Filter):
    name = "duplicates"

    def __init__(self, testset: Testset, *args):
        self.testset = testset

    def apply_filter(self) -> List[int]:
        if len(self.testset.ref) == len(self.testset.src):
            counter = Counter(self.testset.src)
            file_item = 0
        else:
            counter = Counter(self.testset.ref)
            file_item = 1

        segments = []

        for i, item in enumerate(self.testset):
            seg = item[file_item]
            if counter[seg] == 0:
                continue
            # if counter > 1 we set it to 0 to skip the next time it appears
            if counter[seg] > 1:
                counter[seg] = 0

            segments.append(i)
        return segments

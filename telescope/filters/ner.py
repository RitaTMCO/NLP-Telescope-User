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
from typing import List

import stanza
from telescope.filters.filter import Filter
from telescope.testset import Testset

STANZA_NER_LANGS = ["ar", "zh", "nl", "en", "fr", "de", "ru", "uk"]


class NERFilter(Filter):
    name = "named-entities"

    def __init__(self, testset: Testset, source_language: str, target_language:str, *args):
        super().__init__(testset)
        self.set_language(source_language, target_language)
        stanza.download(self.language)
        self.engine = stanza.Pipeline(lang=self.language, processors="tokenize,ner")

    def set_language(self, source_language: str, target_language:str) -> None:
        if source_language in STANZA_NER_LANGS:
            self.language = source_language
            self.segments = self.testset.src
        elif target_language in STANZA_NER_LANGS:
            self.language = target_language
            self.segments = self.testset.ref
        else:
            raise Exception(
                "{} is not supperted by Stanza NER.".format(source_language + "-" + target_language)
            )

    def apply_filter(self) -> List[int]:
        segments_with_ne = []
        for i, segment in enumerate(self.segments):
            doc = self.engine(segment)
            if doc.ents:
                segments_with_ne.append(i)

        return segments_with_ne

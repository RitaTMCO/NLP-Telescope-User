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

from telescope.testset import PairwiseTestset, MultipleTestset
from telescope.collection_testsets import MTTestsets


class TestTestset(unittest.TestCase):
    
    testset = PairwiseTestset(
        src=['Bonjour le monde.', "C'est un test."],
        system_x=['Greetings world', 'This is an experiment.'],
        system_y=['Hi world.', 'This is a Test.'],
        ref=['Hello world.', 'This is a test.'],
        language_pair="en-fr",
        filenames=["src.txt", "google.txt", "unbabel.txt", "ref.txt"]
    )

    multiple_testset_1 = MultipleTestset(
        src=['Bonjour le monde.', "C'est un test."],
        ref=['Hello world.', 'This is a test.'],
        systems_output={"Sys 1": ['Greetings world', 'This is an experiment.'], 
        "Sys 2":['Hi world.', 'This is a Test.'], "Sys 3":['Hello world.', 'This is a test']},
        filenames=["src.txt", "ref_1.txt", "google.txt", "unbabel_1.txt", "unbabel_2.txt"]
    )

    multiple_testset_2 = MultipleTestset(
        src=['Bonjour le monde.', "C'est un test."],
        ref=['Greetings world', 'This is an experiment.'],
        systems_output={"Sys 1": ['Greetings world', 'This is an experiment.'], 
        "Sys 2":['Hi world.', 'This is a Test.'], "Sys 3":['Hello world.', 'This is a test']},
        filenames=["src.txt", "ref_2.txt", "google.txt", "unbabel_1.txt", "unbabel_2.txt"]
    )

    collection =  MTTestsets(
        src_name = 'src.txt',
        refs_names = ['ref_1.txt','ref_2.txt'],
        refs_indexes = {'ref_1.txt':"Ref A", 'ref_2.txt':"Ref B"},
        systems_indexes = {"google.txt":"Sys 1", "unbabel_1.txt":"Sys 2", "unbabel_2.txt":"Sys 3"},
        systems_names = {"Sys 1":"Sys A", "Sys 2":"Sys B", "Sys 3":"Sys C"},
        filenames = ["src.txt", "ref_1.txt", "ref_2.txt", "google.txt", "unbabel_1.txt", "unbabel_2.txt"],
        testsets = [multiple_testset_1, multiple_testset_2],
        language_pair="fr-en"
    )

    def test_length(self):
        self.assertEqual(len(self.testset), 2)
    
    def test_multiple_length(self):
        self.assertEqual(len(self.multiple_testset_1), 2)
        self.assertEqual(len(self.multiple_testset_2), 2)

    def test_get_item(self):
        expected = (
            'Bonjour le monde.',
            'Greetings world',
            'Hi world.',
            'Hello world.'
        )
        self.assertTupleEqual(expected, self.testset[0])
    
    def test_multiple_get_item(self):
        expected_1= (
            'Bonjour le monde.',
            'Hello world.',
            'Greetings world',
            'Hi world.',
            'Hello world.',
        )
        self.assertTupleEqual(expected_1, self.multiple_testset_1[0])

        expected_2=(
            'Bonjour le monde.',
            'Greetings world',
            'Greetings world',
            'Hi world.',
            'Hello world.'
        )
        self.assertTupleEqual(expected_2, self.multiple_testset_2[0])
    
    def test_indexes_of_systems(self):
        self.assertListEqual(["Sys 1", "Sys 2", "Sys 3"], self.collection.indexes_of_systems())
    
    def test_names_of_systems(self):
        self.assertListEqual(["Sys A", "Sys B", "Sys C"], self.collection.names_of_systems())
    
    def test_system_A_id(self):
        self.assertEqual("Sys 1" , self.collection.system_name_id("Sys A"))
    
    def test_system_B_id(self):
        self.assertEqual("Sys 2" , self.collection.system_name_id("Sys B"))
    
    def test_system_C_id(self):
        self.assertEqual("Sys 3" , self.collection.system_name_id("Sys C"))
    
    def test_already_exists_A(self):
        self.assertTrue(self.collection.already_exists("Sys A"))
    
    def test_already_exists_B(self):
        self.assertTrue(self.collection.already_exists("Sys B"))
    
    def test_already_exists_C(self):
        self.assertTrue(self.collection.already_exists("Sys C"))
    
    def test_not_exists(self):
        self.assertFalse(self.collection.already_exists("Sys D"))

    def test_display_systems(self):
        text = "--> google.txt : Sys A \n--> unbabel_1.txt : Sys B \n--> unbabel_2.txt : Sys C \n"
        self.assertEqual(text, self.collection.display_systems())
    
    def test_source_language(self):
        self.assertEqual('fr', self.collection.source_language)
    
    def test_target_language(self):
        self.assertEqual('en', self.collection.target_language)
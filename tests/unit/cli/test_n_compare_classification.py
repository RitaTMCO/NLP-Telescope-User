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

from click.testing import CliRunner
from telescope.cli import n_compare_classification
from tests.data import DATA_PATH


class TestCompareCli(unittest.TestCase):

    system_1 = os.path.join(DATA_PATH, "class/sys1-class.txt")
    system_2 = os.path.join(DATA_PATH, "class/sys2-class.txt")
    system_3 = os.path.join(DATA_PATH, "class/ref-c.txt")
    src = os.path.join(DATA_PATH, "class/src-c.txt")
    ref = os.path.join(DATA_PATH, "class/ref-c.txt")
    sys_names_file = os.path.join(DATA_PATH, "systems_names.txt")
    labels = os.path.join(DATA_PATH, "class/all_labels.txt")
    sys_names = ["Sys A", "Sys B", "Sys C"]

    def setUp(self):
        self.runner = CliRunner()

    def test_with_seg_metric(self):
        args = [
            "-s",
            self.src,
            "-c",
            self.system_1,
            "-c",
            self.system_2,
            "-c",
            self.system_3,
            "-r",
            self.ref,
            "-l",
            self.labels,
            "-m",
            "F1-score",
            "--seg_metric",
            "Accuracy"
        ]
        result = self.runner.invoke(n_compare_classification, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_with_systems_names_file(self):
        args = [
            "-s",
            self.src,
            "-c",
            self.system_1,
            "-c",
            self.system_2,
            "-c",
            self.system_3,
            "-r",
            self.ref,
            "-l",
            self.labels,
            "-m",
            "F1-score",
            "--systems_names",
            self.sys_names_file
        ]
        result = self.runner.invoke(n_compare_classification, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_duplicates_filter(self):
        args = [            
            "-s",
            self.src,
            "-c",
            self.system_1,
            "-c",
            self.system_2,
            "-c",
            self.system_3,
            "-r",
            self.ref,
            "-l",
            self.labels,
            "-m",
            "F1-score",
            "-f",
            "duplicates"
        ]
        result = self.runner.invoke(n_compare_classification, args, catch_exceptions=False)
        self.assertIn("Filters Successfully applied. Corpus reduced in", result.stdout)
        self.assertEqual(result.exit_code, 0)


    def test_with_output(self):
        args = [
            "-s",
            self.src,
            "-c",
            self.system_1,
            "-c",
            self.system_2,
            "-c",
            self.system_3,
            "-r",
            self.ref,
            "-l",
            self.labels,
            "-m",
            "F1-score",
            "--seg_metric",
            "Accuracy",
            "-f",
            "duplicates",
            "--systems_names",
            self.sys_names_file,
            "--output_folder",
            DATA_PATH
        ]

        result = self.runner.invoke(n_compare_classification, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, self.ref.replace("/","_")  + "/results.csv")))
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, self.ref.replace("/","_")  + "/analysis-metric-score.png")))
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, self.ref.replace("/","_")  + "/Accuracy-analysis-labels-bucket.png")))
        for sys_name in self.sys_names:
            if sys_name != "Sys C":
                self.assertTrue(
                    os.path.isfile(os.path.join(DATA_PATH, 
                        self.ref.replace("/","_")  + "/" + sys_name + "/incorrect-examples.csv"))
                )
            self.assertTrue(
                os.path.isfile(os.path.join(DATA_PATH, 
                    self.ref.replace("/","_")  + "/" + sys_name + "/rates.csv"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(DATA_PATH, 
                    self.ref.replace("/","_")  + "/" + sys_name + "/confusion-matrix-" + sys_name.replace(" ", "_") + ".png"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(DATA_PATH, 
                    self.ref.replace("/","_")  + "/" + sys_name + "/singular_confusion_matrix/" + sys_name.replace(" ", "_") + "-label-positive.png"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(DATA_PATH, 
                    self.ref.replace("/","_")  + "/" + sys_name + "/singular_confusion_matrix/" + sys_name.replace(" ", "_") + "-label-negative.png"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(DATA_PATH, 
                    self.ref.replace("/","_")  + "/" + sys_name + "/singular_confusion_matrix/" + sys_name.replace(" ", "_") + "-label-neutral.png"))
            )


            os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/singular_confusion_matrix/" + sys_name.replace(" ", "_") + "-label-neutral.png")
            os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/singular_confusion_matrix/" + sys_name.replace(" ", "_") + "-label-negative.png")
            os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/singular_confusion_matrix/" + sys_name.replace(" ", "_") + "-label-positive.png")
            os.rmdir(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/singular_confusion_matrix/")
            if sys_name != "Sys C":
                os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/incorrect-examples.csv")
            os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/confusion-matrix-" + sys_name.replace(" ", "_") + ".png")
            os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/rates.csv")
            os.rmdir(DATA_PATH + "/" + self.ref.replace("/","_") + "/" + sys_name + "/" )
        os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/results.csv")
        os.remove(DATA_PATH + "/" + self.ref.replace("/","_")  + "/analysis-metric-score.png")
        os.remove(DATA_PATH + "/" + self.ref.replace("/","_") + "/Accuracy-analysis-labels-bucket.png")
        os.rmdir(DATA_PATH + "/" + self.ref.replace("/","_"))

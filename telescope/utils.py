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
import yaml
from io import StringIO

PATH_USER = "user/" 
PATH_DOWNLOADED_PLOTS = "user/downloaded_data/"


def telescope_cache_folder():
    if "HOME" in os.environ:
        cache_directory = os.environ["HOME"] + "/.cache/mt-telescope/"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
        return cache_directory
    else:
        raise Exception("HOME environment variable is not defined.")


def read_lines(file):
    if file is not None:
        file = StringIO(file.getvalue().decode())
        lines = [line.strip() for line in file.readlines()]
        return lines
    return None

def read_yaml_file(file_yaml):
    file = open(PATH_USER + file_yaml, "r")
    data = yaml.safe_load(file)
    file.close()
    return data

def create_downloaded_data_folder():
    if not os.path.exists(PATH_DOWNLOADED_PLOTS):
        os.makedirs(PATH_DOWNLOADED_PLOTS)
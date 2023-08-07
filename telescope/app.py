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
import streamlit as st
import requests
import os
from PIL import Image

from telescope.tasks import AVAILABLE_TASKS
from telescope.metrics.result import MultipleMetricResults
from telescope.testset import MultipleTestset
from telescope.bias_evaluation.gender_bias_evaluation import GenderBiasEvaluation
from telescope.plotting import export_dataframe, analysis_metrics
from telescope.universal_metrics import WeightedMean
from telescope.utils import PATH_DOWNLOADED_PLOTS, create_downloaded_data_folder

available_tasks = {t.name: t for t in AVAILABLE_TASKS}

@st.cache
def load_image(image_url):
    img = Image.open(requests.get(image_url, stream=True).raw)
    return img

#logo = load_image("https://github.com/Unbabel/MT-Telescope/blob/master/data/mt-telescope-logo.jpg?raw=true")
st.sidebar.image("data/nlp-telescope-logo.png")

create_downloaded_data_folder()

# --------------------  APP Settings --------------------

#---------- |NLP task| ------------
task = st.sidebar.selectbox(
    "Select the natural language processing task:",
    list(t.name for t in available_tasks.values()),
    index=0,
)

#---------- |Metrics| ------------
available_metrics = {m.name: m for m in available_tasks[task].metrics}

metrics = st.sidebar.multiselect(
    "Select the system-level metric you wish to run:",
    list(available_metrics.keys()),
    default=list(available_metrics.keys())[0],
)
if available_tasks[task].universal_metrics:
    rank = st.sidebar.checkbox('Rankings of Models')
    available_universal_metrics = available_tasks[task].universal_metrics
    
metric = st.sidebar.selectbox(
    "Select the segment-level metric you wish to run:",
    list(m.name for m in available_metrics.values() if m.segment_level),
    index=0,
)

#---------- |Filters| ------------
available_filters = {f.name: f for f in available_tasks[task].filters}

filters = st.sidebar.multiselect(
    "Select testset filters:", list(available_filters.keys()), default=list(available_filters.keys())[0],
    help = ("For named-entities, the following languages are available: ar, zh, nl, en, fr, de, ru and uk")
)

if "length" in available_filters:
    st.sidebar.subheader("Segment length constraints:")
    length_interval = st.sidebar.slider(
        "Specify the confidence interval for the length distribution:",
        0,
        100,
        step=5,
        value=(0, 100),
        help=(
            "In order to isolate segments according to caracter length "
            "we will create a sequence length distribution that you can constraint "
            "through it's density funcion. This slider is used to specify the confidence interval P(a < X < b)"
        ),
    )
    if length_interval != (0, 100) and "length" not in filters:
        filters = (
            filters
            + [
                "length",
            ]
            if filters is not None
            else [
                "length",
            ]
        ) 

#---------- |Bootstrap resampling| ------------    

if available_tasks[task].bootstrap:
    st.sidebar.subheader("Bootstrap resampling settings:")
    num_samples = st.sidebar.number_input(
        "Number of random partitions:",
        min_value=1,
        max_value=1000,
        value=300,
        step=50,
    )
    sample_ratio = st.sidebar.slider(
        "Proportion (P) of the initial sample:", 0.0, 1.0, value=0.5, step=0.1
    )


#---------- |Bias Evaluation| ------------ 

available_bias_evaluations = {b.name: b for b in available_tasks[task].bias_evaluations}

if available_bias_evaluations:
    st.sidebar.subheader("Bias Evaluations:")

    option_bias_evaluation = ""

    bias_evaluations = st.sidebar.multiselect("Select Bias Evaluations:", list(available_bias_evaluations.keys()))  

    option_bias_evaluation = st.sidebar.selectbox(
            "Select how you want the evaluation to be done:",
            GenderBiasEvaluation.options_bias_evaluation,
            index=0,
            disabled = ("Gender" not in bias_evaluations)
        ) 

     
# --------------------- Streamlit APP Caching functions! --------------------------

cache_time = 10 * 60 * 60  # 10 hour cache time for each object
cache_max_entries = 200  # 10 hour cache time for each object


# --------| Filters |-----------
@st.cache(
    hash_funcs={MultipleTestset: MultipleTestset.hash_func},
    suppress_st_warning=True,
    show_spinner=False,
    allow_output_mutation=True,
    ttl=cache_time,
    max_entries=cache_max_entries,
)
def apply_filters(testset, filters, ref_name, source_language, target_language):
    for filter in filters:
        with st.spinner(f"Applying {filter} filter for reference {ref_name}..." ):
            if filter == "length":
                testset.apply_filter(available_filters[filter](testset, *length_interval))
            elif filter == "named-entities":
                testset.apply_filter(available_filters[filter](testset, source_language,target_language))
            else:
                testset.apply_filter(available_filters[filter](testset))

    # HACK
    # I'll add a new prefix to all testset filenames to "fool" streamlit cache
    if "length" in filters:
        filter_prefix = " ".join([f for f in filters]) + str(length_interval)
    else:
        filter_prefix = " ".join([f for f in filters])
    testset.filenames = [filter_prefix + f for f in testset.filenames]
    return testset



# --------| Metrics |-----------
def hash_metrics(metrics):
    return " ".join([m.name for m in metrics])

@st.cache(
    hash_funcs={MultipleTestset: MultipleTestset.hash_func},
    show_spinner=False,
    allow_output_mutation=True,
    ttl=cache_time,
    max_entries=cache_max_entries,
)
def run_metric(testset, metric, ref_filename, language, labels):
    with st.spinner(f"Running {metric} for reference {ref_filename}..."):
        if labels == [" "]:
            metric = available_metrics[metric](language=language)
        else:
            metric = available_metrics[metric](labels=labels)
        return metric.multiple_comparison(testset)


def run_all_metrics(collection, metrics, filters):
    refs_names = collection.refs_names
    labels = [" "]
    target_language = "X"
    source_language = "X"

    if collection.task == "classification":
        labels = collection.labels
    else:
        source_language = collection.source_language
        target_language = collection.target_language

    if filters:
        for ref_name in refs_names:
            testset = collection_testsets.testsets[ref_name]
            corpus_size = len(testset)
            collection_testsets.testsets[ref_name] = apply_filters(testset,filters,ref_name,
                                                                source_language,
                                                                target_language)
            st.success("Corpus reduced in {:.2f}%".format(
                (1 - (len(collection_testsets.testsets[ref_name]) / corpus_size)) 
                    * 100) + " for reference " + ref_name)

    if any(len(collection_testsets.testsets[ref_name]) == 0 for ref_name in refs_names):
        return {}
    return {
        ref_name: {metric: run_metric(
                            collection_testsets.testsets[ref_name], 
                            metric, 
                            ref_name,
                            target_language,
                            labels) 
        for metric in metrics}
        for ref_name in refs_names
        }


# --------| Universal Metric |--------

@st.cache(
    hash_funcs={MultipleTestset: MultipleTestset.hash_func},
    show_spinner=False,
    allow_output_mutation=True,
    ttl=cache_time,
    max_entries=cache_max_entries,
)
def run_universal_metric(testset, universal_metric, metrics_results, extra_info):
    if universal_metric == "pairwise-comparison":
        universal_metric = available_universal_metrics[universal_metric](metrics_results,extra_info[0],extra_info[1])
    elif available_universal_metrics[universal_metric] == WeightedMean:
        universal_metric = available_universal_metrics[universal_metric](metrics_results,universal_metric)
    else:
        universal_metric = available_universal_metrics[universal_metric](metrics_results)
    return universal_metric.universal_score_calculation_and_ranking(testset)


# --------| Bias Evaluation |-----------

@st.cache(
    hash_funcs={MultipleTestset: MultipleTestset.hash_func},
    show_spinner=False,
    allow_output_mutation=True,
    ttl=cache_time,
    max_entries=cache_max_entries,
)
def run_bias_evalutaion(testset, evaluation, ref_filename, language):
    with st.spinner(f"Running {evaluation} Bias Evaluation for reference {ref_filename}..."):
        bias_evaluation = available_bias_evaluations[evaluation](language)
        return bias_evaluation.evaluation(testset,option_bias_evaluation)

def run_all_bias_evalutaions(collection):
    refs_names = collection.refs_names
    language = collection_testsets.target_language
    # {gender_bias_evaluation:{Ref 1: MultipleBiasResults, Ref 2: MultipleBiasResults}, ....}
    return {
        evaluation: { 
            ref_name: run_bias_evalutaion(collection_testsets.testsets[ref_name], 
                                          evaluation,
                                          ref_name,
                                          language)
            for ref_name in refs_names
            }
        for evaluation in bias_evaluations
    }


# --------| Rename systems |-----------
@st.cache
def rename_system(system_name,sys_id):
    st.session_state[sys_id + "_name"] = system_name
    collection_testsets.systems_names[sys_id] = st.session_state[sys_id + "_name"]


# --------------------  APP  --------------------

st.title("Welcome to NLP-Telescope! :telescope:")

collection_testsets = available_tasks[task].input_web_interface()

if collection_testsets:
    if metric not in metrics:
        metrics = [
            metric,
        ] + metrics

    if available_tasks[task].bias_evaluations and bias_evaluations:
        bias_results_per_evaluation = run_all_bias_evalutaions(collection_testsets)
    metrics_results_per_ref = run_all_metrics(collection_testsets, metrics, filters)


    # ----------------------------| Informations About The Systems |-------------------------------

    st.write("---")
    st.title("Informations About The Systems")

    st.subheader(":blue[Rename Systems]")
    system_filename = st.selectbox(
        "**Select the System Filename**",
        list(collection_testsets.systems_ids.keys()),
        index=0)
    sys_id = collection_testsets.systems_ids[system_filename]
    system_name = st.text_input('Enter the system name')
    if (collection_testsets.systems_names[sys_id] != system_name and collection_testsets.already_exists(system_name)):
        st.warning("This system name already exists")
    elif system_name:
        st.session_state[task + "_" + sys_id + "_rename"] = system_name
        collection_testsets.systems_names[sys_id] = st.session_state[task + "_" + sys_id + "_rename"]

    st.subheader(":blue[Systems Names]" )
    st.text(collection_testsets.display_systems())


    # ----------------------------| Select the Reference |-----------------------------------

    st.write("---")
    st.title("Reference")
    ref_filename = st.selectbox(
        "**:blue[Select the Reference:]**",
        collection_testsets.refs_names,
        index=0,
    )
    st.text("Reference: " + ref_filename)

    # ----------------------------| Ranking Models |-------------------------------

    if available_tasks[task].universal_metrics and rank and len(collection_testsets.systems_names) > 1 and metrics_results_per_ref:
        st.write("---")
        st.title("Models Rankings")
        st.text("\n\n\n")

        universal_results = None
        universal_metric = st.selectbox("**:blue[Select Univeral Metric:]**", list(available_universal_metrics.keys())) 
        if universal_metric == "pairwise-comparison":
            left, right = st.columns(2)
            system_x_name = left.selectbox(
                "Select the system a:",
                collection_testsets.names_of_systems(),
                index=0,
                key = "pairwise_1"
            )
            system_y_name = right.selectbox(
                "Select the system b:",
                collection_testsets.names_of_systems(),
                index=1,
                key = "pairwise_2"
            )

            if system_x_name == system_y_name:
                st.warning("The system x cannot be the same as system y")
            else:
                system_x_id = collection_testsets.system_name_id(system_x_name)
                system_y_id = collection_testsets.system_name_id(system_y_name)
                universal_results = run_universal_metric(collection_testsets.testsets[ref_filename], universal_metric, metrics_results_per_ref[ref_filename], 
                                                         [system_x_id,system_y_id])
        else:
            universal_results = run_universal_metric(collection_testsets.testsets[ref_filename], universal_metric, metrics_results_per_ref[ref_filename], [])

        if universal_results:
            universal_results.plots_web_interface(collection_testsets,ref_filename)

    # ----------------------------| Metric Analysis and Plots |-------------------------------

    if metrics_results_per_ref:
        st.write("---")
        st.title("Metric Analysis")
        st.text("\n\n\n")

        metrics_results = metrics_results_per_ref[ref_filename] 

        if len(metrics_results) > 0:
            left,right = st.columns([0.3, 0.7])
            left_2,right_2 = st.columns([0.3, 0.7])
            path = PATH_DOWNLOADED_PLOTS  + collection_testsets.task + "/" + collection_testsets.src_name + "/" + ref_filename + "/"  
        
            dataframe = MultipleMetricResults.results_to_dataframe(list(metrics_results.values()),collection_testsets.systems_names)
            left.dataframe(dataframe)
            analysis_metrics(list(metrics_results.values()),collection_testsets.systems_names,column=right)

            left_2,right_2 = st.columns([0.3, 0.7]) 
             
            export_dataframe(label="Export table with score",path=path, name= "results.csv", dataframe=dataframe, column=left_2)
            if right_2.button('Download the analysis of each metric'):
                analysis_metrics(list(metrics_results.values()),collection_testsets.systems_names, path)
    
    
        if metric in metrics_results:
            if available_tasks[task].bootstrap:
                available_tasks[task].plots_web_interface(metric, metrics_results, collection_testsets, ref_filename, metrics, available_metrics, 
                                                      num_samples, sample_ratio)
            else:
                available_tasks[task].plots_web_interface(metric, metrics_results, collection_testsets, ref_filename)
    
    # ----------------------------| Bias Evaluation |-------------------------------

    if available_tasks[task].bias_evaluations and bias_evaluations:
        st.write("---")
        st.title("Bias Evaluation")
        for evaluation in bias_evaluations:
            st.header(":blue[" + evaluation + " Bias Evaluation:]")
            multiple_bias_results = bias_results_per_evaluation[evaluation][ref_filename]
            multiple_bias_results.plots_bias_results_web_interface(collection_testsets,ref_filename,option_bias_evaluation)
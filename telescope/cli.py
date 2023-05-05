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
r"""
MT-Telescope command line interface (CLI)
==============
Main commands:
    - score     Used to download Machine Translation metrics.
"""
from typing import List, Union, Tuple
import os
import click
import json
import pandas as pd

from telescope.metrics import AVAILABLE_METRICS, AVAILABLE_CLASSIFICATION_METRICS, AVAILABLE_MT_METRICS, PairwiseResult
from telescope.filters import AVAILABLE_FILTERS, AVAILABLE_CLASSIFICATION_FILTERS
from telescope.tasks import AVAILABLE_NLG
from telescope.metrics.result import MultipleResult
from telescope.testset import PairwiseTestset, MultipleTestset
from telescope.collection_testsets import NLGTestsets, ClassTestsets
from telescope.plot import ClassificationPlot, NLGPlot
from telescope.plotting import (
    plot_segment_comparison,
    plot_pairwise_distributions,
    plot_bucket_comparison,
)

available_metrics = {m.name: m for m in AVAILABLE_METRICS}
available_mt_metrics = {m.name: m for m in AVAILABLE_MT_METRICS}
available_class_metrics = {m.name: m for m in AVAILABLE_CLASSIFICATION_METRICS}
available_filters = {f.name: f for f in AVAILABLE_FILTERS}
available_class_filters = {f.name: f for f in AVAILABLE_CLASSIFICATION_FILTERS}
available_nlg_tasks = {t.name: t for t in AVAILABLE_NLG}


def readlines(ctx, param, file: click.File) -> List[str]:
    return [l.strip() for l in file.readlines()]


def output_folder_exists(ctx, param, output_folder):
    if output_folder != "" and not os.path.exists(output_folder):
        raise click.BadParameter(f"{output_folder} does not exist!")
    return output_folder


@click.group()
def telescope():
    pass


@telescope.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
)
@click.option(
    "--system_x",
    "-x",
    required=True,
    help="System X MT outputs.",
    type=click.File(),
)
@click.option(
    "--system_y",
    "-y",
    required=True,
    help="System Y MT outputs.",
    type=click.File(),
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments.",
    type=click.File(),
)
@click.option(
    "--language",
    "-l",
    required=True,
    help="Language of the evaluated text.",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(list(available_mt_metrics.keys())),
    required=True,
    multiple=True,
    help="MT metric to run.",
)
@click.option(
    "--filter",
    "-f",
    type=click.Choice(list(available_filters.keys())),
    required=False,
    default=[],
    multiple=True,
    help="MT metric to run.",
)
@click.option(
    "--length_min_val",
    type=float,
    required=False,
    default=0.0,
    help="Min interval value for length filtering.",
)
@click.option(
    "--length_max_val",
    type=float,
    required=False,
    default=0.0,
    help="Max interval value for length filtering.",
)
@click.option(
    "--seg_metric",
    type=click.Choice([m.name for m in available_mt_metrics.values() if m.segment_level]),
    required=False,
    default="COMET",
    help="Segment-level metric to use for segment-level analysis.",
)
@click.option(
    "--output_folder",
    "-o",
    required=False,
    default="",
    callback=output_folder_exists,
    type=str,
    help="Folder you wish to use to save plots.",
)
@click.option("--bootstrap", is_flag=True)
@click.option(
    "--num_splits",
    required=False,
    default=300,
    type=int,
    help="Number of random partitions used in Bootstrap resampling.",
)
@click.option(
    "--sample_ratio",
    required=False,
    default=0.5,
    type=float,
    help="Folder you wish to use to save plots.",
)
def compare(
    source: click.File,
    system_x: click.File,
    system_y: click.File,
    reference: click.File,
    language: str,
    metric: Union[Tuple[str], str],
    filter: Union[Tuple[str], str],
    length_min_val: float,
    length_max_val: float,
    seg_metric: str,
    output_folder: str,
    bootstrap: bool,
    num_splits: int,
    sample_ratio: float,
):
    testset = PairwiseTestset(
        src=[l.strip() for l in source.readlines()],
        system_x=[l.strip() for l in system_x.readlines()],
        system_y=[l.strip() for l in system_y.readlines()],
        ref=[l.strip() for l in reference.readlines()],
        language_pair="X-" + language,
        filenames=[source.name, system_x.name, system_y.name, reference.name],
    )
    corpus_size = len(testset)
    if filter:
        filters = [available_filters[f](testset) for f in filter if (f != "length" and f!= "named-entities")]
        if "length" in filter:
            filters.append(available_filters["length"](testset, int(length_min_val*100), int(length_max_val*100)))
        elif "named-entities":
            filters.append(available_filters["named-entities"](testset, testset.source_language,
                                                            testset.target_language)) 


        for filter in filters:
            testset.apply_filter(filter)

        if (1 - (len(testset) / corpus_size)) * 100 == 100:
            click.secho("The current filters reduce the Corpus on 100%!", fg="ref")
            return
    
        click.secho(
            "Filters Successfully applied. Corpus reduced in {:.2f}%.".format(
                (1 - (len(testset) / corpus_size)) * 100
            ),
            fg="green",
        )

    if seg_metric not in metric:
        metric = tuple(
            [
                seg_metric,
            ]
            + list(metric)
        )
    else:
        # Put COMET in first place
        metric = list(metric)
        metric.remove(seg_metric)
        metric = tuple(
            [
                seg_metric,
            ]
            + metric
        )

    results = {
        m: available_metrics[m](language=testset.target_language).pairwise_comparison(
            testset
        )
        for m in metric
    }

    # results_dict = PairwiseResult.results_to_dict(list(results.values()))
    results_df = PairwiseResult.results_to_dataframe(list(results.values()))
    if bootstrap:
        bootstrap_results = []
        for m in metric:
            bootstrap_result = available_metrics[m].bootstrap_resampling(
                testset, num_splits, sample_ratio, results[m]
            )
            bootstrap_results.append(
                available_metrics[m]
                .bootstrap_resampling(testset, num_splits, sample_ratio, results[m])
                .stats
            )
        bootstrap_results = {
            k: [dic[k] for dic in bootstrap_results] for k in bootstrap_results[0]
        }
        for k, v in bootstrap_results.items():
            results_df[k] = v

    click.secho(str(results_df), fg="yellow")
    if output_folder != "":
        if not output_folder.endswith("/"):
            output_folder += "/"
        results_df.to_json(output_folder + "results.json", orient="index", indent=4)
        plot_segment_comparison(results[seg_metric], output_folder)
        plot_pairwise_distributions(results[seg_metric], output_folder)
        plot_bucket_comparison(results[seg_metric], output_folder)


@telescope.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
    callback=readlines,
)
@click.option(
    "--translation",
    "-t",
    required=True,
    help="MT outputs.",
    type=click.File(),
    callback=readlines,
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments.",
    type=click.File(),
    callback=readlines,
)
@click.option(
    "--language",
    "-l",
    required=True,
    help="Language of the evaluated text.",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(list(available_mt_metrics.keys())),
    required=True,
    multiple=True,
    help="MT metric to run.",
)
def score(
    source: List[str],
    translation: List[str],
    reference: List[str],
    language: str,
    metric: Union[Tuple[str], str],
):
    metrics = metric
    for metric in metrics:
        if not available_metrics[metric].language_support(language):
            raise click.ClickException(f"{metric} does not support '{language}'")
    results = []
    for metric in metrics:
        metric = available_metrics[metric](language)
        results.append(metric.score(source, translation, reference))

    for result in results:
        click.secho(str(result), fg="yellow")


@telescope.command()
@click.pass_context
def streamlit(ctx):
    file_path = os.path.realpath(__file__)
    script_path = "/".join(file_path.split("/")[:-1]) + "/app.py"
    os.system("streamlit run " + script_path)


###################################################
############|Commands for N systems|################
###################################################

def seg_metric_in_metrics(seg_metric, metrics):
    if seg_metric not in metrics:
        metrics = tuple(
            [
                seg_metric,
            ]
            + list(metrics)
        )
    else:
        metrics = list(metrics)
        metrics.remove(seg_metric)
        metrics = tuple(
            [
                seg_metric,
            ]
            + metrics
        )
    return metrics

def apply_filter(collection,filter,length_min_val,length_max_val):
    for ref_name in collection.refs_names:
        corpus_size = len(collection.testsets[ref_name])
        
        for f in filter:
            if f == "length":
                fil = available_filters[f](collection.testsets[ref_name], int(length_min_val*100), 
                                    int(length_max_val*100))
            elif f == "named-entities":
                fil = available_filters[f](collection.testsets[ref_name], collection.source_language, 
                                                                            collection.target_language)
            else:
                fil = available_filters[f](collection.testsets[ref_name])
            collection.testsets[ref_name].apply_filter(fil)

        if (1 - (len(collection.testsets[ref_name]) / corpus_size)) * 100 == 100:
            click.secho("For reference " + ref_name + ", the current filters reduce the Corpus on 100%!", fg="green")
            return
    
        click.secho( "Filters Successfully applied. Corpus reduced in {:.2f}%.".format(
            (1 - (len(collection.testsets[ref_name]) / corpus_size)) * 100) + " for reference " + ref_name,
                fg="green" )

def display_table(collection, ref_filename, systems_names, results):
    results_dicts = MultipleResult.results_to_dict(list(results.values()), systems_names)

    click.secho('Reference: ' + ref_filename, fg="yellow")

    for met, systems in results_dicts.items():
        click.secho("metric: " + str(met), fg="yellow")
        click.secho("\t" + "systems:", fg="yellow")
        for sys_name, sys_score in systems.items():
            click.secho("\t" + str(sys_name) + ": " + str(sys_score), fg="yellow")

    return results_dicts

def bootstrap_result(collection,ref_filename,results,metric,system_x,system_y,num_splits,sample_ratio):

    testset = collection.testsets[ref_filename]
    bootstrap_results = []

    for m in metric:
        bootstrap_results.append(
            available_metrics[m]
            .multiple_bootstrap_resampling(testset, num_splits, sample_ratio, system_x,
                                    system_y, collection.target_language, results[m])
            .stats)
            
    bootstrap_results = {k: [dic[k] for dic in bootstrap_results] for k in bootstrap_results[0]}      
    bootstrap_df = pd.DataFrame.from_dict(bootstrap_results)
    bootstrap_df.index = metric
        
    click.secho("\nBootstrap resampling results:", fg="yellow")
    click.secho(str(bootstrap_df), fg="yellow")

    return bootstrap_df


@telescope.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
)
@click.option(
    "--system_output",
    "-c",
    required=True,
    help="System candidate. This option can be multiple.",
    type=click.File(),
    multiple=True,
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments. This option can be multiple.",
    type=click.File(),
    multiple=True,
)
@click.option(
    "--task",
    "-t",
    type=click.Choice(list(available_nlg_tasks.keys())),
    required=True,
    help="NLG to evaluate.",
)
@click.option(
    "--language",
    "-l",
    required=True,
    help="Language of the evaluated text.",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(list(available_metrics.keys())),
    required=True,
    multiple=True,
    help="Metric to run. This option can be multiple.",
)
@click.option(
    "--filter",
    "-f",
    type=click.Choice(list(available_filters.keys())),
    required=False,
    default=[],
    multiple=True,
    help="Filter to run. This option can be multiple.",
)
@click.option(
    "--length_min_val",
    type=float,
    required=False,
    default=0.0,
    help="Min interval value for length filtering.",
)
@click.option(
    "--length_max_val",
    type=float,
    required=False,
    default=0.0,
    help="Max interval value for length filtering.",
)
@click.option(
    "--seg_metric",
    type=click.Choice([m.name for m in available_metrics.values() if m.segment_level]),
    required=False,
    default="BERTScore",
    help="Segment-level metric to use for segment-level analysis.",
)
@click.option(
    "--output_folder",
    "-o",
    required=False,
    default="",
    callback=output_folder_exists,
    type=str,
    help="Folder you wish to use to save plots.",
)
@click.option("--bootstrap", is_flag=True)
@click.option(
    "--system_x",
    "-x",
    required=False,
    help="System X NLG outputs for segment-level comparison and bootstrap resampling.",
    type=click.File(),
)
@click.option(
    "--system_y",
    "-y",
    required=False,
    help="System Y NLG outputs for segment-level comparison and bootstrap resampling.",
    type=click.File(),
)
@click.option(
    "--num_splits",
    required=False,
    default=300,
    type=int,
    help="Number of random partitions used in Bootstrap resampling.",
)
@click.option(
    "--sample_ratio",
    required=False,
    default=0.5,
    type=float,
    help="Proportion (P) of the initial sample.",
)
@click.option(
    "--systems_names",
    "-n",
    required=False,
    type=click.File(),
    help="File that contains the names of the systems per line.",
)
def n_compare_nlg(
    source: click.File,
    system_output: Tuple[click.File],
    reference: Tuple[click.File],
    task: str,
    language: str,
    metric: Union[Tuple[str], str],
    filter: Union[Tuple[str], str],
    length_min_val: float,
    length_max_val: float,
    seg_metric: str,
    output_folder: str,
    bootstrap: bool,
    num_splits: int,
    sample_ratio: float,
    system_x: click.File,
    system_y: click.File,
    systems_names: click.File
):  
    collection = NLGTestsets.read_data_cli(source,systems_names,system_output,reference,language)

    click.secho("Systems:\n" + collection.display_systems(), fg="bright_blue")

    if filter:
        apply_filter(collection,filter,length_min_val,length_max_val)   

    metric = seg_metric_in_metrics(seg_metric,metric)
    systems_index = collection.systems_indexes
    systems_names =  collection.systems_names
    language = collection.language_pair.split("-")[1]

    for ref_filename in collection.refs_names:
        testset = collection.testsets[ref_filename]
        results = {
            m: available_metrics[m](language=language).multiple_comparison(testset) 
            for m in metric }

        results_dicts = display_table(collection,ref_filename,systems_names,results)

        if bootstrap and len(systems_index.values()) > 1: 
            if system_x.name in systems_index and system_y.name in systems_index:
                x_id = systems_index[system_x.name]
                y_id = systems_index[system_y.name]
        
            else:
                x_id = collection.indexes_of_systems()[0]
                y_id = collection.indexes_of_systems()[1]

            bootstrap_df = bootstrap_result(collection,ref_filename,results,metric,x_id,y_id,num_splits,sample_ratio)
        
        if output_folder != "":
            if not output_folder.endswith("/"):
                output_folder += "/"    
            saving_dir = output_folder + ref_filename.replace("/","_") + "/"
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            with open(saving_dir + "results.json", "w") as result_file:
                json.dump(results_dicts, result_file, indent=4)

            if bootstrap and len(systems_index.values()) > 1:
                x_name = systems_names[x_id]
                y_name = systems_names[y_id]
                filename = saving_dir + x_name + "-" + y_name + "_bootstrap_results.json"
                bootstrap_df.to_json(filename, orient="index", indent=4)

            plot = NLGPlot(seg_metric,metric,available_metrics,results,collection,ref_filename,task,num_splits,sample_ratio)
            plot.display_plots_cli(saving_dir,system_x,system_y)


@telescope.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
)
@click.option(
    "--system_output",
    "-c",
    required=True,
    help="System candidate. This option can be multiple.",
    type=click.File(),
    multiple=True,
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments. This option can be multiple.",
    type=click.File(),
    multiple=True,
)
@click.option(
    "--label",
    "-l",
    required=True,
    multiple=True,
    help="Existing labels"
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(list(available_class_metrics.keys())),
    required=True,
    multiple=True,
    help="Metric to run. This option can be multiple.",
)
@click.option(
    "--filter",
    "-f",
    type=click.Choice(list(available_class_filters.keys())),
    required=False,
    default=[],
    multiple=True,
    help="Filter to run. This option can be multiple.",
)
@click.option(
    "--seg_metric",
    type=click.Choice([m.name for m in available_class_metrics.values() if m.segment_level]),
    required=False,
    default="Accuracy",
    help="Segment-level metric to use for segment-level analysis.",
)
@click.option(
    "--output_folder",
    "-o",
    required=False,
    default="",
    callback=output_folder_exists,
    type=str,
    help="Folder you wish to use to save plots.",
)
@click.option(
    "--systems_names",
    "-n",
    required=False,
    type=click.File(),
    help="File that contains the names of the systems per line.",
)
def n_compare_classification(
    source: click.File,
    system_output: Tuple[click.File],
    reference: Tuple[click.File],
    label: Union[Tuple[str], str],
    metric: Union[Tuple[str], str],
    filter: Union[Tuple[str], str],
    seg_metric: str,
    output_folder: str,
    systems_names: click.File
):  
    collection = ClassTestsets.read_data_cli(source,systems_names,system_output,reference,list(label))

    click.secho("Systems:\n" + collection.display_systems(), fg="bright_blue")

    if filter:
        apply_filter(collection,filter,0,0)   

    metric = seg_metric_in_metrics(seg_metric,metric)

    systems_index = collection.systems_indexes
    systems_names = collection.systems_names

    labels = collection.labels
    for ref_filename in collection.refs_names:
        testset = collection.testsets[ref_filename]
        results = {
            m: available_metrics[m](labels=labels).multiple_comparison(testset) 
            for m in metric 
        }

        results_dicts = display_table(collection,ref_filename,systems_names,results)

        if output_folder != "":
            if not output_folder.endswith("/"):
                output_folder += "/"    
            saving_dir = output_folder + ref_filename.replace("/","_") + "/"
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            with open(saving_dir + "results.json", "w") as result_file:
                json.dump(results_dicts, result_file, indent=4)
            
            plot = ClassificationPlot(seg_metric,metric,available_class_metrics,results,
                collection, ref_filename, "classification")
            
            plot.display_plots_cli(saving_dir)
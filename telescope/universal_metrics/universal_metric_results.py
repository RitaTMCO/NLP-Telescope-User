import click
import streamlit as st
import numpy as np
import pandas as pd

from typing import List, Dict
from telescope.utils import PATH_DOWNLOADED_PLOTS
from telescope.collection_testsets import CollectionTestsets
from telescope.metrics.metric import Metric
from telescope.plotting import export_dataframe

class UniversalMetricResult():
    def __init__(
        self,
        ref: List[str],
        system_output: List[str],
        metrics: List[Metric],
        universal_metric: str,
        title:str,
        rank: int,
        universal_score: float,
    ) -> None:
        self.ref = ref
        self.system_output = system_output
        self.metrics = metrics
        self.universal_metric = universal_metric
        self.title = title
        self.rank = rank
        self.universal_score = universal_score

class MultipleUniversalMetricResult():
    def __init__(
        self,
        systems_universal_metrics_results: Dict[str,UniversalMetricResult], # {id_of_systems:UniversalMetricResult}
    ) -> None:
        ref_x = list(systems_universal_metrics_results.values())[0].ref
        metrics_x = list(systems_universal_metrics_results.values())[0].metrics
        universal_metric_x = list(systems_universal_metrics_results.values())[0].universal_metric
        title_x = list(systems_universal_metrics_results.values())[0].title

        for universal_metric_results in list(systems_universal_metrics_results.values()):
            assert universal_metric_results.ref == ref_x
            assert universal_metric_results.metrics == metrics_x
            assert universal_metric_results.universal_metric == universal_metric_x
            assert universal_metric_results.title == title_x
        
        self.ref = ref_x
        self.metrics = metrics_x
        self.universal_metric = universal_metric_x
        self.title = title_x
        self.systems_universal_metrics_results = systems_universal_metrics_results

    def results_to_dataframe(self,systems_names:Dict[str, str]) -> pd.DataFrame:
        summary = []
        ranks = []
        for sys_id, universal_metric_result in self.systems_universal_metrics_results.items():
            summary.append([systems_names[sys_id], universal_metric_result.universal_score])
            ranks.append("rank " + str(universal_metric_result.rank))

        df = pd.DataFrame(np.array(summary), index=ranks, columns=["System", "Score"])
        return df
    
    def plots_web_interface(self, collection_testsets:CollectionTestsets,ref_filename:str):
        st.subheader(self.title)
        path = PATH_DOWNLOADED_PLOTS  + collection_testsets.task + "/" + collection_testsets.src_name + "/" +  ref_filename + "/" 
        df = self.results_to_dataframe(collection_testsets.systems_names)
        st.dataframe(df)
        export_dataframe(label="Export ranks of systems", path=path, name= self.universal_metric + "_ranks_systems.csv", dataframe=df)
    
    def plots_cli_interface(self, collection_testsets:CollectionTestsets):
        click.secho("\nModels Rankings:", fg="yellow")
        df = self.results_to_dataframe(collection_testsets.systems_names)
        click.secho(str(df), fg="yellow")
        return df
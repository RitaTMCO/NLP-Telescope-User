import abc
import os
import streamlit as st
import numpy as np

from typing import List, Dict
from telescope.collection_testsets import CollectionTestsets, NLGTestsets, ClassTestsets
from telescope.plotting import (
    plot_bootstraping_result,
    plot_bucket_multiple_comparison,
    plot_multiple_distributions,
    plot_multiple_segment_comparison,
    overall_confusion_matrix_table,
    singular_confusion_matrix_table,
    analysis_labels,
    incorrect_examples,
    analysis_extractive_summarization
)

class Plot(metaclass=abc.ABCMeta):
    def __init__(
        self,
        metric: str, 
        metrics: List[str],
        available_metrics: dict,
        results: dict,
        collection_testsets: CollectionTestsets,
        ref_filename: str,
        task: str
    ) -> None:

        self.metric = metric
        self.metrics = metrics
        self.available_metrics = available_metrics
        self.results = results
        self.collection_testsets = collection_testsets
        self.ref_filename = ref_filename
        self.task = task

    @abc.abstractmethod
    def display_plots(self) -> None:
        pass
    
    @abc.abstractmethod
    def display_plots_cli(self, saving_dir:str, *args) -> None:
        pass


class NLGPlot(Plot):
    def __init__(
        self,
        metric:str, 
        metrics: List[str],
        available_metrics: dict,
        results:dict, 
        collection_testsets: NLGTestsets,
        ref_filename: str,
        task: str,
        num_samples: float,
        sample_ratio: float,
    ) -> None:

        super().__init__(metric, metrics, available_metrics, results, collection_testsets, ref_filename, task)
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio


    def display_plots(self) -> None:
        if self.metric == "COMET" or self.metric == "BERTScore":
            st.header(":blue[Error-type analysis:]")
            plot_bucket_multiple_comparison(self.results[self.metric], 
                                                        self.collection_testsets.names_of_systems())

        if len(self.collection_testsets.testsets[self.ref_filename]) > 1:
            try:
                st.header(":blue[Segment-level scores histogram:]")
                plot_multiple_distributions(self.results[self.metric],
                                                        self.collection_testsets.names_of_systems())
            except np.linalg.LinAlgError as err:    
                st.write(err)

        if len(self.results[self.metric].systems_metric_results) > 1:
            st.header(":blue[Pairwise comparison:]")

            left, right = st.columns(2)
            system_x_name = left.selectbox(
                "Select the system x:",
                self.collection_testsets.names_of_systems(),
                index=0,
                key = self.ref_filename + "_1"
            )
            system_y_name = right.selectbox(
                "Select the system y:",
                self.collection_testsets.names_of_systems(),
                index=1,
                key = self.ref_filename + "_2"
            )
            if system_x_name == system_y_name:
                st.warning("The system x cannot be the same as system y")
            
            else:
                system_x_id = self.collection_testsets.system_name_id(system_x_name)
                system_x = [system_x_id, system_x_name]
                system_y_id = self.collection_testsets.system_name_id(system_y_name)
                system_y = [system_y_id, system_y_name]

                st.subheader("Segment-level comparison:")
                if self.task == "machine translation":
                    plot_multiple_segment_comparison(self.results[self.metric],system_x,system_y,True)
                else:
                    plot_multiple_segment_comparison(self.results[self.metric],system_x,system_y)

                #Bootstrap Resampling
                _, middle, _ = st.columns(3)
                if middle.button("Perform Bootstrap Resampling",key = self.ref_filename):
                    st.warning(
                        "Running metrics for {} partitions of size {}".format(
                            self.num_samples, self.sample_ratio * len(self.collection_testsets.testsets[self.ref_filename])
                        )
                    )
                    st.subheader("Bootstrap resampling results:")
                    with st.spinner("Running bootstrap resampling..."):
                        for self.metric in self.metrics:
                            bootstrap_result = self.available_metrics[self.metric].multiple_bootstrap_resampling(
                                self.collection_testsets.testsets[self.ref_filename], int(self.num_samples), 
                                self.sample_ratio, system_x_id, system_y_id, self.collection_testsets.target_language, self.results[self.metric])

                            plot_bootstraping_result(bootstrap_result)


    def display_plots_cli(self, saving_dir:str, system_x:str, system_y:str) -> None:
        
        if self.metric == "COMET" or self.metric == "BERTScore":
            plot_bucket_multiple_comparison(self.results[self.metric], self.collection_testsets.names_of_systems(), 
                                    saving_dir)
        
        if len(self.collection_testsets.testsets[self.ref_filename]) > 1:
            plot_multiple_distributions(self.results[self.metric], self.collection_testsets.names_of_systems(),
                                    saving_dir)
        
        if len(self.collection_testsets.systems_indexes.values()) > 1: 
            if ((system_x.name in self.collection_testsets.systems_indexes) 
                and (system_y.name in list(self.collection_testsets.systems_indexes))):
                x_id = self.collection_testsets.systems_indexes[system_x.name]
                y_id = self.collection_testsets.systems_indexes[system_y.name]
            
            else:
                x_id = self.collection_testsets.indexes_of_systems()[0]
                y_id = self.collection_testsets.indexes_of_systems()[1]

            x = [x_id,self.collection_testsets.systems_names[x_id]]
            y = [y_id,self.collection_testsets.systems_names[y_id]]
            if self.task == "machine-translation":
                plot_multiple_segment_comparison(self.results[self.metric],x,y,True,saving_dir)
            else:
                plot_multiple_segment_comparison(self.results[self.metric],x,y,saving_dir=saving_dir)

class ClassificationPlot(Plot):
    def __init__(
        self,
        metric:str, 
        metrics: List[str],
        available_metrics: dict,
        results:dict, 
        collection_testsets: ClassTestsets,
        ref_filename: str,
        task: str
    ) -> None:
        super().__init__(metric, metrics, available_metrics, results, collection_testsets, ref_filename, task)
    
    def display_plots(self) -> None:
        testset = self.collection_testsets.testsets[self.ref_filename]
        labels = self.collection_testsets.labels
        names_of_systems = self.collection_testsets.names_of_systems()

        st.header(":blue[Confusion Matrix]")
        system_name = st.selectbox(
            "Select the system:",
            names_of_systems,
            index=0
        )

        system = self.collection_testsets.system_name_id(system_name)

        st.subheader("Confusion Matrix of :blue[" + system_name + "]")
        overall_confusion_matrix_table(testset,system,labels,system_name)

        st.subheader("Confusion Matrix of :blue[" + system_name + "] focused on one label")
        label = st.selectbox(
            "Select the label:",
            list(labels),
            index=0,
            key = "confusion_matrix"
        )
        singular_confusion_matrix_table(testset,system,labels,label,system_name)

        st.header(":blue[Analysis Of Each Label]")
        analysis_labels(self.results[self.metric], self.collection_testsets.names_of_systems(), labels)

        st.header(":blue[Examples That Are Incorrectly Labelled]")
        system_name = st.selectbox(
            "Select the system:",
            names_of_systems,
            index=0,
            key = "examples"
        )

        system = self.collection_testsets.system_name_id(system_name)
        ref_id = self.collection_testsets.refs_indexes[self.ref_filename]

        num = 'num_' + system + "_" + ref_id
        incorrect_ids = 'incorrect_ids_' + system + "_" + ref_id
        table = 'tables_' + system + "_" + ref_id
        num_incorrect_ids = 'num_incorrect_ids_' + system + "_" + ref_id
        
        if num not in st.session_state:
            st.session_state[num] = int(len(testset.ref)/4) + 1
        if incorrect_ids not in st.session_state:
            st.session_state[incorrect_ids] = []
        if table not in st.session_state:
            st.session_state[table] = []
        if num_incorrect_ids  not in st.session_state:
            st.session_state[num_incorrect_ids] = 0

        df = incorrect_examples(testset, system, st.session_state[num], st.session_state[incorrect_ids],
                st.session_state[table])

        if df is not None:
            st.dataframe(df)
            old_num_incorrect_ids = st.session_state[num_incorrect_ids]
            new_num_incorrect_ids = len(st.session_state[incorrect_ids])

            def callback():
                st.session_state[num] +=  st.session_state[num] 
                st.session_state[num_incorrect_ids] = new_num_incorrect_ids

            if old_num_incorrect_ids != new_num_incorrect_ids:
                _, middle, _ = st.columns(3)
                middle.button("More examples", on_click=callback)
            else:
                st.warning("There are no more examples that are incorrectly labeled.")
        else:
            st.warning("There are no examples that are incorrectly labelled.")

    def display_plots_cli(self, saving_dir:str) -> None:
        testset = self.collection_testsets.testsets[self.ref_filename]
        labels = self.collection_testsets.labels
        systems_names = self.collection_testsets.systems_names

        analysis_labels(self.results[self.metric], self.collection_testsets.names_of_systems(), 
            labels, saving_dir)
        
        for sys_id, sys_name in systems_names.items():
            output_file = saving_dir + sys_name
            if not os.path.exists(output_file):
                os.makedirs(output_file)            
            overall_confusion_matrix_table(testset,sys_id,labels,sys_name,output_file)

            num = int(len(testset.ref)/4) + 1
            incorrect_examples(testset,sys_id,num,[],[], output_file)

            label_file = output_file + "/" + "singular_confusion_matrix"
            if not os.path.exists(label_file):
                os.makedirs(label_file)  
            for label in labels:
                singular_confusion_matrix_table(testset,sys_id,labels,label,sys_name,label_file)
        
        
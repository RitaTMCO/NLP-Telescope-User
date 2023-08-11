import click
import os
import streamlit as st

from typing import Tuple
from telescope.utils import PATH_DOWNLOADED_PLOTS
from telescope.tasks.task import Task
from telescope.collection_testsets import CollectionTestsets, ClassTestsets
from telescope.metrics import AVAILABLE_CLASSIFICATION_METRICS
from telescope.filters import AVAILABLE_CLASSIFICATION_FILTERS
from telescope.bias_evaluation import AVAILABLE_CLASSIFICATION_BIAS_EVALUATIONS
from telescope.universal_metrics import AVAILABLE_CLASSIFICATION_UNIVERSAL_METRICS
from telescope.plotting import (
    confusion_matrix_of_system,
    confusion_matrix_focused_on_one_label,
    analysis_labels,
    incorrect_examples,
    rates_table,
    export_dataframe
)


class Classification(Task):
    name = "classification"
    metrics = AVAILABLE_CLASSIFICATION_METRICS
    filters = AVAILABLE_CLASSIFICATION_FILTERS
    bias_evaluations = AVAILABLE_CLASSIFICATION_BIAS_EVALUATIONS
    universal_metrics = AVAILABLE_CLASSIFICATION_UNIVERSAL_METRICS

    @staticmethod
    def input_web_interface() -> CollectionTestsets:
        """Web Interface to collect the necessary inputs to realization of the task evaluation."""
        class_testset = ClassTestsets.read_data()
        return class_testset

    @staticmethod
    def input_cli_interface(source:click.File, system_names_file:click.File, systems_output:Tuple[click.File], reference:Tuple[click.File], 
                      extra_info:str="",labels_file:click.File=None) -> CollectionTestsets:
        """CLI Interface to collect the necessary inputs to realization of the task evaluation."""
        return  ClassTestsets.read_data_cli(source, system_names_file, systems_output, reference, extra_info, labels_file)
    
    @classmethod
    def plots_web_interface(cls, metric:str, results:dict, collection_testsets: CollectionTestsets, ref_filename: str) -> None:
        """Web Interfave to display the plots"""

        path = PATH_DOWNLOADED_PLOTS  + collection_testsets.task + "/" + collection_testsets.src_name + "/" +  ref_filename + "/" 

        ref_id = collection_testsets.refs_indexes[ref_filename]
        testset = collection_testsets.testsets[ref_filename]
        labels = collection_testsets.labels
        names_of_systems = collection_testsets.names_of_systems()


        #-------------- |Confusion Matrix| --------------------
        st.header(":blue[Confusion Matrix]")
        system_name = st.selectbox(
            "Select the system:",
            names_of_systems,
            index=0
        )
        system = collection_testsets.system_name_id(system_name)

        # Rates
        st.subheader("Rates")
        df_rates = rates_table(labels,testset.ref,testset.systems_output[system])
        st.dataframe(df_rates)
        _,middle,_ = st.columns(3)
        export_dataframe(label="Export rates",path=path, name= system_name.replace(" ", "_") + "_rates.csv", dataframe=df_rates,column=middle)

        # Overall Confusion Matrix

        st.subheader("Confusion Matrix of :blue[" + system_name + "]")
        confusion_matrix_of_system(testset.ref,testset.systems_output[system],labels,system_name)


        # Singular Confusion Matrix
        st.subheader("Confusion Matrix of :blue[" + system_name + "] focused on one label")
        label = st.selectbox(
            "Select the label:",
            list(labels),
            index=0,
            key = "confusion_matrix"
        )
        confusion_matrix_focused_on_one_label(testset.ref,testset.systems_output[system],label,labels,system_name)

        _,middle,_ = st.columns(3)
        if middle.button('Download all confusion matrices of all systems'):
            for sys_name in collection_testsets.names_of_systems():
                path_dir = path + sys_name.replace(" ", "_") + "/"
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)  
                sys = collection_testsets.system_name_id(sys_name)
                confusion_matrix_of_system(testset.ref,testset.systems_output[sys],labels,sys_name,path_dir)
                for l in labels:
                    confusion_matrix_focused_on_one_label(testset.ref,testset.systems_output[sys],l,labels,sys_name, path_dir)


        #-------------- |Analysis Of Each Label| --------------------
        st.header(":blue[Analysis Of Each Label]")
        analysis_labels(results[metric], collection_testsets.names_of_systems(), labels)

        _,middle,_ = st.columns(3)
        if middle.button('Download the plot'):
            if not os.path.exists(path):
                os.makedirs(path)  
            analysis_labels(results[metric], collection_testsets.names_of_systems(), labels, path)

        
        #-------------- |Examples| --------------------
        st.header(":blue[Examples That Are Incorrectly Labelled]")
        system_name = st.selectbox(
            "Select the system:",
            names_of_systems,
            index=0,
            key = "examples"
        )

        system = collection_testsets.system_name_id(system_name)

        num = 'num_' + system_name + system + "_" + ref_id
        incorrect_ids = 'incorrect_ids_' + system_name + system + "_" + ref_id
        table = 'tables_' + system_name + system + "_" + ref_id
        num_incorrect_ids = 'num_incorrect_ids_' + system_name + system + "_" + ref_id
        click = "click_" + system_name + system + "bias_evaluation"
        
        if num not in st.session_state:
            if len(testset.ref) <= 55:
                st.session_state[num] = int(len(testset.ref)/4) + 1
            else:
                st.session_state[num] = 5
        if incorrect_ids not in st.session_state:
            st.session_state[incorrect_ids] = []
        if table not in st.session_state:
            st.session_state[table] = []
        if num_incorrect_ids  not in st.session_state:
            st.session_state[num_incorrect_ids] = 0
        if click not in st.session_state:
            st.session_state[click] =  1

        if st.session_state[click] == 1:
            num_segments = len(collection_testsets.testsets[ref_filename])
            df = incorrect_examples(testset, system, st.session_state[num], st.session_state[incorrect_ids],st.session_state[table], [i for i in range(num_segments)])
        else:
            df = incorrect_examples(testset, system, st.session_state[num], st.session_state[incorrect_ids],st.session_state[table])
    
        if df is not None:
            st.dataframe(df)
            export_dataframe(label="Export incorrect examples", path=path, name= system_name + "_incorrect-examples.csv", dataframe=df)
            
            old_num_incorrect_ids = st.session_state[num_incorrect_ids]
            new_num_incorrect_ids = len(st.session_state[incorrect_ids])

            def callback():
                st.session_state[num] +=  st.session_state[num] 
                st.session_state[num_incorrect_ids] = new_num_incorrect_ids
                st.session_state[click] +=  1

            if old_num_incorrect_ids != new_num_incorrect_ids:
                _, middle, _ = st.columns(3)
                middle.button("More examples", on_click=callback)
            else:
                st.warning("There are no more examples that are incorrectly labeled.")
        else:
            st.warning("There are no examples that are incorrectly labelled.")


    @classmethod
    def plots_cli_interface(cls, metric:str, results:dict, collection_testsets: CollectionTestsets, ref_filename: str, 
                            saving_dir:str) -> None:
        """CLI Interfave to display the plots"""

        testset = collection_testsets.testsets[ref_filename]
        labels = collection_testsets.labels
        systems_names = collection_testsets.systems_names

        analysis_labels(results[metric], collection_testsets.names_of_systems(), labels, saving_dir)
        
        for sys_id, sys_name in systems_names.items():
            output_file = saving_dir + sys_name
            if not os.path.exists(output_file):
                os.makedirs(output_file)            
            confusion_matrix_of_system(testset.ref,testset.systems_output[sys_id],labels,sys_name,output_file)
            rates_table(labels,testset.ref,testset.systems_output[sys_id],output_file)

            num = 15
            incorrect_examples(testset,sys_id,num,[],[],[], output_file)

            label_file = output_file + "/" + "singular_confusion_matrix"
            if not os.path.exists(label_file):
                os.makedirs(label_file)  
            for label in labels:
                confusion_matrix_focused_on_one_label(testset.ref, testset.systems_output[sys_id], label,labels,sys_name,label_file)

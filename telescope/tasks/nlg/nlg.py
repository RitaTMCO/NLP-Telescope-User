import streamlit as st
import pandas as pd
import numpy as np
import os

from telescope.utils import PATH_DOWNLOADED_PLOTS
from telescope.tasks.task import Task
from telescope.collection_testsets import CollectionTestsets, NLGTestsets
from telescope.metrics import AVAILABLE_NLG_METRICS
from telescope.filters import AVAILABLE_NLG_FILTERS
from telescope.bias_evaluation import AVAILABLE_NLG_BIAS_EVALUATIONS
from telescope.universal_metrics import AVAILABLE_NLG_UNIVERSAL_METRICS
from telescope.plotting import (
    plot_bootstraping_result,
    plot_bucket_multiple_comparison,
    plot_multiple_distributions,
    plot_multiple_segment_comparison,
    sentences_similarity,
    export_dataframe
)

class NLG(Task):
    name = "NLG"
    metrics = AVAILABLE_NLG_METRICS 
    filters = AVAILABLE_NLG_FILTERS
    bias_evaluations = AVAILABLE_NLG_BIAS_EVALUATIONS
    universal_metrics = AVAILABLE_NLG_UNIVERSAL_METRICS
    bootstrap = True
    segment_result_source = False
    sentences_similarity = False


    @classmethod
    def plots_web_interface(cls, metric:str, results:dict, collection_testsets: CollectionTestsets, ref_filename: str, 
                            metrics:list, available_metrics:dict, num_samples: int, sample_ratio: float) -> None:
        """Web Interfave to display the plots"""

        path = PATH_DOWNLOADED_PLOTS  + collection_testsets.task + "/" + collection_testsets.src_name + "/" +  ref_filename + "/"  

        # --------------- |Source Sentences Similarity| ----------------
        if cls.sentences_similarity and (collection_testsets.target_language == "pt" or collection_testsets.target_language == "en"):
            st.header(":blue[Similar Source Sentences]")
            system_name = st.selectbox(
                "Select the system:",
                collection_testsets.names_of_systems(),
                index=0,
                key = "sentences_similarity"
            )
            system_id = collection_testsets.system_name_id(system_name)
            output = collection_testsets.testsets[ref_filename].systems_output[system_id]
            df,min_value,max_value = sentences_similarity(collection_testsets.testsets[ref_filename].src, output, collection_testsets.target_language)
            if df is not None:
                name = system_name + "_" + str(min_value) + "-" + str(max_value) + "_similar-source-sentences.csv"
                st.dataframe(df)
                export_dataframe(label="Export similar source sentences", path=path, name = name, dataframe=df)
            else:
                st.warning("Segments not found")

        # -------------- |Error-type analysis| ------------------
        if metric == "COMET" or metric == "BERTScore":
            st.text("\n")
            st.header(":blue[Error-type analysis:]")
            
            left = plot_bucket_multiple_comparison(results[metric], collection_testsets.names_of_systems())

            if left.button('Download the error-type analysis'):
                if not os.path.exists(path):
                    os.makedirs(path)  
                plot_bucket_multiple_comparison(results[metric], collection_testsets.names_of_systems(),path)
        
        # -------------- | Distribution of segment-level scores| -----------
        if len(collection_testsets.testsets[ref_filename]) > 1 and plot_multiple_distributions(results[metric], collection_testsets.names_of_systems(), test=True):

            st.text("\n")
            st.header(":blue[Distribution of segment-level scores:]")
            st.markdown("This displot shows the distribution of segment-level scores. It is composed of histogram, kernel density estimation curve and rug plot.")

            plot_multiple_distributions(results[metric], collection_testsets.names_of_systems())
            _, middle, _ = st.columns(3)
            if middle.button('Download the distribution of segment-level scores'):
                if not os.path.exists(path):
                    os.makedirs(path)  
                plot_multiple_distributions(results[metric], collection_testsets.names_of_systems(),path)
            
        
        # -------------- |Pairwise comparison| -----------------
        if len(results[metric].systems_metric_results) > 1:

            st.text("\n")
            st.header(":blue[Pairwise comparison:]")
            left, right = st.columns(2)
            system_x_name = left.selectbox(
                "Select the system x:",
                collection_testsets.names_of_systems(),
                index=0,
                key = ref_filename + "_1"
            )
            system_y_name = right.selectbox(
                "Select the system y:",
                collection_testsets.names_of_systems(),
                index=1,
                key = ref_filename + "_2"
            )

            if system_x_name == system_y_name:
                st.warning("The system x cannot be the same as system y")
            else:
                system_x_id = collection_testsets.system_name_id(system_x_name)
                system_x = [system_x_id, system_x_name]
                system_y_id = collection_testsets.system_name_id(system_y_name)
                system_y = [system_y_id, system_y_name]
        

                #Segment-level comparison
                st.subheader("Segment-level comparison:")
                plot_multiple_segment_comparison(results[metric],system_x,system_y,cls.segment_result_source)
                _, middle, _ = st.columns(3)
                if middle.button('Download the segment-level comparison'):
                    if not os.path.exists(path):
                        os.makedirs(path)  
                    plot_multiple_segment_comparison(results[metric],system_x,system_y,cls.segment_result_source, path)

                #Bootstrap Resampling

                name = system_x_name + "-" + system_y_name + "_bootstrap_results.csv"

                if 'data_boostrap' not in st.session_state:
                    st.session_state.data_boostrap = None
                
                if st.session_state.get("export-" + name):
                    if not os.path.exists(path):
                        os.makedirs(path)  
                    st.session_state.data_boostrap.to_csv(path + "/" + name)

                _, middle, _ = st.columns(3)
                    
                if middle.button("Perform Bootstrap Resampling", key="button-bootstrap"):
                    st.warning(
                        "Running metrics for {} partitions of size {}".format(
                            num_samples, sample_ratio * len(collection_testsets.testsets[ref_filename])
                        )
                    )
                    st.subheader("Bootstrap resampling results:")
                    list_df = list()
                    with st.spinner("Running bootstrap resampling..."):
                        for metric in metrics:
                            bootstrap_result = available_metrics[metric].multiple_bootstrap_resampling(
                                collection_testsets.testsets[ref_filename], int(num_samples), 
                                sample_ratio, system_x_id, system_y_id, collection_testsets.target_language, results[metric])
                            df = plot_bootstraping_result(bootstrap_result)
                            list_df.append(df)
                        _, middle, _ = st.columns(3)
                        name = system_x_name + "-" + system_y_name + "_bootstrap_results.csv"
                        st.session_state.data_boostrap = pd.concat(list_df)
                        export_dataframe(label="Export bootstrap resampling results", path=path, name= name, dataframe=st.session_state.data_boostrap,column=middle)
    @classmethod
    def plots_cli_interface(cls, metric:str, results:dict, collection_testsets: CollectionTestsets, ref_filename: str, 
                            saving_dir:str, x_id:str ,y_id:str) -> None:
        """CLI Interfave to display the plots"""

        if cls.sentences_similarity:
            for system_name in collection_testsets.names_of_systems():
                system_id = collection_testsets.system_name_id(system_name)
                output = collection_testsets.testsets[ref_filename].systems_output[system_id]
                output_file = saving_dir + system_name
                if not os.path.exists(output_file):
                    os.makedirs(output_file)  
                df = sentences_similarity(collection_testsets.testsets[ref_filename].src, output, collection_testsets.target_language,output_file)
                if df is None:
                    os.rmdir(output_file)

        if metric == "COMET" or metric == "BERTScore":
            plot_bucket_multiple_comparison(results[metric], collection_testsets.names_of_systems(), saving_dir)
        
        if len(collection_testsets.testsets[ref_filename]) > 1 and plot_multiple_distributions(results[metric], collection_testsets.names_of_systems(), test=True):
            plot_multiple_distributions(results[metric], collection_testsets.names_of_systems(), saving_dir)
        
        if len(collection_testsets.systems_ids.values()) > 1: 
            x = [x_id,collection_testsets.systems_names[x_id]]
            y = [y_id,collection_testsets.systems_names[y_id]]
            plot_multiple_segment_comparison(results[metric],x,y,cls.segment_result_source,saving_dir)
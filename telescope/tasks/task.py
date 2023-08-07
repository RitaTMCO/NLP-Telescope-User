import abc
import click

from telescope.collection_testsets import CollectionTestsets
from telescope.metrics import AVAILABLE_NLP_METRICS
from telescope.filters import AVAILABLE_NLP_FILTERS
from telescope.bias_evaluation import AVAILABLE_NLP_BIAS_EVALUATIONS
from telescope.universal_metrics import AVAILABLE_NLP_UNIVERSAL_METRICS
from typing import Tuple

class Task(metaclass=abc.ABCMeta):
    name = None
    metris = AVAILABLE_NLP_METRICS 
    filters = AVAILABLE_NLP_FILTERS
    bias_evaluations = AVAILABLE_NLP_BIAS_EVALUATIONS
    universal_metrics = AVAILABLE_NLP_UNIVERSAL_METRICS
    bootstrap = False

    @staticmethod
    @abc.abstractmethod
    def input_web_interface() -> CollectionTestsets:
        """Web Interface to collect the necessary inputs to realization of the task evaluation."""
        pass
    
    @staticmethod
    @abc.abstractmethod
    def input_cli_interface(source:click.File, system_names_file:click.File, systems_output:Tuple[click.File], reference:Tuple[click.File], 
                      extra_info:str) -> CollectionTestsets:
        """CLI Interface to collect the necessary inputs to realization of the task evaluation."""
        pass

        
    @classmethod
    @abc.abstractmethod
    def plots_web_interface(cls, metric:str, results:dict, collection_testsets: CollectionTestsets, ref_filename: str) -> None:
        """Web Interface to display the plots"""
        pass
    
    @classmethod
    @abc.abstractmethod
    def plots_cli_interface(cls, metric:str, results:dict, collection_testsets: CollectionTestsets, ref_filename: str, 
                            saving_dir:str) -> None:
        """CLI Interfave to display the plots"""
        pass
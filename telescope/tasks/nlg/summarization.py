import click

from typing import Tuple

from telescope.tasks.nlg.nlg import NLG
from telescope.collection_testsets import CollectionTestsets, SummTestsets
from telescope.metrics import AVAILABLE_SUMMARIZATION_METRICS
from telescope.filters import AVAILABLE_SUMMARIZATION_FILTERS
from telescope.bias_evaluation import AVAILABLE_SUMMARIZATION_BIAS_EVALUATIONS
from telescope.universal_metrics import AVAILABLE_SUMMARIZATION_UNIVERSAL_METRICS

class Summarization(NLG):
    name = "summarization"
    metrics = AVAILABLE_SUMMARIZATION_METRICS
    filters = AVAILABLE_SUMMARIZATION_FILTERS
    bias_evaluations = AVAILABLE_SUMMARIZATION_BIAS_EVALUATIONS
    universal_metrics = AVAILABLE_SUMMARIZATION_UNIVERSAL_METRICS
    sentences_similarity = True

    @staticmethod
    def input_web_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        summ_testset = SummTestsets.read_data()
        return summ_testset

    @staticmethod
    def input_cli_interface(source:click.File, system_names_file:click.File, systems_output:Tuple[click.File], reference:Tuple[click.File], 
                      extra_info:str="",labels_file:click.File=None) -> CollectionTestsets:
        """CLI Interface to collect the necessary inputs to realization of the task evaluation."""
        target_language = extra_info
        return  SummTestsets.read_data_cli(source, system_names_file, systems_output, reference, "X-" + target_language)
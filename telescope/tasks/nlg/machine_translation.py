import click

from typing import Tuple
from telescope.tasks.nlg.nlg import NLG
from telescope.collection_testsets import CollectionTestsets, MTTestsets
from telescope.metrics import AVAILABLE_MT_METRICS
from telescope.filters import AVAILABLE_MT_FILTERS
from telescope.bias_evaluation import AVAILABLE_MT_BIAS_EVALUATIONS
from telescope.universal_metrics import AVAILABLE_MT_UNIVERSAL_METRICS

class MachineTranslation(NLG):
    name = "machine-translation"
    metrics = AVAILABLE_MT_METRICS
    filters = AVAILABLE_MT_FILTERS
    bias_evaluations = AVAILABLE_MT_BIAS_EVALUATIONS
    universal_metrics = AVAILABLE_MT_UNIVERSAL_METRICS
    segment_result_source = True

    @staticmethod
    def input_web_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        mt_testset = MTTestsets.read_data()
        return mt_testset
    
    @staticmethod
    def input_cli_interface(source:click.File, system_names_file:click.File, systems_output:Tuple[click.File], reference:Tuple[click.File], 
                      extra_info:str="",labels_file:click.File=None) -> CollectionTestsets:
        """CLI Interface to collect the necessary inputs to realization of the task evaluation."""
        target_language = extra_info
        return  MTTestsets.read_data_cli(source, system_names_file, systems_output, reference, "X-" + target_language)
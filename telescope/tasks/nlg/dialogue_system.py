import click

from typing import Tuple
from telescope.tasks.nlg.nlg import NLG
from telescope.collection_testsets import CollectionTestsets, DialogueTestsets
from telescope.metrics import AVAILABLE_DIALOGUE_METRICS
from telescope.filters import AVAILABLE_DIALOGUE_FILTERS
from telescope.bias_evaluation import AVAILABLE_DIALOGUE_BIAS_EVALUATIONS
from telescope.universal_metrics import AVAILABLE_DIALOGUE_UNIVERSAL_METRICS

class DialogueSystem(NLG):
    name = "dialogue-system"
    metrics = AVAILABLE_DIALOGUE_METRICS
    filters = AVAILABLE_DIALOGUE_FILTERS
    bias_evaluations = AVAILABLE_DIALOGUE_BIAS_EVALUATIONS
    universal_metrics = AVAILABLE_DIALOGUE_UNIVERSAL_METRICS
    sentences_similarity = True

    @staticmethod
    def input_web_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        dialogue_testset = DialogueTestsets.read_data()
        return dialogue_testset
    
    @staticmethod
    def input_cli_interface(source:click.File, system_names_file:click.File, systems_output:Tuple[click.File], reference:Tuple[click.File], 
                      extra_info:str="",labels_file:click.File=None) -> CollectionTestsets:
        """CLI Interface to collect the necessary inputs to realization of the task evaluation."""
        target_language = extra_info
        return  DialogueTestsets.read_data_cli(source, system_names_file, systems_output, reference, "X-" + target_language)
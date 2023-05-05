from telescope.tasks.nlg.nlg import NLG
from telescope.collection_testsets import CollectionTestsets, SummTestsets
from telescope.metrics import AVAILABLE_SUMMARIZATION_METRICS
from telescope.filters import AVAILABLE_SUMMARIZATION_FILTERS

class Summarization(NLG):
    name = "summarization"
    metrics = AVAILABLE_SUMMARIZATION_METRICS
    filters = AVAILABLE_SUMMARIZATION_FILTERS

    @staticmethod
    def input_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        summ_testset = SummTestsets.read_data()
        return summ_testset
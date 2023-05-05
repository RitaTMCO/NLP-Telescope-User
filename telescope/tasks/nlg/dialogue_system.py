from telescope.tasks.nlg.nlg import NLG
from telescope.collection_testsets import CollectionTestsets, DialogueTestsets
from telescope.metrics import AVAILABLE_DIALOGUE_METRICS
from telescope.filters import AVAILABLE_DIALOGUE_FILTERS

class DialogueSystem(NLG):
    name = "dialogue-system"
    metrics = AVAILABLE_DIALOGUE_METRICS
    filters = AVAILABLE_DIALOGUE_FILTERS

    @staticmethod
    def input_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        dialogue_testset = DialogueTestsets.read_data()
        return dialogue_testset
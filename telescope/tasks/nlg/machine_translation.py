from telescope.tasks.nlg.nlg import NLG
from telescope.collection_testsets import CollectionTestsets, MTTestsets
from telescope.metrics import AVAILABLE_MT_METRICS
from telescope.filters import AVAILABLE_MT_FILTERS

class MachineTranslation(NLG):
    name = "machine-translation"
    metrics = AVAILABLE_MT_METRICS
    filters = AVAILABLE_MT_FILTERS

    @staticmethod
    def input_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        mt_testset = MTTestsets.read_data()
        return mt_testset
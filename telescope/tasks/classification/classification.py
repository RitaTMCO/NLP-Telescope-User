from telescope.tasks.task import Task
from telescope.collection_testsets import CollectionTestsets, ClassTestsets
from telescope.plot import ClassificationPlot
from telescope.metrics import AVAILABLE_CLASSIFICATION_METRICS
from telescope.filters import AVAILABLE_CLASSIFICATION_FILTERS

class Classification(Task):
    name = "classification"
    metrics = AVAILABLE_CLASSIFICATION_METRICS
    filters = AVAILABLE_CLASSIFICATION_FILTERS

    @staticmethod
    def input_interface() -> CollectionTestsets:
        """Interface to collect the necessary inputs to realization of the task evaluation."""
        class_testset = ClassTestsets.read_data()
        return class_testset
    
    @classmethod
    def plots_interface(cls, metric:str, metrics:list, available_metrics:dict, results:dict, 
                        collection_testsets: CollectionTestsets, ref_file: str) -> None:
        """ Interfave to display the plots"""
        return ClassificationPlot(metric,metrics,available_metrics,
                                    results,collection_testsets,ref_file,cls.name).display_plots()
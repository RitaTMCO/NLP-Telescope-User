from telescope.tasks.task import Task
from telescope.collection_testsets import CollectionTestsets
from telescope.plot import NLGPlot
from telescope.metrics import AVAILABLE_NLG_METRICS

class NLG(Task):
    name = None
    metrics = AVAILABLE_NLG_METRICS 

    @classmethod
    def plots_interface(cls, metric:str, metrics:list, available_metrics:dict, results:dict, 
                        collection_testsets: CollectionTestsets, ref_file: str,
                        num_samples: int, sample_ratio: float) -> None:
        """ Interfave to display the plots"""
        return NLGPlot(metric,metrics,available_metrics,results,collection_testsets,
                        ref_file,cls.name,num_samples,sample_ratio).display_plots()


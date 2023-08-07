import statistics
from typing import Dict
from telescope.testset import MultipleTestset
from telescope.universal_metrics.universal_metric import UniversalMetric

class Median(UniversalMetric):

    name = "median"
    title = "Median"

    def universal_score(self,testset:MultipleTestset) -> Dict[str,float]:
        systems_outputs = testset.systems_output
        systems_ids = list(systems_outputs.keys())
        median_scores = {sys_id:[] for sys_id in systems_ids}
    
        for metric_results in list(self.multiple_metrics_results.values()):
            for sys_id, metric_result in metric_results.systems_metric_results.items():
                median_scores[sys_id].append(metric_result.sys_score)
        
        median_scores = {sys_id:statistics.median(scores) for sys_id,scores in median_scores.items()}        
        return median_scores
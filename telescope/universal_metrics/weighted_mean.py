from typing import Dict
from telescope.metrics import METRICS_WEIGHTS
from telescope.metrics.metric import MultipleMetricResults
from telescope.testset import MultipleTestset
from telescope.universal_metrics.universal_metric import UniversalMetric

class WeightedMean(UniversalMetric):

    title = "Weighted Mean"

    def __init__(self, multiple_metrics_results: Dict[str, MultipleMetricResults], name:str, weights:Dict[str,float]={}):
        super().__init__(multiple_metrics_results)
        if name in list(METRICS_WEIGHTS.keys()):
            self.metrics_weight = METRICS_WEIGHTS[name]
        else:
            self.metrics_weight = weights
        self.name = name

    def universal_score(self,testset:MultipleTestset) -> Dict[str,float]:
        systems_outputs = testset.systems_output
        systems_ids = list(systems_outputs.keys())
        metrics = list(self.multiple_metrics_results.keys())
        num_metrics = len(metrics)
        weighted_scores = {sys_id:0.0 for sys_id in systems_ids}
    
        for metric_results in list(self.multiple_metrics_results.values()):
            for sys_id, metric_result in metric_results.systems_metric_results.items():
                if metric_result.metric in list(self.metrics_weight.keys()):
                    weighted_scores[sys_id] += metric_result.sys_score * float(self.metrics_weight[metric_result.metric])
                else:
                    weighted_scores[sys_id] += metric_result.sys_score * 0.0

        weighted_scores = {sys_id:score/num_metrics for sys_id,score in weighted_scores.items()}
        
        return weighted_scores
import abc
from typing import List,Dict
from telescope.metrics.metric import MultipleMetricResults
from telescope.testset import MultipleTestset
from telescope.universal_metrics.universal_metric import UniversalMetric
from telescope.universal_metrics.universal_metric_results import UniversalMetricResult, MultipleUniversalMetricResult

class SocialChoiceTheory(UniversalMetric):

    name = "social-choice-theory"
    title = "Social Choice Theory"


    def borda(self,systems_ids:List[str], ranking_systems_per_metrics: Dict[str,dict]):
        metrics = list(self.multiple_metrics_results.keys())
        sum_scores = {sys_id:0.0 for sys_id in systems_ids}
    
        for metric in metrics:
            ranking_systems = ranking_systems_per_metrics[metric]
            for sys_id in systems_ids:
                sum_scores[sys_id] += ranking_systems[sys_id]["rank"]
        return sum_scores


    def universal_score(self,testset:MultipleTestset) -> Dict[str,float]:
        systems_outputs = testset.systems_output
        systems_ids = list(systems_outputs.keys())
        ranking_systems_per_metrics= {}
    
        for metric_results in list(self.multiple_metrics_results.values()):
            scores = {}
            for sys_id, metric_result in metric_results.systems_metric_results.items():
                scores[sys_id] = metric_result.sys_score
            
            metric = metric_results.metric
            if metric == "TER":
                rank_scores = self.ranking_systems(scores,False)
            else:
                rank_scores = self.ranking_systems(scores)
            ranking_systems_per_metrics[metric] = rank_scores
        
        sum_scores = self.borda(systems_ids,ranking_systems_per_metrics)
            
        return sum_scores